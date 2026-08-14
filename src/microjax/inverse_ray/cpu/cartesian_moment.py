"""Root-free Cartesian-strip moment ICRS for uniform binary sources."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from microjax.point_source import lens_eq
from microjax.poly_solver import poly_roots

from ..geometry.lens import binary_geometry
from .angular_moment import (
    ANGULAR_MOMENT_EXHAUSTED,
    ANGULAR_MOMENT_TOPOLOGY,
    _angular_support_cells_from_trace,
    _uniform_contact_refinement,
    mag_uniform_angular_moment_compact,
    mag_uniform_angular_moment_refined,
)
from .roots import batched_companion_roots, batched_polished_real_companion_roots
from .quadrature import G7_W_ON_GK15, GK15_W, GK15_X
from .sentinel import hidden_caustic_candidate
from .support import trace_binary_source_limb, tracked_limb_neighbors

Array = jnp.ndarray

_SEXTIC_DEGREE = 6
_BERNSTEIN_SAMPLE_FRACTION = np.linspace(0.0, 1.0, _SEXTIC_DEGREE + 1)
_BERNSTEIN_VALUE_MATRIX = np.asarray(
    [
        [
            math.comb(_SEXTIC_DEGREE, index) * fraction**index * (1.0 - fraction) ** (_SEXTIC_DEGREE - index)
            for index in range(_SEXTIC_DEGREE + 1)
        ]
        for fraction in _BERNSTEIN_SAMPLE_FRACTION
    ]
)
_BERNSTEIN_VALUE_TO_COEFFICIENT = np.linalg.inv(_BERNSTEIN_VALUE_MATRIX).T


@jax.custom_jvp
def _attach_implicit_root_jvp(coefficients: Array, roots: Array) -> Array:
    """Keep isolated real roots but differentiate the exact root equation."""

    del coefficients
    return roots


@_attach_implicit_root_jvp.defjvp
def _attach_implicit_root_jvp_rule(primals, tangents):
    coefficients, roots = primals
    coefficient_tangent, _ = tangents
    degree = coefficients.shape[-1] - 1
    derivative = coefficients[:, :-1] * jnp.arange(degree, 0, -1, dtype=coefficients.dtype)
    numerator = jax.vmap(jnp.polyval)(coefficient_tangent, roots)
    denominator = jax.vmap(jnp.polyval)(derivative, roots)
    return roots, -numerator / denominator


class CartesianMomentResult(NamedTuple):
    """Value and diagnostics from one fixed-order Cartesian ICRS pass."""

    magnification: Array
    estimated_error: Array
    n_slices: Array
    invalid_root_count: Array
    ghost_residual_ratio: Array
    limb_topology: Array
    status: Array


class CartesianAdaptiveResult(NamedTuple):
    """Result of the asymmetric sequential Cartesian CPU integrator.

    In the legacy adaptive path, ``status == 0`` is a route-specific numerical
    acceptance flag. In the production one-shot path it means only that no
    structural failure was detected; that path reports ``estimated_error`` as
    NaN and does not estimate the unknown integration error.
    """

    magnification: Array
    estimated_error: Array
    n_slices: Array
    stage: Array
    status: Array


class CartesianTopologyProbe(NamedTuple):
    """Sampled critical-root diagnostics for one strip support."""

    minimum_critical_value: Array
    minimum_interior_critical_value: Array
    root_count_span: Array
    invalid_critical_count: Array


def _normalized_axis(axis: complex | Array, dtype: jnp.dtype) -> Array:
    """Return a unit complex projection axis without introducing NaNs."""

    axis = jnp.asarray(axis, dtype=dtype)
    magnitude = jnp.abs(axis)
    return jnp.where(magnitude > 0.0, axis / magnitude, jnp.asarray(1.0, dtype=dtype))


def _fixed_polynomial_product_ascending(left: Array, right: Array) -> Array:
    """Multiply two tiny ascending polynomials with statically unrolled sums."""

    terms = []
    for output_index in range(left.shape[0] + right.shape[0] - 1):
        products = [
            left[left_index] * right[output_index - left_index]
            for left_index in range(left.shape[0])
            if 0 <= output_index - left_index < right.shape[0]
        ]
        value = products[0]
        for product in products[1:]:
            value = value + product
        terms.append(value)
    return jnp.stack(terms)


def binary_line_level_set_coefficients(
    offset: complex | Array,
    direction: complex | Array,
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
) -> Array:
    """Return coefficients of ``H(offset + t * direction)`` for real ``t``."""

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    offset = jnp.asarray(offset, dtype=w_center.dtype)
    direction = jnp.asarray(direction, dtype=w_center.dtype)
    rho = jnp.asarray(rho, dtype=real_dtype)
    lens = binary_geometry(s, q)

    base_bar = jnp.conjugate(offset) - lens.shifted
    direction_bar = jnp.conjugate(direction)
    denominator = jnp.asarray(
        [
            base_bar**2 - lens.a**2,
            2.0 * base_bar * direction_bar,
            direction_bar**2,
        ],
        dtype=w_center.dtype,
    )
    deflection_constant = -lens.shifted + lens.a * (2.0 * lens.e1 - 1.0)
    numerator = _fixed_polynomial_product_ascending(
        jnp.asarray([offset - w_center, direction], dtype=w_center.dtype),
        denominator,
    ) - jnp.asarray(
        [
            jnp.conjugate(offset) + deflection_constant,
            direction_bar,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        dtype=w_center.dtype,
    )
    numerator_square = _fixed_polynomial_product_ascending(numerator, jnp.conjugate(numerator))
    denominator_square = _fixed_polynomial_product_ascending(denominator, jnp.conjugate(denominator))
    ascending = jnp.real(numerator_square - rho**2 * jnp.pad(denominator_square, (0, 2)))
    descending = ascending[::-1]
    scale = jnp.max(jnp.abs(descending))
    return descending / jnp.maximum(scale, jnp.finfo(real_dtype).tiny)


def _strip_widths(
    coefficients: Array,
    *,
    continuation: bool | str = False,
    ordinate_bound: Array | None = None,
    source_radius: Array | None = None,
) -> tuple[Array, Array]:
    def bernstein_with_overflow_repair(
        max_depth: int,
        work_capacity: int = 10,
        compact_lookup: bool = False,
    ) -> tuple[Array, Array]:
        width, invalid = _bernstein_strip_widths(
            coefficients,
            ordinate_bound=ordinate_bound,
            max_depth=max_depth,
            capacity=work_capacity,
            compact_lookup=compact_lookup,
        )

        # Ten work slots cover the common six-real-root geometry with less
        # scalar CPU work than the historical twelve-slot buffer.  A root
        # landing almost exactly on a subdivision boundary can temporarily
        # require all twelve children, however.  Repair only those rare strip
        # polynomials with an independent companion solve instead of making
        # every strip pay for the worst case.
        def repair_one(inputs):
            polynomial, current_width, did_overflow = inputs

            def repair(_):
                repaired_width, repaired_invalid = _companion_strip_widths(polynomial[None, :])
                return repaired_width[0], repaired_invalid[0]

            return jax.lax.cond(
                did_overflow != 0,
                repair,
                lambda _: (current_width, jnp.int32(0)),
                operand=None,
            )

        return jax.lax.cond(
            jnp.any(invalid != 0),
            lambda _: jax.lax.map(repair_one, (coefficients, width, invalid)),
            lambda _: (width, jnp.zeros_like(invalid)),
            operand=None,
        )

    if isinstance(continuation, str) and continuation.startswith("sampled"):
        if ordinate_bound is None:
            raise ValueError("sampled root isolation requires ordinate_bound")
        grid_text = continuation.removeprefix("sampled")
        n_grid = int(grid_text) if grid_text else 32
        return _sampled_strip_widths(
            coefficients,
            ordinate_bound=ordinate_bound,
            n_grid=n_grid,
            n_refine=8,
        )
    if continuation in (
        "bernstein_adaptive",
        "bernstein_lean",
        "bernstein_lookup",
        "bernstein_lookup_no_repair",
        "bernstein_clip_no_repair",
        "bernstein_lookup_full",
        "bernstein_capacity8",
    ):
        if ordinate_bound is None or source_radius is None:
            raise ValueError("adaptive Bernstein subdivision requires ordinate_bound and source_radius")

        if continuation in (
            "bernstein_lean",
            "bernstein_capacity8",
            "bernstein_lookup",
            "bernstein_lookup_no_repair",
            "bernstein_clip_no_repair",
            "bernstein_lookup_full",
        ):
            work_capacity = 8
        else:
            work_capacity = 10

        def evaluate_bernstein(depth):
            if continuation == "bernstein_clip_no_repair":
                return _bernstein_strip_widths(
                    coefficients,
                    ordinate_bound=ordinate_bound,
                    max_depth=depth,
                    capacity=work_capacity,
                    clip_children=True,
                )
            if continuation == "bernstein_lookup_no_repair":
                return _bernstein_strip_widths(
                    coefficients,
                    ordinate_bound=ordinate_bound,
                    max_depth=depth,
                    capacity=work_capacity,
                    compact_lookup=True,
                )
            if continuation in ("bernstein_lookup", "bernstein_lookup_full"):
                return bernstein_with_overflow_repair(depth, work_capacity, compact_lookup=True)
            return bernstein_with_overflow_repair(depth, work_capacity)

        depth_offset = (
            1
            if continuation
            in (
                "bernstein_lean",
                "bernstein_lookup",
                "bernstein_lookup_no_repair",
                "bernstein_clip_no_repair",
            )
            else 0
        )

        def depth10(_):
            return evaluate_bernstein(10 - depth_offset)

        def depth12(_):
            return evaluate_bernstein(12 - depth_offset)

        def depth14(_):
            return evaluate_bernstein(14 - depth_offset)

        def depth16(_):
            return evaluate_bernstein(16 - depth_offset)

        # Below rho ~= 4e-4 the narrowest image strips no longer scale safely
        # with the old depth-15 floor.  Increasing n_limb cannot repair this:
        # the missing width is created inside a strip, not on the traced limb.
        # Keep deeper tiers static so XLA still emits bounded work, and give
        # the optional 1e-4 path two additional bisections.
        small_depth = 20 if continuation == "bernstein_lookup_full" else 18
        very_small_depth = 22 if continuation == "bernstein_lookup_full" else 20

        def depth_small(_):
            return evaluate_bernstein(small_depth)

        def depth_very_small(_):
            return evaluate_bernstein(very_small_depth)

        return jax.lax.cond(
            source_radius >= 0.025,
            depth10,
            lambda _: jax.lax.cond(
                source_radius >= 0.00625,
                depth12,
                lambda _: jax.lax.cond(
                    source_radius >= 0.0015625,
                    depth14,
                    lambda _: jax.lax.cond(
                        source_radius >= 0.000390625,
                        depth16,
                        lambda _: jax.lax.cond(
                            source_radius >= 0.00009765625,
                            depth_small,
                            depth_very_small,
                            operand=None,
                        ),
                        operand=None,
                    ),
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )
    if isinstance(continuation, str) and continuation.startswith("bernstein"):
        if ordinate_bound is None:
            raise ValueError("Bernstein subdivision requires ordinate_bound")
        depth_text = continuation.removeprefix("bernstein")
        max_depth = int(depth_text) if depth_text else 20
        return bernstein_with_overflow_repair(max_depth)
    if isinstance(continuation, str) and continuation.startswith("aberth"):
        iteration_text = continuation.removeprefix("aberth")
        iterations = int(iteration_text) if iteration_text else 4
        roots = _continued_companion_roots(coefficients, iterations=iterations)
        return _widths_from_roots(coefficients, roots)
    if continuation == "ea_independent":
        # GPU-style cold Ehrlich--Aberth solve at every strip node.  Unlike
        # the historical ``aberth`` mode, this does not continue roots from a
        # neighbouring strip, so branch identity is never a correctness
        # assumption.  The mode is kept explicit while it is benchmarked
        # against Bernstein isolation on the current 64-point CPU schedule.
        return _widths_from_roots(coefficients, poly_roots(coefficients))
    if continuation == "ea_scaled":
        if ordinate_bound is None:
            raise ValueError("scaled EA requires ordinate_bound")
        # Solve in t=y/B rather than the physical ordinate y.  Thin images can
        # otherwise produce a very large Cauchy initialization circle even
        # though every physical real boundary lies in [-B, B].  The change of
        # variable preserves all six roots and remains independent per strip.
        bound = jnp.asarray(ordinate_bound, dtype=coefficients.dtype)
        degree = coefficients.shape[-1] - 1
        powers = jnp.arange(degree, -1, -1, dtype=coefficients.dtype)
        scaled = coefficients * bound**powers
        scale = jnp.max(jnp.abs(scaled), axis=-1, keepdims=True)
        scaled = scaled / jnp.maximum(
            scale, jnp.finfo(coefficients.dtype).tiny
        )
        roots = poly_roots(scaled) * bound
        return _widths_from_roots(coefficients, roots)
    if isinstance(continuation, str) and continuation.startswith("ea_fixed"):
        if ordinate_bound is None:
            raise ValueError("fixed EA requires ordinate_bound")
        iteration_text = continuation.removeprefix("ea_fixed")
        iterations = int(iteration_text) if iteration_text else 32
        roots = _fixed_independent_ea_roots(
            coefficients,
            ordinate_bound,
            iterations=iterations,
        )
        return _widths_from_roots(coefficients, roots)
    if continuation == "companion":
        return _companion_strip_widths(coefficients)
    if continuation:
        roots = _continued_companion_roots(coefficients)
        return _widths_from_roots(coefficients, roots)
    return _companion_strip_widths(coefficients)


def _widths_from_roots(coefficients: Array, roots: Array) -> tuple[Array, Array]:
    """Select real boundary roots and sum negative polynomial intervals."""

    real = roots.real
    scale = jnp.maximum(1.0, jnp.abs(real))
    residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    nearly_real = jnp.abs(roots.imag) <= 2.0e-7 * scale
    valid = nearly_real & (residual <= 2.0e-7)
    ordinates = jnp.sort(jnp.where(valid, real, jnp.inf), axis=-1)
    maximum = jnp.max(jnp.where(valid, jnp.abs(real), 0.0), axis=-1)
    outer = jnp.maximum(maximum + 1.0, 1.0)
    clean = jnp.where(jnp.isfinite(ordinates), ordinates, outer[:, None])
    boundaries = jnp.concatenate((-outer[:, None], clean, outer[:, None]), axis=-1)
    lower = boundaries[:, :-1]
    upper = boundaries[:, 1:]
    midpoint = 0.5 * (lower + upper)
    values = jax.vmap(jax.vmap(jnp.polyval, in_axes=(None, 0)))(coefficients, midpoint)
    inside = values <= 0.0
    widths = jnp.sum(jnp.where(inside, upper - lower, 0.0), axis=-1)
    suspicious = (~valid) & nearly_real
    return widths, jnp.sum(suspicious, axis=-1, dtype=jnp.int32)


def _companion_strip_widths(coefficients: Array) -> tuple[Array, Array]:
    """Compute exact strip widths with the independent companion solver."""

    return _widths_from_roots(
        coefficients,
        batched_polished_real_companion_roots(coefficients),
    )


def _fixed_independent_ea_roots(
    coefficients: Array,
    ordinate_bound: Array,
    *,
    iterations: int,
) -> Array:
    """Solve every real sextic independently with a fixed CPU EA schedule."""

    real_dtype = coefficients.dtype
    complex_dtype = jnp.result_type(real_dtype, jnp.complex64)
    bound = jnp.asarray(ordinate_bound, dtype=real_dtype)
    degree = coefficients.shape[-1] - 1
    powers = jnp.arange(degree, -1, -1, dtype=real_dtype)
    scaled_real = coefficients * bound**powers
    coefficient_scale = jnp.max(jnp.abs(scaled_real), axis=-1, keepdims=True)
    scaled = (
        scaled_real
        / jnp.maximum(coefficient_scale, jnp.finfo(real_dtype).tiny)
    ).astype(complex_dtype)
    derivative = scaled[:, :-1] * jnp.arange(
        degree, 0, -1, dtype=real_dtype
    )
    leading = jnp.maximum(jnp.abs(scaled[:, :1]), jnp.finfo(real_dtype).eps)
    radius = 1.0 + jnp.max(jnp.abs(scaled[:, 1:]) / leading, axis=-1)
    phase = 2.0j * jnp.pi * (
        jnp.arange(degree, dtype=real_dtype) + 0.25
    ) / degree
    roots = radius[:, None] * jnp.exp(phase)[None, :]
    epsilon = 10.0 * jnp.finfo(real_dtype).eps

    def step(_, current):
        values = jax.vmap(jnp.polyval)(scaled, current)
        slopes = jax.vmap(jnp.polyval)(derivative, current)
        differences = current[:, :, None] - current[:, None, :]
        reciprocal = jnp.where(
            differences == 0.0,
            jnp.asarray(0.0 + 0.0j, dtype=complex_dtype),
            1.0 / differences,
        )
        denominator = slopes - values * jnp.sum(reciprocal, axis=-1)
        denominator = jnp.where(
            jnp.abs(denominator) > epsilon,
            denominator,
            slopes,
        )
        return current - values / denominator

    roots = jax.lax.fori_loop(0, iterations, step, roots) * bound
    return _attach_implicit_root_jvp(coefficients, roots)


def _bernstein_half_subdivide(coefficients: Array) -> tuple[Array, Array]:
    """Split degree-six Bernstein coefficients at the interval midpoint."""

    degree = coefficients.shape[-1] - 1
    temporary = coefficients
    left = [temporary[..., 0]]
    right = [temporary[..., -1]]
    for _ in range(degree):
        temporary = 0.5 * (temporary[..., :-1] + temporary[..., 1:])
        left.append(temporary[..., 0])
        right.append(temporary[..., -1])
    return jnp.stack(left, axis=-1), jnp.stack(right[::-1], axis=-1)


def _bernstein_strip_widths(
    coefficients: Array,
    *,
    ordinate_bound: Array,
    max_depth: int = 20,
    capacity: int = 10,
    compact_lookup: bool = False,
    clip_children: bool = False,
) -> tuple[Array, Array]:
    """Integrate negative polynomial intervals by Bernstein clipping."""

    degree = coefficients.shape[-1] - 1
    if degree != _SEXTIC_DEGREE:
        raise ValueError("Cartesian Bernstein isolation requires a sextic")
    real_dtype = coefficients.dtype
    upper_bound = jnp.asarray(ordinate_bound, dtype=real_dtype)
    lower_bound = -upper_bound
    sample_fraction = jnp.asarray(_BERNSTEIN_SAMPLE_FRACTION, dtype=real_dtype)
    value_to_bernstein = jnp.asarray(_BERNSTEIN_VALUE_TO_COEFFICIENT, dtype=real_dtype)
    sample_ordinate = lower_bound + (upper_bound - lower_bound) * sample_fraction
    values = jax.vmap(lambda polynomial: jnp.polyval(polynomial, sample_ordinate))(coefficients)
    initial = values @ value_to_bernstein
    n_polynomials = coefficients.shape[0]
    intervals = jnp.zeros((n_polynomials, capacity, 2), dtype=real_dtype)
    intervals = intervals.at[:, 0, 0].set(lower_bound)
    intervals = intervals.at[:, 0, 1].set(upper_bound)
    bernstein = jnp.zeros((n_polynomials, capacity, degree + 1), dtype=real_dtype).at[:, 0, :].set(initial)
    if clip_children:
        initial_scale = jnp.maximum(jnp.max(jnp.abs(initial), axis=-1), 1.0)
        initial_tolerance = 128.0 * jnp.finfo(real_dtype).eps * initial_scale
        initial_positive = jnp.all(
            initial > initial_tolerance[:, None], axis=-1
        )
        initial_negative = jnp.all(
            initial < -initial_tolerance[:, None], axis=-1
        )
        initial_mixed = ~initial_positive & ~initial_negative
        active = jnp.zeros((n_polynomials, capacity), dtype=bool).at[:, 0].set(
            initial_mixed
        )
        width = jnp.where(
            initial_negative,
            upper_bound - lower_bound,
            0.0,
        )
    else:
        active = jnp.zeros((n_polynomials, capacity), dtype=bool).at[:, 0].set(True)
        width = jnp.zeros((n_polynomials,), dtype=real_dtype)
    overflow = jnp.zeros((n_polynomials,), dtype=bool)

    def subdivision_step(_, state):
        current, bounds, current_active, inside_width, did_overflow = state
        if clip_children:
            # Every active interval was already classified as mixed when it
            # was packed by the preceding step.
            mixed = current_active
        else:
            coefficient_scale = jnp.maximum(
                jnp.max(jnp.abs(current), axis=-1), 1.0
            )
            tolerance = (
                128.0 * jnp.finfo(real_dtype).eps * coefficient_scale
            )
            positive = current_active & jnp.all(
                current > tolerance[..., None], axis=-1
            )
            negative = current_active & jnp.all(
                current < -tolerance[..., None], axis=-1
            )
            inside_width = inside_width + jnp.sum(
                jnp.where(
                    negative,
                    bounds[..., 1] - bounds[..., 0],
                    0.0,
                ),
                axis=-1,
            )
            mixed = current_active & ~positive & ~negative
        left, right = _bernstein_half_subdivide(current)
        midpoint = 0.5 * (bounds[..., 0] + bounds[..., 1])
        left_bounds = jnp.stack((bounds[..., 0], midpoint), axis=-1)
        right_bounds = jnp.stack((midpoint, bounds[..., 1]), axis=-1)
        children = jnp.stack((left, right), axis=2).reshape(n_polynomials, 2 * capacity, degree + 1)
        child_bounds = jnp.stack((left_bounds, right_bounds), axis=2).reshape(n_polynomials, 2 * capacity, 2)
        if clip_children:
            child_parent_active = jnp.repeat(mixed, 2, axis=-1)

            # Classify children before packing. A sextic can leave six parent
            # intervals mixed; temporarily duplicating them into twelve slots
            # is not a genuine capacity event.
            child_scale = jnp.maximum(
                jnp.max(jnp.abs(children), axis=-1), 1.0
            )
            child_tolerance = (
                128.0 * jnp.finfo(real_dtype).eps * child_scale
            )
            child_positive = child_parent_active & jnp.all(
                children > child_tolerance[..., None], axis=-1
            )
            child_negative = child_parent_active & jnp.all(
                children < -child_tolerance[..., None], axis=-1
            )
            inside_width = inside_width + jnp.sum(
                jnp.where(
                    child_negative,
                    child_bounds[..., 1] - child_bounds[..., 0],
                    0.0,
                ),
                axis=-1,
            )
            child_active = (
                child_parent_active & ~child_positive & ~child_negative
            )
        else:
            child_active = jnp.repeat(mixed, 2, axis=-1)
        child_count = jnp.sum(child_active, axis=-1, dtype=jnp.int32)
        did_overflow = did_overflow | (child_count > capacity)
        if compact_lookup and not clip_children:
            lookup = np.zeros((1 << capacity, capacity), dtype=np.int32)
            for mask in range(1 << capacity):
                indices = [
                    child_index
                    for parent_index in range(capacity)
                    if mask & (1 << parent_index)
                    for child_index in (2 * parent_index, 2 * parent_index + 1)
                ][:capacity]
                lookup[mask, : len(indices)] = indices
            mask_weight = jnp.asarray(
                1 << np.arange(capacity), dtype=jnp.int32
            )
            mask_index = jnp.sum(
                mixed.astype(jnp.int32) * mask_weight,
                axis=-1,
                dtype=jnp.int32,
            )
            selected_index = jnp.asarray(lookup)[mask_index]
            selected_active = (
                jnp.arange(capacity, dtype=jnp.int32)[None, :]
                < child_count[:, None]
            ).astype(jnp.int32)
        else:
            selected_active, selected_index = jax.lax.top_k(
                child_active.astype(jnp.int32), capacity
            )
        gather_coefficients = selected_index[..., None]
        current = jnp.take_along_axis(
            children,
            jnp.broadcast_to(gather_coefficients, (n_polynomials, capacity, degree + 1)),
            axis=1,
        )
        bounds = jnp.take_along_axis(
            child_bounds,
            jnp.broadcast_to(gather_coefficients, (n_polynomials, capacity, 2)),
            axis=1,
        )
        current_active = selected_active > 0
        return current, bounds, current_active, inside_width, did_overflow

    bernstein, intervals, active, width, overflow = jax.lax.fori_loop(
        0,
        max_depth,
        subdivision_step,
        (bernstein, intervals, active, width, overflow),
    )

    # Polish every final sign-changing interval.  Same-sign unresolved cells
    # are smaller than the returned subdivision uncertainty and are assigned
    # by their midpoint sign.
    lower = intervals[..., 0]
    upper = intervals[..., 1]
    midpoint = 0.5 * (lower + upper)
    lower_value = jax.vmap(jnp.polyval)(coefficients, lower)
    upper_value = jax.vmap(jnp.polyval)(coefficients, upper)
    midpoint_value = jax.vmap(jnp.polyval)(coefficients, midpoint)
    left_cross = active & (jnp.signbit(lower_value) != jnp.signbit(midpoint_value))
    right_cross = active & (jnp.signbit(midpoint_value) != jnp.signbit(upper_value))
    derivative_coefficients = coefficients[:, :-1] * jnp.arange(degree, 0, -1, dtype=real_dtype)

    def negative_half_length(left, right, left_value, right_value, crosses):
        denominator = jnp.abs(left_value) + jnp.abs(right_value)
        fraction = jnp.abs(left_value) / jnp.maximum(denominator, jnp.finfo(real_dtype).tiny)
        root = left + fraction * (right - left)

        def polish(_, state):
            estimate, low, high, low_value = state
            value = jax.vmap(jnp.polyval)(coefficients, estimate)
            slope = jax.vmap(jnp.polyval)(derivative_coefficients, estimate)
            newton = estimate - value / slope
            converged = jnp.abs(value) <= 32.0 * jnp.finfo(real_dtype).eps
            working = crosses & ~converged
            candidate = jnp.where(
                working
                & jnp.isfinite(newton)
                & (jnp.abs(slope) > 64.0 * jnp.finfo(real_dtype).eps)
                & (newton > low)
                & (newton < high),
                newton,
                jnp.where(working, 0.5 * (low + high), estimate),
            )
            candidate_value = jax.vmap(jnp.polyval)(coefficients, candidate)
            same_side = jnp.signbit(candidate_value) == jnp.signbit(low_value)
            low = jnp.where(working & same_side, candidate, low)
            low_value = jnp.where(working & same_side, candidate_value, low_value)
            high = jnp.where(working & ~same_side, candidate, high)
            return candidate, low, high, low_value

        root, _, _, _ = jax.lax.fori_loop(
            0,
            1,
            polish,
            (root, left, right, left_value),
        )
        root = _attach_implicit_root_jvp(coefficients, root)
        negative_lower = jnp.where(jnp.signbit(left_value), left, root)
        negative_upper = jnp.where(jnp.signbit(left_value), root, right)
        crossing_measure = negative_upper - negative_lower
        constant_measure = jnp.where(jnp.signbit(left_value), right - left, 0.0)
        return jnp.where(crosses, crossing_measure, constant_measure)

    final_width = negative_half_length(lower, midpoint, lower_value, midpoint_value, left_cross) + negative_half_length(
        midpoint, upper, midpoint_value, upper_value, right_cross
    )
    width = width + jnp.sum(jnp.where(active, final_width, 0.0), axis=-1)
    invalid = overflow.astype(jnp.int32)
    return width, invalid


def _sampled_strip_widths(
    coefficients: Array,
    *,
    ordinate_bound: Array,
    n_grid: int,
    n_refine: int = 20,
) -> tuple[Array, Array]:
    """Find real strip roots by sign sampling and safeguarded polishing.

    The source-disk boundary is a degree-six real polynomial on each strip.
    Simple real roots therefore announce themselves by a sign change; a dense
    shared ordinate grid replaces repeated complex eigensolves with Horner
    evaluations.  A root pair too narrow for the grid contributes at most one
    grid cell to that strip and is independently tested by the orthogonal
    projection certificate.
    """

    if n_grid < 16:
        raise ValueError("sampled strip grid must contain at least 16 cells")
    degree = coefficients.shape[-1] - 1
    real_dtype = coefficients.dtype
    ordinate_bound = jnp.asarray(ordinate_bound, dtype=real_dtype)
    grid = jnp.linspace(-ordinate_bound, ordinate_bound, n_grid + 1)
    values = jax.vmap(lambda polynomial: jnp.polyval(polynomial, grid))(coefficients)
    finite_values = jnp.isfinite(values)
    sign_change = (
        finite_values[:, :-1] & finite_values[:, 1:] & (jnp.signbit(values[:, :-1]) != jnp.signbit(values[:, 1:]))
    )

    def first_indices(mask):
        return jnp.nonzero(mask, size=degree, fill_value=-1)[0]

    indices = jax.vmap(first_indices)(sign_change)
    root_count = jnp.sum(indices >= 0, axis=-1, dtype=jnp.int32)
    safe_indices = jnp.maximum(indices, 0)
    lower = grid[safe_indices]
    upper = grid[jnp.minimum(safe_indices + 1, n_grid)]
    lower_value = jnp.take_along_axis(values[:, :-1], safe_indices, axis=1)
    active = indices >= 0
    powers = jnp.arange(degree, 0, -1, dtype=real_dtype)
    derivatives = coefficients[:, :-1] * powers

    def refine(_, state):
        low, high, low_value = state
        midpoint = 0.5 * (low + high)
        midpoint_value = jax.vmap(jnp.polyval)(coefficients, midpoint)
        midpoint_slope = jax.vmap(jnp.polyval)(derivatives, midpoint)
        newton = midpoint - midpoint_value / midpoint_slope
        use_newton = (
            active
            & jnp.isfinite(newton)
            & (jnp.abs(midpoint_slope) > 64.0 * jnp.finfo(real_dtype).eps)
            & (newton > low)
            & (newton < high)
        )
        candidate = jnp.where(use_newton, newton, midpoint)
        candidate_value = jax.vmap(jnp.polyval)(coefficients, candidate)
        same_side = jnp.signbit(candidate_value) == jnp.signbit(low_value)
        low = jnp.where(active & same_side, candidate, low)
        low_value = jnp.where(active & same_side, candidate_value, low_value)
        high = jnp.where(active & ~same_side, candidate, high)
        return low, high, low_value

    lower, upper, _ = jax.lax.fori_loop(0, n_refine, refine, (lower, upper, lower_value))
    roots = 0.5 * (lower + upper)
    root_value = jax.vmap(jnp.polyval)(coefficients, roots)
    root_slope = jax.vmap(jnp.polyval)(derivatives, roots)
    newton = roots - root_value / root_slope
    roots = jnp.where(
        active
        & jnp.isfinite(newton)
        & (jnp.abs(root_slope) > 64.0 * jnp.finfo(real_dtype).eps)
        & (newton > lower)
        & (newton < upper),
        newton,
        roots,
    )
    # Every active interval retains opposite endpoint signs throughout the
    # bisection, so its final width is the direct root-error certificate.
    roots_valid = (~active) | jnp.isfinite(roots)
    pair_index = jnp.arange(0, degree, 2)
    widths = jnp.sum(
        jnp.where(
            pair_index[None, :] + 1 < root_count[:, None],
            roots[:, pair_index + 1] - roots[:, pair_index],
            0.0,
        ),
        axis=-1,
    )
    invalid = (
        jnp.sum(~roots_valid, axis=-1, dtype=jnp.int32)
        + jnp.mod(root_count, 2)
        + jnp.where(root_count > degree, jnp.int32(1), jnp.int32(0))
    )
    return widths, invalid


def _polyline_seeded_strip_widths(
    coefficients: Array,
    abscissa: Array,
    *,
    image_limb: Array,
    physical_mask: Array,
    axis: Array,
) -> tuple[Array, Array]:
    """Polish strip roots seeded by the already available limb-image trace."""

    degree = coefficients.shape[-1] - 1
    real_dtype = coefficients.dtype
    rotated = image_limb * jnp.conjugate(axis)
    # The source limb is periodic but does not duplicate its first phase at
    # the end.  Include the closing segment; omitting it loses legitimate
    # strip crossings precisely when a branch extremum straddles phase zero.
    _, following_limb, _, following_mask = tracked_limb_neighbors(image_limb, physical_mask)
    segment_start = rotated
    segment_end = following_limb * jnp.conjugate(axis)
    segment_valid = physical_mask & following_mask
    x0 = segment_start.real.reshape(-1)
    x1 = segment_end.real.reshape(-1)
    y0 = segment_start.imag.reshape(-1)
    y1 = segment_end.imag.reshape(-1)
    valid_segment = segment_valid.reshape(-1)
    delta_x = x1 - x0
    safe_delta = jnp.where(
        jnp.abs(delta_x) > 64.0 * jnp.finfo(real_dtype).eps,
        delta_x,
        1.0,
    )
    fraction = (abscissa[:, None] - x0[None, :]) / safe_delta[None, :]
    intersections = (
        valid_segment[None, :]
        & (jnp.abs(delta_x)[None, :] > 64.0 * jnp.finfo(real_dtype).eps)
        & (fraction >= 0.0)
        & (fraction < 1.0)
    )
    candidate_capacity = 2 * degree
    candidate_active, safe_indices = jax.lax.top_k(intersections.astype(jnp.int32), candidate_capacity)
    candidate_active = candidate_active > 0
    candidate_fraction = jnp.take_along_axis(fraction, safe_indices, axis=1)
    candidates = y0[safe_indices] + candidate_fraction * (y1[safe_indices] - y0[safe_indices])
    powers = jnp.arange(degree, 0, -1, dtype=real_dtype)
    derivatives = coefficients[:, :-1] * powers

    def polish(_, roots):
        value = jax.vmap(jnp.polyval)(coefficients, roots)
        slope = jax.vmap(jnp.polyval)(derivatives, roots)
        safe = (
            candidate_active
            & jnp.isfinite(value)
            & jnp.isfinite(slope)
            & (jnp.abs(slope) > 64.0 * jnp.finfo(real_dtype).eps)
        )
        step = jnp.where(safe, value / slope, 0.0)
        return roots - step

    roots = jax.lax.fori_loop(0, 4, polish, candidates)
    residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    candidate_valid = candidate_active & jnp.isfinite(roots) & (residual <= 2.0e-7)
    order = jnp.argsort(jnp.where(candidate_valid, roots, jnp.inf), axis=-1)
    roots = jnp.take_along_axis(roots, order, axis=1)
    candidate_valid = jnp.take_along_axis(candidate_valid, order, axis=1)
    previous = jnp.concatenate((jnp.full((roots.shape[0], 1), -jnp.inf), roots[:, :-1]), axis=1)
    root_scale = jnp.maximum(1.0, jnp.abs(roots))
    distinct = (roots - previous) > 2.0e-7 * root_scale
    unique = candidate_valid & distinct

    def compact(row_roots, row_unique):
        indices = jnp.nonzero(row_unique, size=degree, fill_value=-1)[0]
        return jnp.where(indices >= 0, row_roots[jnp.maximum(indices, 0)], jnp.inf)

    roots = jax.vmap(compact)(roots, unique)
    root_count = jnp.minimum(jnp.sum(unique, axis=-1, dtype=jnp.int32), jnp.int32(degree))
    pair_index = jnp.arange(0, degree, 2)
    widths = jnp.sum(
        jnp.where(
            pair_index[None, :] + 1 < root_count[:, None],
            roots[:, pair_index + 1] - roots[:, pair_index],
            0.0,
        ),
        axis=-1,
    )
    overflow = jnp.sum(intersections, axis=-1, dtype=jnp.int32) > candidate_capacity
    invalid = jnp.mod(root_count, 2) + overflow.astype(jnp.int32) + (root_count == 0).astype(jnp.int32)
    return widths, invalid


def _real_root_selection(coefficients: Array, roots: Array) -> tuple[Array, Array]:
    """Sort numerically real polynomial roots and return their validity mask."""

    real = roots.real
    scale = jnp.maximum(1.0, jnp.abs(real))
    residual = jnp.abs(jnp.polyval(coefficients, roots))
    residual_scale = jnp.polyval(jnp.abs(coefficients), jnp.abs(roots))
    valid = (jnp.abs(roots.imag) <= 2.0e-7 * scale) & (
        residual <= 2.0e-7 * jnp.maximum(residual_scale, jnp.finfo(coefficients.dtype).tiny)
    )
    order = jnp.argsort(jnp.where(valid, real, jnp.inf))
    return real[order], valid[order]


def _continued_bracketed_real_roots(coefficients: Array) -> Array:
    """Track ordered real roots outwards from the centre of one strip cell.

    Inside a projection cell the number and ordering of real boundary roots is
    fixed.  Starting at the best-conditioned cell centre lets scalar safeguarded
    Newton iterations preserve that ordering; a cold companion solve is used
    only when the predicted brackets no longer certify one sign change each.
    """

    coefficients = jnp.asarray(coefficients)
    n_nodes = coefficients.shape[0]
    degree = coefficients.shape[-1] - 1
    center = (n_nodes - 1) // 2
    center_coefficients = coefficients[center]
    center_cold = batched_companion_roots(center_coefficients[None, :])[0]
    center_roots, center_valid = _real_root_selection(center_coefficients, center_cold)
    n_real = jnp.sum(center_valid, dtype=jnp.int32)
    active = jnp.arange(degree, dtype=jnp.int32) < n_real
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.dtype)

    def select_or_previous(current_coefficients, cold, previous):
        selected, valid = _real_root_selection(current_coefficients, cold)
        same_count = jnp.sum(valid, dtype=jnp.int32) == n_real
        return jnp.where(active & same_count, selected, previous)

    def advance(carry, current_coefficients):
        previous, previous_coefficients = carry
        coefficient_step = current_coefficients - previous_coefficients
        previous_derivative = previous_coefficients[:-1] * powers
        predictor_denominator = jnp.polyval(previous_derivative, previous)
        predictor_numerator = jnp.polyval(coefficient_step, previous)
        safe_predictor = active & (jnp.abs(predictor_denominator) > 64.0 * jnp.finfo(coefficients.dtype).eps)
        predicted = jnp.where(
            safe_predictor,
            previous - predictor_numerator / predictor_denominator,
            previous,
        )
        predicted = jnp.sort(jnp.where(active, predicted, jnp.inf))

        # Adjacent Gauss nodes lie in the same topology cell.  A unit margin
        # around the previous real roots is much tighter than the often huge
        # Cauchy bound when the normalized sextic leading term is small.
        root_bound = 1.0 + jnp.max(jnp.where(active, jnp.abs(previous), 0.0))
        left_neighbour = jnp.concatenate((jnp.asarray([-root_bound], dtype=predicted.dtype), predicted[:-1]))
        right_neighbour = jnp.concatenate((predicted[1:], jnp.asarray([root_bound], dtype=predicted.dtype)))
        lower = jnp.where(
            jnp.arange(degree) == 0,
            -root_bound,
            0.5 * (left_neighbour + predicted),
        )
        upper = jnp.where(
            jnp.arange(degree) == n_real - 1,
            root_bound,
            0.5 * (predicted + right_neighbour),
        )
        lower = jnp.where(active, lower, 0.0)
        upper = jnp.where(active, upper, 0.0)
        f_lower = jnp.polyval(current_coefficients, lower)
        f_upper = jnp.polyval(current_coefficients, upper)
        bracketed = active & jnp.isfinite(lower) & jnp.isfinite(upper) & (f_lower * f_upper <= 0.0)
        brackets_ok = jnp.all(bracketed | ~active)
        derivative = current_coefficients[:-1] * powers

        def polish(state):
            root, low, high, low_value = state
            value = jnp.polyval(current_coefficients, root)
            slope = jnp.polyval(derivative, root)
            newton = root - value / slope
            midpoint = 0.5 * (low + high)
            use_newton = (
                active
                & jnp.isfinite(newton)
                & (jnp.abs(slope) > 64.0 * jnp.finfo(coefficients.dtype).eps)
                & (newton > low)
                & (newton < high)
            )
            candidate = jnp.where(use_newton, newton, midpoint)
            candidate_value = jnp.polyval(current_coefficients, candidate)
            same_side = jnp.signbit(candidate_value) == jnp.signbit(low_value)
            low = jnp.where(active & same_side, candidate, low)
            low_value = jnp.where(active & same_side, candidate_value, low_value)
            high = jnp.where(active & ~same_side, candidate, high)
            return candidate, low, high, low_value

        initial = (
            jnp.clip(predicted, lower, upper),
            lower,
            upper,
            f_lower,
        )
        polished, _, _, _ = jax.lax.fori_loop(0, 40, lambda _, state: polish(state), initial)

        def use_polished(_):
            return polished

        def use_cold(_):
            cold = batched_companion_roots(current_coefficients[None, :])[0]
            return select_or_previous(current_coefficients, cold, previous)

        current = jax.lax.cond(brackets_ok, use_polished, use_cold, operand=None)
        current = jnp.where(active, current, jnp.inf)
        return (current, current_coefficients), current

    initial_carry = (jnp.where(active, center_roots, jnp.inf), center_coefficients)
    _, left = jax.lax.scan(
        advance,
        initial_carry,
        coefficients[:center][::-1],
    )
    _, right = jax.lax.scan(
        advance,
        initial_carry,
        coefficients[center + 1 :],
    )
    ordered = jnp.concatenate((left[::-1], initial_carry[0][None, :], right), axis=0)
    finite = jnp.isfinite(ordered)
    return jax.lax.complex(
        jnp.where(finite, ordered, 0.0),
        jnp.where(finite, 0.0, jnp.inf),
    )


def _continued_companion_roots(coefficients: Array, *, iterations: int = 8) -> Array:
    """Track polynomial roots across ordered strips after one cold solve."""

    coefficients = jnp.asarray(coefficients)
    first_roots = batched_companion_roots(coefficients[:1])[0]
    degree = coefficients.shape[-1] - 1
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.dtype)

    def continue_roots(carry, current_coefficients):
        roots, previous_coefficients = carry
        previous_derivative = previous_coefficients[:-1] * powers
        coefficient_step = current_coefficients - previous_coefficients
        predictor_denominator = jnp.polyval(previous_derivative, roots)
        predictor_numerator = jnp.polyval(coefficient_step, roots)
        safe_predictor = jnp.abs(predictor_denominator) > 64.0 * jnp.finfo(coefficients.dtype).eps
        predicted = jnp.where(
            safe_predictor,
            roots - predictor_numerator / predictor_denominator,
            roots,
        )
        current_derivative = current_coefficients[:-1] * powers

        def aberth_step(_, roots):
            residual = jnp.polyval(current_coefficients, roots)
            derivative = jnp.polyval(current_derivative, roots)
            difference = roots[:, None] - roots[None, :]
            off_diagonal = ~jnp.eye(degree, dtype=bool)
            safe_difference = off_diagonal & (jnp.abs(difference) > 64.0 * jnp.finfo(coefficients.dtype).eps)
            guarded_difference = jnp.where(safe_difference, difference, 1.0)
            repulsion = jnp.sum(jnp.where(safe_difference, 1.0 / guarded_difference, 0.0), axis=1)
            denominator = derivative - residual * repulsion
            safe = jnp.abs(denominator) > 64.0 * jnp.finfo(coefficients.dtype).eps
            return jnp.where(safe, roots - residual / denominator, roots)

        roots = jax.lax.fori_loop(0, iterations, aberth_step, predicted)
        return (roots, current_coefficients), roots

    (_, _), remaining_roots = jax.lax.scan(
        continue_roots,
        (first_roots, coefficients[0]),
        coefficients[1:],
    )
    return jnp.concatenate((first_roots[None, :], remaining_roots), axis=0)


def _critical_root_diagnostics(coefficients: Array) -> tuple[Array, Array, Array]:
    """Return normalized critical values and real-root counts for strip polynomials."""

    degree = coefficients.shape[-1] - 1
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.dtype)
    derivative_coefficients = coefficients[..., :-1] * powers
    critical_roots = batched_companion_roots(derivative_coefficients)
    critical_real = critical_roots.real
    critical_scale = jnp.maximum(1.0, jnp.abs(critical_real))
    critical_nearly_real = jnp.abs(critical_roots.imag) <= 2.0e-7 * critical_scale
    critical_residual = jnp.abs(jax.vmap(jnp.polyval)(derivative_coefficients, critical_roots))
    critical_residual_scale = jax.vmap(jnp.polyval)(jnp.abs(derivative_coefficients), jnp.abs(critical_roots))
    critical_valid = critical_nearly_real & (
        critical_residual
        <= 2.0e-7
        * jnp.maximum(
            critical_residual_scale,
            jnp.finfo(coefficients.dtype).tiny,
        )
    )
    critical_value = jnp.abs(jax.vmap(jnp.polyval)(coefficients, critical_real.astype(coefficients.dtype)))
    critical_value_scale = jax.vmap(jnp.polyval)(jnp.abs(coefficients), jnp.abs(critical_real))
    normalized_critical_value = critical_value / jnp.maximum(
        critical_value_scale,
        jnp.finfo(coefficients.dtype).tiny,
    )
    minimum_critical_value = jnp.min(jnp.where(critical_valid, normalized_critical_value, jnp.inf), axis=-1)

    roots = batched_companion_roots(coefficients)
    root_scale = jnp.maximum(1.0, jnp.abs(roots.real))
    root_residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    root_residual_scale = jax.vmap(jnp.polyval)(jnp.abs(coefficients), jnp.abs(roots))
    real_roots = (jnp.abs(roots.imag) <= 2.0e-7 * root_scale) & (
        root_residual <= 2.0e-7 * jnp.maximum(root_residual_scale, jnp.finfo(coefficients.dtype).tiny)
    )
    root_count = jnp.sum(real_roots, axis=-1, dtype=jnp.int32)
    invalid_critical = jnp.sum(critical_nearly_real & ~critical_valid, axis=-1, dtype=jnp.int32)
    return minimum_critical_value, root_count, invalid_critical


def _cartesian_topology_probe_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    support: tuple[Array, Array, Array, Array, Array],
    n_samples: int = 9,
    axis: complex | Array = 1.0 + 0.0j,
) -> CartesianTopologyProbe:
    """Probe derivative roots across every active Cartesian support cell."""

    if n_samples < 5:
        raise ValueError("n_samples must be at least five")
    axis = _normalized_axis(axis, w_center.dtype)
    cells, active, *_ = support
    fractions = 0.5 * (1.0 - jnp.cos(jnp.pi * (jnp.arange(n_samples, dtype=w_center.real.dtype) + 0.5) / n_samples))
    abscissa = cells[:, :1] + (cells[:, 1:] - cells[:, :1]) * fractions[None, :]
    abscissa = jnp.where(active[:, None], abscissa, 0.0)
    flat_abscissa = abscissa.reshape(-1)
    coefficients = jax.vmap(
        lambda x: binary_line_level_set_coefficients(x * axis, 1.0j * axis, w_center, rho, s=s, q=q)
    )(flat_abscissa)
    minimum, root_count, invalid = _critical_root_diagnostics(coefficients)
    minimum = minimum.reshape(abscissa.shape)
    root_count = root_count.reshape(abscissa.shape)
    invalid = invalid.reshape(abscissa.shape)
    active_minimum = jnp.where(active[:, None], minimum, jnp.inf)
    interior_minimum = jnp.where(active[:, None], minimum[:, 2:-2], jnp.inf)
    count_minimum = jnp.min(root_count, axis=1)
    count_maximum = jnp.max(root_count, axis=1)
    return CartesianTopologyProbe(
        jnp.min(active_minimum),
        jnp.min(interior_minimum),
        jnp.max(jnp.where(active, count_maximum - count_minimum, 0)),
        jnp.sum(jnp.where(active[:, None], invalid, 0), dtype=jnp.int32),
    )


def _cartesian_support_cells(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_limb: int,
    axis: complex | Array = 1.0 + 0.0j,
) -> tuple[Array, Array, Array, Array, Array]:
    """Trace the limb and construct support cells for one projection axis."""

    image_limb, physical_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        include_all_roots=False,
    )
    trace_neighbors = tracked_limb_neighbors(image_limb, physical_mask)
    traced = _cartesian_trace_diagnostics(
        w_center,
        rho,
        s=s,
        q=q,
        image_limb=image_limb,
        physical_mask=physical_mask,
    )
    return _cartesian_support_cells_from_trace(
        image_limb,
        physical_mask,
        topology_uncertain=traced[0],
        minimum_ghost_residual=traced[1],
        limb_topology=traced[2],
        axis=axis,
        neighbors=trace_neighbors,
    )


def _cartesian_trace_diagnostics(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    image_limb: Array,
    physical_mask: Array,
) -> tuple[Array, Array, Array]:
    """Compute axis-independent topology diagnostics for one limb trace."""

    physical_counts = jnp.sum(physical_mask, axis=0)
    limb_transition = jnp.any(physical_counts != physical_counts[0])
    buried = hidden_caustic_candidate(
        w_center,
        rho,
        s=s,
        q=q,
        limb_transition=limb_transition,
    )
    limb_topology = limb_transition | buried
    topology_uncertain = limb_topology

    lens = binary_geometry(s, q)
    n_limb = image_limb.shape[1]
    phases = 2.0 * jnp.pi * jnp.arange(n_limb, dtype=w_center.real.dtype) / n_limb
    source_limb = w_center + rho * jnp.exp(1.0j * phases)
    residual = jax.vmap(
        lambda images, source: jnp.abs(
            lens_eq(
                images - lens.shifted,
                nlenses=2,
                a=lens.a,
                e1=lens.e1,
            )
            - (source - lens.shifted)
        ),
        in_axes=(1, 0),
    )(image_limb, source_limb).T
    minimum_ghost_residual = jnp.min(jnp.where(~physical_mask, residual, jnp.inf))
    topology_uncertain = topology_uncertain | (minimum_ghost_residual <= 4.0 * rho)
    # A non-finite algebraic (including ghost) root means the quintic trace is
    # not a complete support certificate, even if its three best roots happen
    # to remain finite.  Propagate this as topology uncertainty so Cartesian
    # and polar schedulers cannot certify a partial trace.
    topology_uncertain = topology_uncertain | ~jnp.all(jnp.isfinite(image_limb))

    return topology_uncertain, minimum_ghost_residual, limb_topology


def _cartesian_support_cells_from_trace(
    image_limb: Array,
    physical_mask: Array,
    *,
    topology_uncertain: Array,
    minimum_ghost_residual: Array,
    limb_topology: Array,
    axis: complex | Array = 1.0 + 0.0j,
    maximum_extrema: int = 20,
    neighbors: tuple[Array, Array, Array, Array] | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Construct strip support cells from a shared traced source limb."""

    axis = _normalized_axis(axis, image_limb.dtype)
    coordinate = jnp.real(image_limb * jnp.conjugate(axis))
    if neighbors is None:
        neighbors = tracked_limb_neighbors(image_limb, physical_mask)
    previous_limb, following_limb, previous_mask, following_mask = neighbors
    previous = jnp.real(previous_limb * jnp.conjugate(axis))
    following = jnp.real(following_limb * jnp.conjugate(axis))
    incoming = coordinate - previous
    outgoing = following - coordinate
    candidate_mask = physical_mask & (
        (incoming * outgoing <= 0.0) | (~previous_mask) | (~following_mask)
    )

    # Monodromy can carry a physical image branch into a different algebraic
    # root slot around the periodic closing edge.  Such a slot can be physical
    # throughout the trace yet have no local turning-point candidate of its
    # own.  The support as a whole is then nonempty, so the global empty check
    # below cannot detect the missing projection range.  Guarantee two sampled
    # endpoints for exactly those branches.  This adds no trace or integral;
    # ordinary branches keep the fitted extrema used below.
    sample_index = jnp.arange(image_limb.shape[1], dtype=jnp.int32)
    finite_physical = physical_mask & jnp.isfinite(coordinate)
    branch_physical = jnp.any(finite_physical, axis=1)
    branch_candidate = jnp.any(candidate_mask, axis=1)
    missing_branch = branch_physical & ~branch_candidate
    branch_minimum = jnp.argmin(
        jnp.where(finite_physical, coordinate, jnp.inf),
        axis=1,
    )
    branch_maximum = jnp.argmax(
        jnp.where(finite_physical, coordinate, -jnp.inf),
        axis=1,
    )
    fallback = missing_branch[:, None] & (
        (sample_index[None, :] == branch_minimum[:, None])
        | (sample_index[None, :] == branch_maximum[:, None])
    )
    candidate_mask = candidate_mask | fallback

    curvature = previous - 2.0 * coordinate + following
    safe_curvature = jnp.where(
        jnp.abs(curvature) > 64.0 * jnp.finfo(coordinate.dtype).eps,
        curvature,
        1.0,
    )
    offset = jnp.clip(0.5 * (previous - following) / safe_curvature, -1.0, 1.0)
    fitted = coordinate + 0.5 * (following - previous) * offset + 0.5 * curvature * offset**2
    fitted = jnp.where(previous_mask & following_mask, fitted, coordinate)

    # A caustic-crossing binary source can expose three persistent-image
    # pairs plus a transient fold pair in one projection.  Each branch may
    # contribute two turning points, and mask-transition endpoints add two
    # more for the fold pair.  Sixteen slots therefore truncate legitimate
    # 17--18 endpoint configurations.  Keep fixed shape, but retain enough
    # room for the complete binary-lens limb topology.
    flat_mask = candidate_mask.reshape(-1)
    indices = jnp.nonzero(
        flat_mask,
        size=maximum_extrema,
        fill_value=-1,
    )[0]
    safe_indices = jnp.maximum(indices, 0)
    selected = fitted.reshape(-1)[safe_indices]
    selected = jnp.where(indices >= 0, selected, jnp.inf)
    n_extrema = jnp.sum(flat_mask, dtype=jnp.int32)
    trace_finite = jnp.all(jnp.isfinite(image_limb))
    # A finite periodic image branch must expose at least one projection
    # turning point (and a branch endpoint is also a candidate).  If all
    # candidates disappear, the trace is numerically unusable; an empty
    # support must never be allowed to certify a zero-area magnification.
    topology_uncertain = topology_uncertain | (n_extrema > maximum_extrema)
    topology_uncertain = topology_uncertain | ~trace_finite | (n_extrema == 0)
    endpoints = jnp.sort(selected)
    cells = jnp.stack((endpoints[:-1], endpoints[1:]), axis=-1)
    cell_index = jnp.arange(maximum_extrema - 1, dtype=jnp.int32)
    active = (cell_index < n_extrema - 1) & jnp.isfinite(cells[:, 1]) & (cells[:, 1] > cells[:, 0])
    return cells, active, topology_uncertain, minimum_ghost_residual, limb_topology


def _cartesian_orthogonal_supports(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_limb: int,
    axis: complex | Array = 1.0 + 0.0j,
    return_trace: bool = False,
) -> tuple:
    """Build orthogonal strip supports while sharing the expensive limb trace."""

    image_limb, physical_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        include_all_roots=False,
    )
    trace_neighbors = tracked_limb_neighbors(image_limb, physical_mask)
    topology, ghost, limb_topology = _cartesian_trace_diagnostics(
        w_center,
        rho,
        s=s,
        q=q,
        image_limb=image_limb,
        physical_mask=physical_mask,
    )
    axis = _normalized_axis(axis, w_center.dtype)
    common = {
        "topology_uncertain": topology,
        "minimum_ghost_residual": ghost,
        "limb_topology": limb_topology,
    }
    primary = _cartesian_support_cells_from_trace(
        image_limb,
        physical_mask,
        axis=axis,
        neighbors=trace_neighbors,
        **common,
    )
    orthogonal = _cartesian_support_cells_from_trace(
        image_limb,
        physical_mask,
        axis=1.0j * axis,
        neighbors=trace_neighbors,
        **common,
    )
    if return_trace:
        return primary, orthogonal, axis, image_limb, physical_mask
    return primary, orthogonal, axis


def mag_uniform_cartesian_moment_fixed(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_slice: int = 8,
    n_limb: int = 128,
    axis: complex | Array = 1.0 + 0.0j,
    continuation: bool = False,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Integrate image-plane vertical-strip widths with fixed Gauss order."""

    if n_slice <= 0:
        raise ValueError("n_slice must be positive")
    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    cells, active, topology, ghost, limb_topology = _cartesian_support_cells(
        w_center, rho, s=s, q=q, n_limb=n_limb, axis=axis
    )
    result = _cartesian_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_slice=n_slice,
        support=(cells, active, topology, ghost, limb_topology),
        axis=axis,
        continuation=continuation,
    )
    return result if return_info else result.magnification


def _cartesian_result_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_slice: int,
    support: tuple[Array, Array, Array, Array, Array],
    axis: complex | Array = 1.0 + 0.0j,
    continuation: bool = False,
    image_limb: Array | None = None,
    physical_mask: Array | None = None,
) -> CartesianMomentResult:
    """Integrate one Cartesian Gauss rule on precomputed strip support."""

    cells, active, topology, ghost, limb_topology = support
    axis = _normalized_axis(axis, w_center.dtype)
    nodes, weights = np.polynomial.legendre.leggauss(n_slice)
    nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
    weights = jnp.asarray(weights, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)
    lens = binary_geometry(s, q)
    lens_radius = jnp.maximum(jnp.abs(lens.shifted - lens.a), jnp.abs(lens.shifted + lens.a))
    source_bound = jnp.abs(w_center) + rho
    radial_offset = source_bound - lens_radius
    image_bound = lens_radius + 0.5 * (radial_offset + jnp.sqrt(radial_offset**2 + 4.0))
    image_bound *= 1.0 + 32.0 * jnp.finfo(w_center.real.dtype).eps

    if continuation in ("seeded", "polyline"):
        if image_limb is None or physical_mask is None:
            raise ValueError("seeded continuation requires a limb trace")
        cell_width = cells[:, 1] - cells[:, 0]
        abscissa = cells[:, :1] + cell_width[:, None] * jnp.sin(transform) ** 2
        abscissa = jnp.where(active[:, None], abscissa, 0.0)
        strip_weights = weights[None, :] * 0.25 * jnp.pi * cell_width[:, None] * jnp.sin(2.0 * transform)[None, :]
        flat_abscissa = abscissa.reshape(-1)
        coefficients = jax.vmap(
            lambda x: binary_line_level_set_coefficients(x * axis, 1.0j * axis, w_center, rho, s=s, q=q)
        )(flat_abscissa)
        seeded_width, seeded_invalid = _polyline_seeded_strip_widths(
            coefficients,
            flat_abscissa,
            image_limb=image_limb,
            physical_mask=physical_mask,
            axis=axis,
        )
        if continuation == "seeded":
            sampled_width, sampled_invalid = _sampled_strip_widths(
                coefficients,
                ordinate_bound=image_bound,
                n_grid=64,
            )
            grid_width = 2.0 * image_bound / 64.0
            seeded_correction = seeded_width - sampled_width
            trust_seed = (
                (seeded_invalid == 0)
                & (seeded_correction >= -2.0e-6 * jnp.maximum(jnp.abs(seeded_width), 1.0))
                & (seeded_correction <= 1.25 * grid_width)
            )
            flat_width = jnp.where(
                trust_seed,
                jnp.maximum(seeded_width, sampled_width),
                sampled_width,
            )
            flat_invalid = jnp.where(trust_seed, seeded_invalid, sampled_invalid).reshape(abscissa.shape)
        else:
            flat_width = seeded_width
            flat_invalid = seeded_invalid.reshape(abscissa.shape)
        strip_width = flat_width.reshape(abscissa.shape)
        area = jnp.sum(jnp.where(active[:, None], strip_weights * strip_width, 0.0))
        return CartesianMomentResult(
            area / (jnp.pi * rho**2),
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.int32(n_slice) * jnp.sum(active, dtype=jnp.int32),
            jnp.sum(jnp.where(active[:, None], flat_invalid, 0), dtype=jnp.int32),
            ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
            limb_topology,
            jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        )

    if continuation in (
        "bernstein_dynamic",
        "bernstein_dynamic_full",
        "bernstein_lookup_no_repair",
        "bernstein_clip_no_repair",
    ):

        strip_continuation = (
            "bernstein_lookup_full"
            if continuation == "bernstein_dynamic_full"
            else continuation
            if continuation in (
                "bernstein_lookup_no_repair",
                "bernstein_clip_no_repair",
            )
            else "bernstein_lookup"
        )

        def integrate_active(cell_index, state):
            area, invalid_total = state
            bounds = cells[cell_index]
            width = bounds[1] - bounds[0]
            abscissa = bounds[0] + width * jnp.sin(transform) ** 2
            strip_weights = weights * 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
            coefficients = jax.vmap(
                lambda x: binary_line_level_set_coefficients(x * axis, 1.0j * axis, w_center, rho, s=s, q=q)
            )(abscissa)
            strip_width, invalid = _strip_widths(
                coefficients,
                continuation=strip_continuation,
                ordinate_bound=image_bound,
                source_radius=rho,
            )
            return (
                area + jnp.sum(strip_weights * strip_width),
                invalid_total + jnp.sum(invalid, dtype=jnp.int32),
            )

        area, invalid = jax.lax.fori_loop(
            jnp.int32(0),
            jnp.sum(active, dtype=jnp.int32),
            integrate_active,
            (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
            ),
        )
        return CartesianMomentResult(
            area / (jnp.pi * rho**2),
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.int32(n_slice) * jnp.sum(active, dtype=jnp.int32),
            invalid,
            ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
            limb_topology,
            jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        )

    def integrate_cell(inputs):
        bounds, cell_active = inputs

        def evaluate(_):
            width = bounds[1] - bounds[0]
            abscissa = bounds[0] + width * jnp.sin(transform) ** 2
            strip_weights = weights * 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
            coefficients = jax.vmap(
                lambda x: binary_line_level_set_coefficients(x * axis, 1.0j * axis, w_center, rho, s=s, q=q)
            )(abscissa)
            if continuation in ("seeded", "seeded_cell"):
                if image_limb is None or physical_mask is None:
                    raise ValueError("seeded continuation requires a limb trace")
                strip_width, invalid = _polyline_seeded_strip_widths(
                    coefficients,
                    abscissa,
                    image_limb=image_limb,
                    physical_mask=physical_mask,
                    axis=axis,
                )
                if continuation == "seeded":
                    sampled_width, sampled_invalid = _sampled_strip_widths(
                        coefficients,
                        ordinate_bound=image_bound,
                        n_grid=64,
                    )
                    grid_width = 2.0 * image_bound / 64.0
                    seeded_correction = strip_width - sampled_width
                    trust_seed = (
                        (invalid == 0)
                        & (seeded_correction >= -2.0e-6 * jnp.maximum(strip_width, 1.0))
                        & (seeded_correction <= 1.25 * grid_width)
                    )
                    strip_width = jnp.where(
                        trust_seed,
                        jnp.maximum(strip_width, sampled_width),
                        sampled_width,
                    )
                    invalid = jnp.where(trust_seed, invalid, sampled_invalid)
                else:

                    def repair_seed(_):
                        repaired_width, repaired_invalid = _strip_widths(
                            coefficients,
                            continuation="bernstein_adaptive",
                            ordinate_bound=image_bound,
                            source_radius=rho,
                        )
                        failed = invalid != 0
                        return (
                            jnp.where(failed, repaired_width, strip_width),
                            jnp.where(failed, repaired_invalid, jnp.int32(0)),
                        )

                    strip_width, invalid = jax.lax.cond(
                        jnp.any(invalid != 0),
                        repair_seed,
                        lambda _: (strip_width, jnp.zeros_like(invalid)),
                        operand=None,
                    )
            else:
                strip_width, invalid = _strip_widths(
                    coefficients,
                    continuation=continuation,
                    ordinate_bound=image_bound,
                    source_radius=rho,
                )
            return jnp.sum(strip_weights * strip_width), jnp.sum(invalid, dtype=jnp.int32)

        return jax.lax.cond(
            cell_active,
            evaluate,
            lambda _: (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
            ),
            operand=None,
        )

    cell_areas, invalid = jax.lax.map(integrate_cell, (cells, active))
    area = jnp.sum(cell_areas)
    return CartesianMomentResult(
        area / (jnp.pi * rho**2),
        jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
        jnp.int32(n_slice) * jnp.sum(active, dtype=jnp.int32),
        jnp.sum(invalid, dtype=jnp.int32),
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
    )


def _cartesian_gk15_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    support: tuple[Array, Array, Array, Array, Array],
    axis: complex | Array,
    continuation: str = "bernstein_lookup",
) -> tuple[CartesianMomentResult, CartesianMomentResult]:
    """Evaluate a nested Kronrod-15/Gauss-7 Cartesian strip pair once."""

    cells, active, topology, ghost, limb_topology = support
    axis = _normalized_axis(axis, w_center.dtype)
    nodes = jnp.asarray(GK15_X, dtype=w_center.real.dtype)
    fine_weights = jnp.asarray(GK15_W, dtype=w_center.real.dtype)
    coarse_weights = jnp.asarray(G7_W_ON_GK15, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)
    lens = binary_geometry(s, q)
    lens_radius = jnp.maximum(jnp.abs(lens.shifted - lens.a), jnp.abs(lens.shifted + lens.a))
    source_bound = jnp.abs(w_center) + rho
    radial_offset = source_bound - lens_radius
    image_bound = lens_radius + 0.5 * (radial_offset + jnp.sqrt(radial_offset**2 + 4.0))
    image_bound *= 1.0 + 32.0 * jnp.finfo(w_center.real.dtype).eps

    def integrate_active(cell_index, state):
        fine_area, coarse_area, fine_invalid, coarse_invalid = state
        bounds = cells[cell_index]
        width = bounds[1] - bounds[0]
        abscissa = bounds[0] + width * jnp.sin(transform) ** 2
        jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
        coefficients = jax.vmap(
            lambda x: binary_line_level_set_coefficients(x * axis, 1.0j * axis, w_center, rho, s=s, q=q)
        )(abscissa)
        strip_width, invalid = _strip_widths(
            coefficients,
            continuation=continuation,
            ordinate_bound=image_bound,
            source_radius=rho,
        )
        transformed_width = jacobian * strip_width
        return (
            fine_area + jnp.sum(fine_weights * transformed_width),
            coarse_area + jnp.sum(coarse_weights * transformed_width),
            fine_invalid + jnp.sum(invalid, dtype=jnp.int32),
            coarse_invalid + jnp.sum(jnp.where(coarse_weights != 0.0, invalid, 0), dtype=jnp.int32),
        )

    fine_area, coarse_area, fine_invalid, coarse_invalid = jax.lax.fori_loop(
        jnp.int32(0),
        jnp.sum(active, dtype=jnp.int32),
        integrate_active,
        (
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            jnp.int32(0),
            jnp.int32(0),
        ),
    )
    normalization = jnp.pi * rho**2
    status = jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0))
    ghost_ratio = ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    n_active = jnp.sum(active, dtype=jnp.int32)

    def result(area, invalid, order):
        return CartesianMomentResult(
            area / normalization,
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.int32(order) * n_active,
            invalid,
            ghost_ratio,
            limb_topology,
            status,
        )

    return result(fine_area, fine_invalid, 15), result(coarse_area, coarse_invalid, 7)


def _cartesian_gl12_20_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    support: tuple[Array, Array, Array, Array, Array],
    axis: complex | Array,
    continuation: bool | str = "bernstein_clip_no_repair",
) -> tuple[CartesianMomentResult, CartesianMomentResult]:
    """Evaluate a fixed Cartesian Gauss-12/Gauss-20 pair.

    This fixed high-order path is reserved for trace-identified fragmented
    fold arcs.  Its Bernstein kernel clips children before packing, so a
    transient 6-to-12 subdivision does not discard any of the sextic's real
    root intervals.  Independent Gauss phases deliberately avoid the common
    narrow-feature miss seen in nested rules on fragmented fold arcs.
    """

    cells, active, topology, ghost, limb_topology = support
    axis = _normalized_axis(axis, w_center.dtype)
    fine_nodes, fine_weights = np.polynomial.legendre.leggauss(20)
    coarse_nodes, coarse_weights = np.polynomial.legendre.leggauss(12)
    nodes = jnp.asarray(
        np.concatenate((fine_nodes, coarse_nodes)),
        dtype=w_center.real.dtype,
    )
    fine_weights = jnp.asarray(fine_weights, dtype=w_center.real.dtype)
    coarse_weights = jnp.asarray(
        coarse_weights, dtype=w_center.real.dtype
    )
    transform = 0.25 * jnp.pi * (nodes + 1.0)
    lens = binary_geometry(s, q)
    lens_radius = jnp.maximum(
        jnp.abs(lens.shifted - lens.a), jnp.abs(lens.shifted + lens.a)
    )
    source_bound = jnp.abs(w_center) + rho
    radial_offset = source_bound - lens_radius
    image_bound = lens_radius + 0.5 * (
        radial_offset + jnp.sqrt(radial_offset**2 + 4.0)
    )
    image_bound *= 1.0 + 32.0 * jnp.finfo(w_center.real.dtype).eps

    def integrate_active(cell_index, state):
        fine_area, coarse_area, fine_invalid, coarse_invalid = state
        bounds = cells[cell_index]
        width = bounds[1] - bounds[0]
        abscissa = bounds[0] + width * jnp.sin(transform) ** 2
        jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
        coefficients = jax.vmap(
            lambda x: binary_line_level_set_coefficients(
                x * axis,
                1.0j * axis,
                w_center,
                rho,
                s=s,
                q=q,
            )
        )(abscissa)
        strip_width, invalid = _strip_widths(
            coefficients,
            continuation=continuation,
            ordinate_bound=image_bound,
            source_radius=rho,
        )
        transformed_width = jacobian * strip_width
        return (
            fine_area + jnp.sum(fine_weights * transformed_width[:20]),
            coarse_area + jnp.sum(coarse_weights * transformed_width[20:]),
            fine_invalid + jnp.sum(invalid[:20], dtype=jnp.int32),
            coarse_invalid
            + jnp.sum(invalid[20:], dtype=jnp.int32),
        )

    fine_area, coarse_area, fine_invalid, coarse_invalid = jax.lax.fori_loop(
        jnp.int32(0),
        jnp.sum(active, dtype=jnp.int32),
        integrate_active,
        (
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            jnp.int32(0),
            jnp.int32(0),
        ),
    )
    normalization = jnp.pi * rho**2
    status = jnp.where(
        topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)
    )
    ghost_ratio = ghost / jnp.maximum(
        rho, jnp.finfo(rho.dtype).tiny
    )
    n_active = jnp.sum(active, dtype=jnp.int32)

    def result(area, invalid, order):
        return CartesianMomentResult(
            area / normalization,
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.int32(order) * n_active,
            invalid,
            ghost_ratio,
            limb_topology,
            status,
        )

    return result(fine_area, fine_invalid, 20), result(
        coarse_area, coarse_invalid, 12
    )


def _mag_uniform_cartesian_moment_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    rtol: Array,
    support: tuple[Array, Array, Array, Array, Array],
    axis: complex | Array = 1.0 + 0.0j,
) -> CartesianMomentResult:
    """Evaluate and certify the 4/6-point Cartesian ICRS pair."""

    fine = _cartesian_result_from_support(w_center, rho, s=s, q=q, n_slice=6, support=support, axis=axis)
    topology = support[2]
    ghost_ratio = fine.ghost_residual_ratio
    needs_coarse_check = (fine.status == 0) | ((fine.status == ANGULAR_MOMENT_TOPOLOGY) & (ghost_ratio > 5.0))
    coarse_magnification = jax.lax.cond(
        needs_coarse_check,
        lambda _: _cartesian_result_from_support(
            w_center, rho, s=s, q=q, n_slice=4, support=support, axis=axis
        ).magnification,
        lambda _: fine.magnification,
        operand=None,
    )
    scale = jnp.maximum(jnp.abs(fine.magnification), 1.0)
    tier_difference = jnp.abs(fine.magnification - coarse_magnification)
    relative_difference = tier_difference / scale
    regular_certificate = (fine.status == 0) & (relative_difference <= 2.0 * rtol)
    topology_certificate = topology & (ghost_ratio > 5.0) & (relative_difference <= 0.8 * rtol)
    certified = (
        jnp.isfinite(fine.magnification)
        & jnp.isfinite(relative_difference)
        & (regular_certificate | topology_certificate)
    )
    estimated_error = jnp.where(
        certified,
        rtol * scale,
        jnp.maximum(1.5 * tier_difference, rtol * scale),
    )
    return fine._replace(
        estimated_error=estimated_error,
        status=jnp.where(
            certified,
            jnp.int32(0),
            jnp.bitwise_or(fine.status, jnp.int32(ANGULAR_MOMENT_EXHAUSTED)),
        ),
    )


def mag_uniform_cartesian_moment(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    axis: complex | Array = 1.0 + 0.0j,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Evaluate the certified primary Cartesian-strip CPU ICRS."""

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    support = _cartesian_support_cells(w_center, rho, s=s, q=q, n_limb=128, axis=axis)
    result = _mag_uniform_cartesian_moment_from_support(w_center, rho, s=s, q=q, rtol=rtol, support=support, axis=axis)
    return result if return_info else result.magnification


def _cartesian_cross_result(
    primary: CartesianMomentResult,
    orthogonal: CartesianMomentResult,
    primary_support: tuple[Array, Array, Array, Array, Array],
    orthogonal_support: tuple[Array, Array, Array, Array, Array],
    *,
    rtol: Array,
    cross_fraction: Array,
    topology_cross_fraction: Array,
) -> CartesianMomentResult:
    """Combine two orthogonal projection results into one certificate."""

    magnification = 0.5 * (primary.magnification + orthogonal.magnification)
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(primary.magnification), jnp.abs(orthogonal.magnification)),
        1.0,
    )
    cross_difference = jnp.abs(primary.magnification - orthogonal.magnification)
    finite = jnp.isfinite(primary.magnification) & jnp.isfinite(orthogonal.magnification)
    roots_valid = (primary.invalid_root_count == 0) & (orthogonal.invalid_root_count == 0)
    topology = primary_support[2] | orthogonal_support[2]
    effective_cross_fraction = jnp.where(
        primary.limb_topology | orthogonal.limb_topology,
        jnp.minimum(cross_fraction, topology_cross_fraction),
        cross_fraction,
    )
    certified = (
        finite
        & roots_valid
        & (effective_cross_fraction >= 0.0)
        & (cross_difference <= effective_cross_fraction * rtol * scale)
    )
    failed_status = jnp.bitwise_or(
        jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
    )
    return CartesianMomentResult(
        magnification,
        jnp.maximum(0.5 * cross_difference, rtol * scale),
        primary.n_slices + orthogonal.n_slices,
        primary.invalid_root_count + orthogonal.invalid_root_count,
        jnp.minimum(primary.ghost_residual_ratio, orthogonal.ghost_residual_ratio),
        primary.limb_topology | orthogonal.limb_topology,
        jnp.where(certified, jnp.int32(0), failed_status),
    )


def _cartesian_consensus_result(
    projected: CartesianMomentResult,
    *,
    topology: Array,
    rtol: Array,
    consensus_fraction: Array,
) -> CartesianMomentResult:
    """Certify the median of four already evaluated projections."""

    values = jnp.sort(projected.magnification)
    magnification = 0.5 * (values[1] + values[2])
    scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
    lower_span = values[2] - values[0]
    upper_span = values[3] - values[1]
    three_span = jnp.minimum(lower_span, upper_span)
    roots_valid = jnp.sum(projected.invalid_root_count) == 0
    finite = jnp.all(jnp.isfinite(values))
    certified = finite & roots_valid & (consensus_fraction >= 0.0) & (three_span <= consensus_fraction * rtol * scale)
    failed_status = jnp.bitwise_or(
        jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
    )
    return CartesianMomentResult(
        magnification,
        jnp.maximum(0.5 * three_span, rtol * scale),
        jnp.sum(projected.n_slices, dtype=jnp.int32),
        jnp.sum(projected.invalid_root_count, dtype=jnp.int32),
        jnp.min(projected.ghost_residual_ratio),
        jnp.any(projected.limb_topology),
        jnp.where(certified, jnp.int32(0), failed_status),
    )


def mag_uniform_cartesian_cross_moment(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    cross_fraction: float | Array = 0.10,
    topology_cross_fraction: float | Array = 0.05,
    n_slice: int = 12,
    n_limb: int = 128,
    axis: complex | Array = 1.0 + 0.0j,
    continuation: bool = False,
    _supports: tuple | None = None,
    _trace: tuple | None = None,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Certify a strip integral by agreement of orthogonal projections.

    Both projections integrate the same image area but have independent strip
    topology and endpoint quadrature errors.  The expensive source-limb trace is
    shared, so the second projection only adds support construction and strip
    polynomial solves.
    """

    if n_slice <= 0:
        raise ValueError("n_slice must be positive")
    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    cross_fraction = jnp.asarray(cross_fraction, dtype=w_center.real.dtype)
    topology_cross_fraction = jnp.asarray(topology_cross_fraction, dtype=w_center.real.dtype)
    if (_supports is not None) and (_trace is not None):
        raise ValueError("pass either precomputed supports or a shared trace")
    if _supports is not None:
        (
            primary_support,
            orthogonal_support,
            axis,
            image_limb,
            physical_mask,
        ) = _supports
    elif _trace is not None:
        image_limb, physical_mask, topology, ghost, limb_topology = _trace
        axis = _normalized_axis(axis, w_center.dtype)
        common = {
            "topology_uncertain": topology,
            "minimum_ghost_residual": ghost,
            "limb_topology": limb_topology,
        }
        primary_support = _cartesian_support_cells_from_trace(image_limb, physical_mask, axis=axis, **common)
        orthogonal_support = _cartesian_support_cells_from_trace(image_limb, physical_mask, axis=1.0j * axis, **common)
    elif continuation == "seeded":
        (
            primary_support,
            orthogonal_support,
            axis,
            image_limb,
            physical_mask,
        ) = _cartesian_orthogonal_supports(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=n_limb,
            axis=axis,
            return_trace=True,
        )
    else:
        primary_support, orthogonal_support, axis = _cartesian_orthogonal_supports(
            w_center, rho, s=s, q=q, n_limb=n_limb, axis=axis
        )
        image_limb = None
        physical_mask = None
    primary = _cartesian_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_slice=n_slice,
        support=primary_support,
        axis=axis,
        continuation=continuation,
        image_limb=image_limb,
        physical_mask=physical_mask,
    )
    orthogonal = _cartesian_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_slice=n_slice,
        support=orthogonal_support,
        axis=1.0j * axis,
        continuation=continuation,
        image_limb=image_limb,
        physical_mask=physical_mask,
    )
    result = _cartesian_cross_result(
        primary,
        orthogonal,
        primary_support,
        orthogonal_support,
        rtol=rtol,
        cross_fraction=cross_fraction,
        topology_cross_fraction=topology_cross_fraction,
    )
    return result if return_info else result.magnification


def mag_uniform_cartesian_consensus(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    consensus_fraction: float | Array = 0.10,
    n_slice: int = 12,
    n_limb: int = 128,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Certify the median of four projections when three tightly agree."""

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    consensus_fraction = jnp.asarray(consensus_fraction, dtype=w_center.real.dtype)
    axes = jnp.exp(1.0j * jnp.deg2rad(jnp.asarray([0.0, 22.5, 45.0, 67.5], dtype=w_center.real.dtype)))

    def projection(axis):
        return mag_uniform_cartesian_moment_fixed(
            w_center,
            rho,
            s=s,
            q=q,
            n_slice=n_slice,
            n_limb=n_limb,
            axis=axis,
            return_info=True,
        )

    projected = jax.lax.map(projection, axes)
    values = jnp.sort(projected.magnification)
    magnification = 0.5 * (values[1] + values[2])
    scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
    lower_span = values[2] - values[0]
    upper_span = values[3] - values[1]
    three_span = jnp.minimum(lower_span, upper_span)
    roots_valid = jnp.sum(projected.invalid_root_count) == 0
    finite = jnp.all(jnp.isfinite(values))
    certified = finite & roots_valid & (consensus_fraction >= 0.0) & (three_span <= consensus_fraction * rtol * scale)
    topology = jnp.any(projected.status == ANGULAR_MOMENT_TOPOLOGY)
    failed_status = jnp.bitwise_or(
        jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
    )
    result = CartesianMomentResult(
        magnification,
        # A successful consensus certifies agreement, not a strict interval
        # enclosure.  Report the requested tolerance as the calibrated error
        # floor so diagnostics never imply a stronger bound than the
        # scheduler actually requested.
        jnp.maximum(0.5 * three_span, rtol * scale),
        jnp.sum(projected.n_slices, dtype=jnp.int32),
        jnp.sum(projected.invalid_root_count, dtype=jnp.int32),
        jnp.min(projected.ghost_residual_ratio),
        jnp.any(projected.limb_topology),
        jnp.where(certified, jnp.int32(0), failed_status),
    )
    return result if return_info else result.magnification


def mag_uniform_cartesian_bernstein_consensus(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    consensus_fraction: float | Array = 0.10,
    n_slice: int = 12,
    n_limb: int = 192,
    _trace: tuple | None = None,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Certify four root-free projections using one shared limb trace."""

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    consensus_fraction = jnp.asarray(consensus_fraction, dtype=w_center.real.dtype)
    if _trace is None:
        image_limb, physical_mask = trace_binary_source_limb(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=n_limb,
            include_all_roots=False,
        )
        topology, ghost, limb_topology = _cartesian_trace_diagnostics(
            w_center,
            rho,
            s=s,
            q=q,
            image_limb=image_limb,
            physical_mask=physical_mask,
        )
    else:
        image_limb, physical_mask, topology, ghost, limb_topology = _trace
    common = {
        "topology_uncertain": topology,
        "minimum_ghost_residual": ghost,
        "limb_topology": limb_topology,
    }
    axes = jnp.exp(1.0j * jnp.deg2rad(jnp.asarray([0.0, 22.5, 45.0, 67.5], dtype=w_center.real.dtype)))

    def projection(axis):
        support = _cartesian_support_cells_from_trace(
            image_limb,
            physical_mask,
            axis=axis,
            **common,
        )
        return _cartesian_result_from_support(
            w_center,
            rho,
            s=s,
            q=q,
            n_slice=n_slice,
            support=support,
            axis=axis,
            continuation="bernstein_adaptive",
        )

    projected = jax.lax.map(projection, axes)
    values = jnp.sort(projected.magnification)
    magnification = 0.5 * (values[1] + values[2])
    scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
    lower_span = values[2] - values[0]
    upper_span = values[3] - values[1]
    three_span = jnp.minimum(lower_span, upper_span)
    roots_valid = jnp.sum(projected.invalid_root_count) == 0
    finite = jnp.all(jnp.isfinite(values))
    certified = finite & roots_valid & (consensus_fraction >= 0.0) & (three_span <= consensus_fraction * rtol * scale)
    failed_status = jnp.bitwise_or(
        jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
    )
    result = CartesianMomentResult(
        magnification,
        jnp.maximum(0.5 * three_span, rtol * scale),
        jnp.sum(projected.n_slices, dtype=jnp.int32),
        jnp.sum(projected.invalid_root_count, dtype=jnp.int32),
        jnp.min(projected.ghost_residual_ratio),
        jnp.any(projected.limb_topology),
        jnp.where(certified, jnp.int32(0), failed_status),
    )
    return result if return_info else result.magnification


def mag_uniform_cartesian_moment_refined(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    coarse_magnification: Array | None = None,
    _support: tuple[Array, Array, Array, Array, Array] | None = None,
    axis: complex | Array = 1.0 + 0.0j,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Refine difficult Cartesian-strip cells with an 8/12-point pair."""

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    support = _cartesian_support_cells(w_center, rho, s=s, q=q, n_limb=128, axis=axis) if _support is None else _support
    medium = _cartesian_result_from_support(w_center, rho, s=s, q=q, n_slice=8, support=support, axis=axis)
    if coarse_magnification is None:
        coarse_magnification = _cartesian_result_from_support(
            w_center, rho, s=s, q=q, n_slice=6, support=support, axis=axis
        ).magnification
    scale = jnp.maximum(jnp.abs(medium.magnification), 1.0)
    difference = jnp.abs(medium.magnification - coarse_magnification)
    structural = jnp.bitwise_and(medium.status, jnp.bitwise_not(jnp.int32(ANGULAR_MOMENT_TOPOLOGY)))
    estimated_error = jnp.maximum(2.0 * difference, 0.75 * rtol * scale)
    certified = (structural == 0) & jnp.isfinite(medium.magnification) & (estimated_error <= rtol * scale)

    def use_medium(_):
        return medium._replace(estimated_error=estimated_error, status=jnp.int32(0))

    def use_high(_):
        high = _cartesian_result_from_support(w_center, rho, s=s, q=q, n_slice=12, support=support, axis=axis)
        high_scale = jnp.maximum(jnp.abs(high.magnification), 1.0)
        high_difference = jnp.abs(high.magnification - medium.magnification)
        high_structural = jnp.bitwise_and(high.status, jnp.bitwise_not(jnp.int32(ANGULAR_MOMENT_TOPOLOGY)))
        high_error = jnp.maximum(2.0 * high_difference, 0.75 * rtol * high_scale)
        high_certified = (high_structural == 0) & jnp.isfinite(high.magnification) & (high_error <= rtol * high_scale)
        return high._replace(
            estimated_error=high_error,
            status=jnp.where(
                high_certified,
                jnp.int32(0),
                jnp.bitwise_or(high.status, jnp.int32(ANGULAR_MOMENT_EXHAUSTED)),
            ),
        )

    result = jax.lax.cond(certified, use_medium, use_high, operand=None)
    return result if return_info else result.magnification


def _mag_uniform_cartesian_cpu_adaptive_noncentral(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    n_limb: int = 64,
    continuation: bool | str = "bernstein_dynamic",
    maximum_extrema: int = 20,
    external_magnification: Array | None = None,
    _trace=None,
) -> CartesianAdaptiveResult:
    """Integrate one noncentral source with the Cartesian-first CPU schedule.

    A high-quality dynamic projection is paired with a cheaper 22.5-degree
    scout. Disagreement adds the lens-axis projection; topology points may add
    two rotations and require a cluster of three non-parallel axes. If the
    Cartesian views remain ill-conditioned, the default graph changes to a
    polar radial-moment chart with its 64-sample support trace. The stricter
    ``1e-4`` continuation is compiled separately and adds a conditional
    high-order Cartesian pass.
    """

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    safe_magnitude = jnp.maximum(jnp.abs(w_center), jnp.finfo(w_center.real.dtype).tiny)
    base_axis = jnp.where(
        jnp.abs(w_center) > 0.0,
        1.0j * w_center / safe_magnitude,
        jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype),
    )
    offsets = jnp.deg2rad(jnp.asarray([0.0, 22.5, 45.0, 67.5], dtype=w_center.real.dtype))
    axes = base_axis * jnp.exp(1.0j * offsets)

    scout_n_limb = min(
        n_limb,
        64 if continuation == "bernstein_dynamic_full" else 52,
    )
    if _trace is None:
        image_limb, physical_mask = trace_binary_source_limb(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=scout_n_limb,
            include_all_roots=False,
        )
        trace_neighbors = tracked_limb_neighbors(image_limb, physical_mask)
        topology, ghost, limb_topology = _cartesian_trace_diagnostics(
            w_center,
            rho,
            s=s,
            q=q,
            image_limb=image_limb,
            physical_mask=physical_mask,
        )
    else:
        (
            image_limb,
            physical_mask,
            trace_neighbors,
            topology,
            ghost,
            limb_topology,
        ) = _trace
    common = {
        # This scheduler uses independent projections for topology.  Reserve
        # the support status bit for fixed-capacity overflow so a smaller
        # common-case cell buffer can fail closed.
        "topology_uncertain": jnp.asarray(False),
        "minimum_ghost_residual": ghost,
        "limb_topology": limb_topology,
        "maximum_extrema": maximum_extrema,
    }
    primary_support = _cartesian_support_cells_from_trace(
        image_limb,
        physical_mask,
        axis=axes[0],
        neighbors=trace_neighbors,
        **common,
    )
    require_valid_roots = continuation != "polyline"

    def evaluate_projection(support, axis, n_slice: int) -> CartesianMomentResult:
        def evaluate(selected_continuation):
            return _cartesian_result_from_support(
                w_center,
                rho,
                s=s,
                q=q,
                n_slice=n_slice,
                support=support,
                axis=axis,
                continuation=selected_continuation,
                image_limb=image_limb,
                physical_mask=physical_mask,
            )

        return evaluate(continuation)

    def pair_certified(first, second, fraction):
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(first.magnification), jnp.abs(second.magnification)),
            1.0,
        )
        difference = jnp.abs(first.magnification - second.magnification)
        roots_valid = ((first.invalid_root_count == 0) & (second.invalid_root_count == 0)) | ~jnp.asarray(
            require_valid_roots
        )
        valid = (
            roots_valid
            & (first.status == 0)
            & (second.status == 0)
            & jnp.isfinite(first.magnification)
            & jnp.isfinite(second.magnification)
        )
        return valid & (difference <= fraction * rtol * scale), difference, scale

    def three_chart_consensus(first, second, third):
        """Select the best-conditioned agreeing pair of three projections."""

        values = jnp.stack((first.magnification, second.magnification, third.magnification))
        roots_valid = jnp.stack(
            (
                first.invalid_root_count == 0,
                second.invalid_root_count == 0,
                third.invalid_root_count == 0,
            )
        ) | ~jnp.asarray(require_valid_roots)
        projection_valid = (
            roots_valid & jnp.stack((first.status == 0, second.status == 0, third.status == 0)) & jnp.isfinite(values)
        )
        lower = jnp.asarray([0, 0, 1], dtype=jnp.int32)
        upper = jnp.asarray([1, 2, 2], dtype=jnp.int32)
        pair_scale = jnp.maximum(
            jnp.maximum(jnp.abs(values[lower]), jnp.abs(values[upper])),
            1.0,
        )
        pair_difference = jnp.abs(values[lower] - values[upper])
        pair_valid = projection_valid[lower] & projection_valid[upper]
        relative_difference = jnp.where(
            pair_valid,
            pair_difference / pair_scale,
            jnp.inf,
        )
        selected = jnp.argmin(jax.lax.stop_gradient(relative_difference))
        magnification = 0.5 * (values[lower[selected]] + values[upper[selected]])
        selected_difference = pair_difference[selected]
        selected_scale = pair_scale[selected]
        all_valid = jnp.all(projection_valid)
        all_span = jnp.max(values) - jnp.min(values)
        topology_consensus = all_valid & (all_span <= 0.20 * rtol * jnp.maximum(jnp.max(jnp.abs(values)), 1.0))
        certified = (
            pair_valid[selected]
            & (relative_difference[selected] <= 0.20 * rtol)
            # A topology-changing trace cannot be certified by the closest
            # pair alone: two axes can share the same fold-area bias. Permit
            # the fast result only when all three already-computed charts
            # agree at a substantially tighter threshold.
            & (~limb_topology | topology_consensus)
        )
        if continuation == "bernstein_dynamic_full":
            certified = jnp.asarray(False)
        return CartesianAdaptiveResult(
            magnification,
            jnp.maximum(
                0.5 * selected_difference,
                0.75 * rtol * selected_scale,
            ),
            first.n_slices + second.n_slices + third.n_slices,
            jnp.int32(1),
            jnp.where(
                certified,
                jnp.int32(0),
                jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
            ),
        )

    primary8 = evaluate_projection(primary_support, axes[0], 8)

    def cross_schedule(_):
        scout_support = _cartesian_support_cells_from_trace(
            image_limb,
            physical_mask,
            axis=axes[1],
            neighbors=trace_neighbors,
            **common,
        )
        scout6 = evaluate_projection(scout_support, axes[1], 6)
        first_ok, first_difference, first_scale = pair_certified(
            primary8, scout6, jnp.asarray(0.25, dtype=w_center.real.dtype)
        )
        # A topology-changing trace needs the third, lens-axis view; two
        # projections alone do not exclude correlated fold-area error.
        first_ok = first_ok & ~limb_topology
        if continuation == "bernstein_dynamic_full":
            first_ok = jnp.asarray(False)
        first_nodes = primary8.n_slices + scout6.n_slices

        def later(_):
            lens_axis = jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype)
            lens_support = _cartesian_support_cells_from_trace(
                image_limb,
                physical_mask,
                axis=lens_axis,
                neighbors=trace_neighbors,
                **common,
            )
            lens8 = evaluate_projection(lens_support, lens_axis, 8)
            consensus = three_chart_consensus(primary8, scout6, lens8)

            if continuation != "bernstein_dynamic_full":

                def use_polar(_):
                    polar_n_limb = min(n_limb, 64)
                    if polar_n_limb == scout_n_limb:
                        polar_limb = image_limb
                        polar_mask = physical_mask
                        polar_topology = topology
                        polar_ghost = ghost
                        polar_limb_topology = limb_topology
                        polar_neighbors = trace_neighbors
                    else:
                        polar_limb, polar_mask = trace_binary_source_limb(
                            w_center,
                            rho,
                            s=s,
                            q=q,
                            n_limb=polar_n_limb,
                            include_all_roots=False,
                        )
                        polar_topology, polar_ghost, polar_limb_topology = _cartesian_trace_diagnostics(
                            w_center,
                            rho,
                            s=s,
                            q=q,
                            image_limb=polar_limb,
                            physical_mask=polar_mask,
                        )
                        polar_neighbors = tracked_limb_neighbors(polar_limb, polar_mask)
                    angular_support = _angular_support_cells_from_trace(
                        w_center,
                        rho,
                        s=s,
                        q=q,
                        physical_limb=polar_limb,
                        physical_mask=polar_mask,
                        topology_uncertain=polar_topology,
                        minimum_ghost_residual=polar_ghost,
                        limb_topology=polar_limb_topology,
                        neighbors=polar_neighbors,
                    )

                    def evaluate_polar(support):
                        return mag_uniform_angular_moment_refined(
                            w_center,
                            rho,
                            s=s,
                            q=q,
                            rtol=rtol,
                            _support=support,
                            return_info=True,
                        )

                    polar = evaluate_polar(angular_support)
                    return CartesianAdaptiveResult(
                        polar.magnification,
                        polar.estimated_error,
                        consensus.n_slices + polar.n_theta,
                        jnp.int32(4),
                        polar.status,
                    )

                def topology_five_chart(_):
                    # Three charts can contain one correlated pair and one
                    # independent outlier. Two additional rotations make a
                    # three-chart cluster possible without trusting the
                    # closest pair. This remains cheaper than the polar chart
                    # on topology points whose image geometry is Cartesian-
                    # friendly, and rejects the known 2-of-5 biased clusters.
                    support45 = _cartesian_support_cells_from_trace(
                        image_limb,
                        physical_mask,
                        axis=axes[2],
                        neighbors=trace_neighbors,
                        **common,
                    )
                    support67 = _cartesian_support_cells_from_trace(
                        image_limb,
                        physical_mask,
                        axis=axes[3],
                        neighbors=trace_neighbors,
                        **common,
                    )
                    rotated45 = evaluate_projection(support45, axes[2], 8)
                    rotated67 = evaluate_projection(support67, axes[3], 8)
                    chart_values = jnp.stack(
                        (
                            primary8.magnification,
                            scout6.magnification,
                            lens8.magnification,
                            rotated45.magnification,
                            rotated67.magnification,
                        )
                    )
                    chart_axes = jnp.stack(
                        (
                            axes[0],
                            axes[1],
                            jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype),
                            axes[2],
                            axes[3],
                        )
                    )
                    chart_valid = jnp.stack(
                        (
                            (primary8.invalid_root_count == 0) & (primary8.status == 0),
                            (scout6.invalid_root_count == 0) & (scout6.status == 0),
                            (lens8.invalid_root_count == 0) & (lens8.status == 0),
                            (rotated45.invalid_root_count == 0) & (rotated45.status == 0),
                            (rotated67.invalid_root_count == 0) & (rotated67.status == 0),
                        )
                    ) & jnp.isfinite(chart_values)
                    triples = jnp.asarray(
                        (
                            (0, 1, 2),
                            (0, 1, 3),
                            (0, 1, 4),
                            (0, 2, 3),
                            (0, 2, 4),
                            (0, 3, 4),
                            (1, 2, 3),
                            (1, 2, 4),
                            (1, 3, 4),
                            (2, 3, 4),
                        ),
                        dtype=jnp.int32,
                    )
                    triple_values = chart_values[triples]
                    triple_axes = chart_axes[triples]
                    # Projection axes are unoriented: a and -a are the same
                    # chart. Do not let an accidental duplicate count twice
                    # toward a 3-chart certificate (e.g. on the x=y diagonal
                    # the lens axis and the nominal 45-degree chart coincide).
                    axis_separation = jnp.stack(
                        (
                            jnp.abs(jnp.imag(triple_axes[:, 0] * jnp.conjugate(triple_axes[:, 1]))),
                            jnp.abs(jnp.imag(triple_axes[:, 0] * jnp.conjugate(triple_axes[:, 2]))),
                            jnp.abs(jnp.imag(triple_axes[:, 1] * jnp.conjugate(triple_axes[:, 2]))),
                        ),
                        axis=1,
                    )
                    independent_axes = jnp.all(axis_separation >= jnp.sin(jnp.deg2rad(10.0)), axis=1)
                    triple_valid = jnp.all(chart_valid[triples], axis=1) & independent_axes
                    triple_span = jnp.max(triple_values, axis=1) - jnp.min(triple_values, axis=1)
                    triple_scale = jnp.maximum(jnp.max(jnp.abs(triple_values), axis=1), 1.0)
                    relative_span = jnp.where(triple_valid, triple_span / triple_scale, jnp.inf)
                    selected_triple = jnp.argmin(jax.lax.stop_gradient(relative_span))
                    selected_values = triple_values[selected_triple]
                    selected_scale = triple_scale[selected_triple]
                    selected_span = triple_span[selected_triple]
                    certified = triple_valid[selected_triple] & (relative_span[selected_triple] <= 0.20 * rtol)
                    clustered = CartesianAdaptiveResult(
                        jnp.median(selected_values),
                        jnp.maximum(selected_span, 0.75 * rtol * selected_scale),
                        consensus.n_slices + rotated45.n_slices + rotated67.n_slices,
                        jnp.int32(2),
                        jnp.where(
                            certified,
                            jnp.int32(0),
                            jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
                        ),
                    )
                    return jax.lax.cond(
                        certified,
                        lambda _: clustered,
                        use_polar,
                        operand=None,
                    )

                return jax.lax.cond(
                    consensus.status == 0,
                    lambda _: consensus,
                    lambda _: jax.lax.cond(
                        topology,
                        topology_five_chart,
                        use_polar,
                        operand=None,
                    ),
                    operand=None,
                )

            def refine(_):
                if n_limb == scout_n_limb:
                    refined_image_limb = image_limb
                    refined_physical_mask = physical_mask
                    refined_topology = topology
                    refined_ghost = ghost
                    refined_limb_topology = limb_topology
                    refined_neighbors = trace_neighbors
                else:
                    refined_image_limb, refined_physical_mask = trace_binary_source_limb(
                        w_center,
                        rho,
                        s=s,
                        q=q,
                        n_limb=n_limb,
                        include_all_roots=False,
                    )
                    (
                        refined_topology,
                        refined_ghost,
                        refined_limb_topology,
                    ) = _cartesian_trace_diagnostics(
                        w_center,
                        rho,
                        s=s,
                        q=q,
                        image_limb=refined_image_limb,
                        physical_mask=refined_physical_mask,
                    )
                    refined_neighbors = tracked_limb_neighbors(refined_image_limb, refined_physical_mask)
                refined_common = {
                    "topology_uncertain": jnp.asarray(False),
                    "minimum_ghost_residual": refined_ghost,
                    "limb_topology": refined_limb_topology,
                    "maximum_extrema": maximum_extrema,
                }
                refined_primary_support = _cartesian_support_cells_from_trace(
                    refined_image_limb,
                    refined_physical_mask,
                    axis=axes[0],
                    neighbors=refined_neighbors,
                    **refined_common,
                )
                refined_scout_support = _cartesian_support_cells_from_trace(
                    refined_image_limb,
                    refined_physical_mask,
                    axis=axes[1],
                    neighbors=refined_neighbors,
                    **refined_common,
                )

                def evaluate_refined(support, axis):
                    return _cartesian_result_from_support(
                        w_center,
                        rho,
                        s=s,
                        q=q,
                        n_slice=12,
                        support=support,
                        axis=axis,
                        continuation=continuation,
                        image_limb=refined_image_limb,
                        physical_mask=refined_physical_mask,
                    )

                primary12 = evaluate_refined(refined_primary_support, axes[0])
                scout12 = evaluate_refined(refined_scout_support, axes[1])
                second_ok, second_difference, second_scale = pair_certified(
                    primary12, scout12, jnp.asarray(0.10, dtype=w_center.real.dtype)
                )
                # When the source limb changes image multiplicity, two strip
                # projections can share the same fold-area bias and agree at
                # high order.  Such points require the independent polar
                # chart; rho=0.02 central-caustic sweeps expose percent-level
                # false certificates if Cartesian agreement is accepted here.
                second_ok = second_ok & ~refined_limb_topology
                second_nodes = consensus.n_slices + primary12.n_slices + scout12.n_slices

                def later_pair(_):
                    # The same converged order-12 pair supplies a final,
                    # looser certificate before changing coordinate chart.
                    final_ok, final_difference, final_scale = pair_certified(
                        primary12,
                        scout12,
                        jnp.asarray(
                            1.0 if continuation == "bernstein_dynamic_full" else 1.5,
                            dtype=w_center.real.dtype,
                        ),
                    )
                    final_ok = final_ok & ~refined_limb_topology

                    def use_cartesian(_):
                        return CartesianAdaptiveResult(
                            0.5 * (primary12.magnification + scout12.magnification),
                            jnp.maximum(
                                0.5 * final_difference,
                                0.75 * rtol * final_scale,
                            ),
                            second_nodes,
                            jnp.int32(3),
                            jnp.int32(0),
                        )

                    def use_polar(_):
                        angular_support = _angular_support_cells_from_trace(
                            w_center,
                            rho,
                            s=s,
                            q=q,
                            physical_limb=refined_image_limb,
                            physical_mask=refined_physical_mask,
                            topology_uncertain=refined_topology,
                            minimum_ghost_residual=refined_ghost,
                            limb_topology=refined_limb_topology,
                            neighbors=refined_neighbors,
                        )
                        polar = mag_uniform_angular_moment_compact(
                            w_center,
                            rho,
                            s=s,
                            q=q,
                            rtol=rtol,
                            _support=angular_support,
                            return_info=True,
                        )
                        return CartesianAdaptiveResult(
                            polar.magnification,
                            polar.estimated_error,
                            second_nodes + polar.n_theta,
                            jnp.int32(4),
                            polar.status,
                        )

                    return jax.lax.cond(
                        final_ok,
                        use_cartesian,
                        use_polar,
                        operand=None,
                    )

                return jax.lax.cond(
                    second_ok,
                    lambda _: CartesianAdaptiveResult(
                        0.5 * (primary12.magnification + scout12.magnification),
                        jnp.maximum(
                            0.5 * second_difference,
                            0.75 * rtol * second_scale,
                        ),
                        second_nodes,
                        jnp.int32(2),
                        jnp.int32(0),
                    ),
                    later_pair,
                    operand=None,
                )

            return jax.lax.cond(
                consensus.status == 0,
                lambda _: consensus,
                refine,
                operand=None,
            )

        return jax.lax.cond(
            first_ok,
            lambda _: CartesianAdaptiveResult(
                primary8.magnification,
                jnp.maximum(first_difference, 0.75 * rtol * first_scale),
                first_nodes,
                jnp.int32(1),
                jnp.int32(0),
            ),
            later,
            operand=None,
        )

    if external_magnification is None:
        return cross_schedule(None)

    external_magnification = jnp.asarray(external_magnification, dtype=w_center.real.dtype)
    external_scale = jnp.maximum(
        jnp.maximum(jnp.abs(primary8.magnification), jnp.abs(external_magnification)),
        1.0,
    )
    external_difference = jnp.abs(primary8.magnification - external_magnification)
    external_ok = (
        (primary8.invalid_root_count == 0)
        & (primary8.status == 0)
        & jnp.isfinite(primary8.magnification)
        & jnp.isfinite(external_magnification)
        & (external_difference <= 0.25 * rtol * external_scale)
    )
    return jax.lax.cond(
        external_ok,
        lambda _: CartesianAdaptiveResult(
            primary8.magnification,
            jnp.maximum(external_difference, 0.75 * rtol * external_scale),
            primary8.n_slices,
            jnp.int32(1),
            jnp.int32(0),
        ),
        cross_schedule,
        operand=None,
    )


def mag_uniform_cartesian_cpu_adaptive(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    n_limb: int = 64,
    continuation: bool | str = "bernstein_dynamic",
    maximum_extrema: int = 20,
    external_magnification: Array | None = None,
) -> CartesianAdaptiveResult:
    """Select the well-conditioned central-polar or Cartesian CPU chart.

    When traced images form long, thin arcs close to an Einstein ring, several
    Cartesian projections share a trace-discretization bias and are neither
    fast nor independent certificates.  A polar chart reuses that same trace
    and is both faster and better conditioned in that regime.  All other
    points retain the Cartesian-first graph.
    """

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)

    # Keep the separately compiled high-accuracy continuation unchanged until
    # its 1e-4 certificate is calibrated independently.  The Roman-default
    # gate is intentionally geometric and virtually free compared with the
    # source-limb solve that follows either branch.
    if continuation != "bernstein_dynamic":
        return _mag_uniform_cartesian_cpu_adaptive_noncentral(
            w_center,
            rho,
            s=s,
            q=q,
            rtol=rtol,
            n_limb=n_limb,
            continuation=continuation,
            maximum_extrema=maximum_extrema,
            external_magnification=external_magnification,
        )

    scout_n_limb = min(n_limb, 52)
    image_limb, physical_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=scout_n_limb,
        include_all_roots=False,
    )
    neighbors = tracked_limb_neighbors(image_limb, physical_mask)
    topology, ghost, limb_topology = _cartesian_trace_diagnostics(
        w_center,
        rho,
        s=s,
        q=q,
        image_limb=image_limb,
        physical_mask=physical_mask,
    )
    shared_trace = (
        image_limb,
        physical_mask,
        neighbors,
        topology,
        ghost,
        limb_topology,
    )

    # Measure chart conditioning from image motion already present in the
    # source-limb trace.  Arc length around the polar origin is compared with
    # radial motion of the same tracked edges.  A ratio above 512 identifies
    # an annular image whose Cartesian strip charts share almost the same
    # support error.  The reduction is negligible beside the root trace and
    # avoids an extra lens-equation solve in either branch.
    _, following_limb, _, following_mask = neighbors
    connected = physical_mask & following_mask
    radius = jnp.abs(image_limb)
    following_radius = jnp.abs(following_limb)
    angular_step = jnp.abs(jnp.angle(following_limb * jnp.conjugate(image_limb)))
    tangential_motion = jnp.sum(
        jnp.where(
            connected,
            0.5 * (radius + following_radius) * angular_step,
            0.0,
        )
    )
    radial_motion = jnp.sum(jnp.where(connected, jnp.abs(following_radius - radius), 0.0))
    motion_floor = 128.0 * jnp.finfo(w_center.real.dtype).eps
    polar_conditioning = tangential_motion / jnp.maximum(
        radial_motion,
        motion_floor,
    )
    use_central_polar = jax.lax.stop_gradient(polar_conditioning >= 512.0)

    def central_polar(_):
        support = _angular_support_cells_from_trace(
            w_center,
            rho,
            s=s,
            q=q,
            physical_limb=image_limb,
            physical_mask=physical_mask,
            topology_uncertain=topology,
            minimum_ghost_residual=ghost,
            limb_topology=limb_topology,
            neighbors=neighbors,
        )
        # In the nearly single-lens limit the angular moment converges
        # non-monotonically when the source limb approaches the polar origin.
        # On the compact shared trace, the 12/16 pair can co-converge with a
        # roughly 1.8e-3 common bias. Send only this thin contact band directly
        # to the existing 24/32 pair; central interiors retain the faster
        # low-order hierarchy and the contact band avoids computing both.
        source_ratio = jnp.abs(w_center) / jnp.maximum(
            rho,
            jnp.finfo(rho.dtype).tiny,
        )
        contact_band = jax.lax.stop_gradient((source_ratio >= 0.75) & (source_ratio <= 1.25))

        def solve_polar(selected_support):
            def contact_polar(_):
                return _uniform_contact_refinement(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    rtol=jnp.asarray(rtol),
                    support=selected_support,
                )

            def interior_polar(_):
                return mag_uniform_angular_moment_refined(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    rtol=rtol,
                    _support=selected_support,
                    return_info=True,
                )

            return jax.lax.cond(
                contact_band,
                contact_polar,
                interior_polar,
                operand=None,
            )

        polar = solve_polar(support)
        refined_n_limb = min(n_limb, 64)
        if refined_n_limb > scout_n_limb:

            def refine_failed_support(_):
                refined_limb, refined_mask = trace_binary_source_limb(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    n_limb=refined_n_limb,
                    include_all_roots=False,
                )
                refined_neighbors = tracked_limb_neighbors(
                    refined_limb,
                    refined_mask,
                )
                refined_topology, refined_ghost, refined_limb_topology = _cartesian_trace_diagnostics(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    image_limb=refined_limb,
                    physical_mask=refined_mask,
                )
                refined_support = _angular_support_cells_from_trace(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    physical_limb=refined_limb,
                    physical_mask=refined_mask,
                    topology_uncertain=refined_topology,
                    minimum_ghost_residual=refined_ghost,
                    limb_topology=refined_limb_topology,
                    neighbors=refined_neighbors,
                )
                refined = solve_polar(refined_support)
                return refined._replace(
                    n_theta=polar.n_theta + refined.n_theta,
                )

            polar = jax.lax.cond(
                polar.status != 0,
                refine_failed_support,
                lambda _: polar,
                operand=None,
            )
        return CartesianAdaptiveResult(
            polar.magnification,
            polar.estimated_error,
            polar.n_theta,
            jnp.int32(4),
            polar.status,
        )

    def cartesian_first(_):
        return _mag_uniform_cartesian_cpu_adaptive_noncentral(
            w_center,
            rho,
            s=s,
            q=q,
            rtol=rtol,
            n_limb=n_limb,
            continuation=continuation,
            maximum_extrema=maximum_extrema,
            external_magnification=external_magnification,
            _trace=shared_trace,
        )

    return jax.lax.cond(
        use_central_polar,
        central_polar,
        cartesian_first,
        operand=None,
    )


__all__ = [
    "CartesianAdaptiveResult",
    "CartesianMomentResult",
    "CartesianTopologyProbe",
    "_cartesian_support_cells",
    "_cartesian_orthogonal_supports",
    "_cartesian_support_cells_from_trace",
    "_cartesian_topology_probe_from_support",
    "_mag_uniform_cartesian_moment_from_support",
    "binary_line_level_set_coefficients",
    "mag_uniform_cartesian_moment",
    "mag_uniform_cartesian_cross_moment",
    "mag_uniform_cartesian_consensus",
    "mag_uniform_cartesian_bernstein_consensus",
    "mag_uniform_cartesian_cpu_adaptive",
    "mag_uniform_cartesian_moment_fixed",
    "mag_uniform_cartesian_moment_refined",
]
