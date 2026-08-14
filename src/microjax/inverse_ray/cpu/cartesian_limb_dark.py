"""Cartesian strip-profile ICRS for linear limb darkening on CPU."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry.lens import binary_geometry
from .angular_limb_dark import mag_limb_dark_angular_moment_compact
from .angular_moment import ANGULAR_MOMENT_EXHAUSTED, ANGULAR_MOMENT_TOPOLOGY
from .angular_moment import _angular_support_cells_from_trace
from .angular_profile import (
    _JACOBI4_W,
    _JACOBI4_X,
    _JACOBI8_W,
    _JACOBI8_X,
)
from .cartesian_moment import (
    CartesianAdaptiveResult,
    CartesianMomentResult,
    _cartesian_support_cells,
    _cartesian_support_cells_from_trace,
    _cartesian_trace_diagnostics,
    _bernstein_half_subdivide,
    _fixed_independent_ea_roots,
    _normalized_axis,
    binary_line_level_set_coefficients,
)
from .roots import batched_polished_real_companion_roots
from .quadrature import G7_W_ON_GK15, GK15_W, GK15_X
from .support import trace_binary_source_limb, tracked_limb_neighbors

Array = jnp.ndarray

_SEXTIC_DEGREE = 6
_BERNSTEIN_SAMPLE_FRACTION = np.linspace(0.0, 1.0, _SEXTIC_DEGREE + 1)
_BERNSTEIN_VALUE_MATRIX = np.asarray(
    [
        [
            math.comb(_SEXTIC_DEGREE, index)
            * fraction**index
            * (1.0 - fraction) ** (_SEXTIC_DEGREE - index)
            for index in range(_SEXTIC_DEGREE + 1)
        ]
        for fraction in _BERNSTEIN_SAMPLE_FRACTION
    ]
)
_BERNSTEIN_VALUE_TO_COEFFICIENT = np.linalg.inv(_BERNSTEIN_VALUE_MATRIX).T
_POWER_TO_BERNSTEIN = (
    np.vander(2.0 * _BERNSTEIN_SAMPLE_FRACTION - 1.0, _SEXTIC_DEGREE + 1).T
    @ _BERNSTEIN_VALUE_TO_COEFFICIENT
)


@jax.custom_jvp
def _attach_implicit_root_jvp(coefficients: Array, roots: Array) -> Array:
    """Keep isolated real roots but differentiate the exact root equation."""

    del coefficients
    return roots


@_attach_implicit_root_jvp.defjvp
def _attach_implicit_root_jvp_rule(primals, tangents):
    coefficients, roots = primals
    coefficient_tangent, _ = tangents
    derivative = coefficients[:, :-1] * jnp.arange(
        _SEXTIC_DEGREE, 0, -1, dtype=coefficients.dtype
    )
    numerator = jax.vmap(jnp.polyval)(coefficient_tangent, roots)
    denominator = jax.vmap(jnp.polyval)(derivative, roots)
    root_tangent = -numerator / denominator
    return roots, root_tangent


def _companion_negative_intervals(
    coefficients: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return the at-most-three negative intervals of real sextics."""

    return _root_negative_intervals(
        coefficients,
        batched_polished_real_companion_roots(coefficients),
    )


def _root_negative_intervals(
    coefficients: Array,
    roots: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return negative intervals from an independently computed root set."""

    real = roots.real
    scale = jnp.maximum(1.0, jnp.abs(real))
    residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    nearly_real = jnp.abs(roots.imag) <= 2.0e-7 * scale
    valid = nearly_real & (residual <= 2.0e-7)
    ordered = jnp.sort(jnp.where(valid, real, jnp.inf), axis=-1)
    root_count = jnp.sum(valid, axis=-1, dtype=jnp.int32)
    pair = jnp.arange(0, coefficients.shape[-1] - 1, 2)
    active = pair[None, :] + 1 < root_count[:, None]
    lower = jnp.where(active, ordered[:, pair], 0.0)
    upper = jnp.where(active, ordered[:, pair + 1], 0.0)
    suspicious = (~valid) & nearly_real
    invalid = jnp.sum(suspicious, axis=-1, dtype=jnp.int32) + jnp.mod(
        root_count, 2
    )
    return lower, upper, active, invalid


def _bernstein_negative_intervals(
    coefficients: Array,
    ordinate_bound: Array,
    *,
    max_depth: int = 19,
    capacity: int = 10,
) -> tuple[Array, Array, Array, Array]:
    """Isolate the negative intervals of real sextics without eigensolves."""

    if max_depth <= 0 or capacity < _SEXTIC_DEGREE:
        raise ValueError("Bernstein depth/capacity are too small")
    real_dtype = coefficients.dtype
    n_polynomials = coefficients.shape[0]
    ordinate_bound = jnp.asarray(ordinate_bound, dtype=real_dtype)
    power_scale = ordinate_bound ** jnp.arange(
        _SEXTIC_DEGREE, -1, -1, dtype=real_dtype
    )
    power_to_bernstein = jnp.asarray(_POWER_TO_BERNSTEIN, dtype=real_dtype)
    initial = (coefficients * power_scale[None, :]) @ power_to_bernstein
    bernstein = (
        jnp.zeros((n_polynomials, capacity, _SEXTIC_DEGREE + 1), dtype=real_dtype)
        .at[:, 0, :]
        .set(initial)
    )
    bounds = (
        jnp.zeros((n_polynomials, capacity, 2), dtype=real_dtype)
        .at[:, 0, :]
        .set(jnp.stack((-ordinate_bound, ordinate_bound)))
    )
    active = jnp.zeros((n_polynomials, capacity), dtype=bool).at[:, 0].set(True)
    overflow = jnp.zeros((n_polynomials,), dtype=bool)

    def subdivide(_, state):
        current, current_bounds, current_active, did_overflow = state
        scale = jnp.maximum(jnp.max(jnp.abs(current), axis=-1), 1.0)
        tolerance = 128.0 * jnp.finfo(real_dtype).eps * scale
        sign_definite = jnp.all(current > tolerance[..., None], axis=-1) | jnp.all(
            current < -tolerance[..., None], axis=-1
        )
        mixed = current_active & ~sign_definite
        left, right = _bernstein_half_subdivide(current)
        midpoint = 0.5 * (current_bounds[..., 0] + current_bounds[..., 1])
        left_bounds = jnp.stack((current_bounds[..., 0], midpoint), axis=-1)
        right_bounds = jnp.stack((midpoint, current_bounds[..., 1]), axis=-1)
        children = jnp.stack((left, right), axis=2).reshape(
            n_polynomials, 2 * capacity, _SEXTIC_DEGREE + 1
        )
        child_bounds = jnp.stack((left_bounds, right_bounds), axis=2).reshape(
            n_polynomials, 2 * capacity, 2
        )
        child_active = jnp.repeat(mixed, 2, axis=-1)
        child_count = jnp.sum(child_active, axis=-1, dtype=jnp.int32)
        did_overflow = did_overflow | (child_count > capacity)
        selected_active, selected_index = jax.lax.top_k(
            child_active.astype(jnp.int32), capacity
        )
        gather_index = selected_index[..., None]
        current = jnp.take_along_axis(
            children,
            jnp.broadcast_to(
                gather_index,
                (n_polynomials, capacity, _SEXTIC_DEGREE + 1),
            ),
            axis=1,
        )
        current_bounds = jnp.take_along_axis(
            child_bounds,
            jnp.broadcast_to(gather_index, (n_polynomials, capacity, 2)),
            axis=1,
        )
        return current, current_bounds, selected_active > 0, did_overflow

    bernstein, bounds, active, overflow = jax.lax.fori_loop(
        0,
        max_depth,
        subdivide,
        (bernstein, bounds, active, overflow),
    )
    lower = bounds[..., 0]
    upper = bounds[..., 1]
    midpoint = 0.5 * (lower + upper)
    lower_value = jax.vmap(jnp.polyval)(coefficients, lower)
    midpoint_value = jax.vmap(jnp.polyval)(coefficients, midpoint)
    upper_value = jax.vmap(jnp.polyval)(coefficients, upper)
    left_cross = active & (jnp.signbit(lower_value) != jnp.signbit(midpoint_value))
    right_cross = active & (jnp.signbit(midpoint_value) != jnp.signbit(upper_value))
    # A cell that remains Bernstein-mixed after the final subdivision but has
    # no sampled sign change can contain a tangency or a very close root pair.
    # Neither is allowed to disappear silently: reject this projection so the
    # independent companion pair (and ultimately the polar fallback) handles
    # it.  This is deliberately stricter than assigning the tiny cell by its
    # midpoint, because the LD profile needs explicit interval endpoints.
    coefficient_scale = jnp.maximum(jnp.max(jnp.abs(bernstein), axis=-1), 1.0)
    coefficient_tolerance = (
        128.0 * jnp.finfo(real_dtype).eps * coefficient_scale
    )
    coefficient_sign = jnp.where(
        bernstein > coefficient_tolerance[..., None],
        1,
        jnp.where(bernstein < -coefficient_tolerance[..., None], -1, 0),
    )
    previous_sign = jnp.zeros(coefficient_sign.shape[:-1], dtype=jnp.int32)
    sign_variations = jnp.zeros_like(previous_sign)
    for coefficient_index in range(_SEXTIC_DEGREE + 1):
        current_sign = coefficient_sign[..., coefficient_index]
        sign_variations += (
            (previous_sign != 0)
            & (current_sign != 0)
            & (previous_sign != current_sign)
        ).astype(jnp.int32)
        previous_sign = jnp.where(
            current_sign != 0,
            current_sign,
            previous_sign,
        )
    unresolved = (
        active
        & (sign_variations > 0)
        & ~left_cross
        & ~right_cross
    )
    crossing = jnp.concatenate((left_cross, right_cross), axis=-1)
    bracket_lower = jnp.concatenate((lower, midpoint), axis=-1)
    bracket_upper = jnp.concatenate((midpoint, upper), axis=-1)
    crossing_count = jnp.sum(crossing, axis=-1, dtype=jnp.int32)
    selected, selected_index = jax.lax.top_k(
        crossing.astype(jnp.int32), _SEXTIC_DEGREE
    )
    root_active = selected > 0
    root_lower = jnp.take_along_axis(bracket_lower, selected_index, axis=1)
    root_upper = jnp.take_along_axis(bracket_upper, selected_index, axis=1)
    root_lower_value = jax.vmap(jnp.polyval)(coefficients, root_lower)
    derivative = coefficients[:, :-1] * jnp.arange(
        _SEXTIC_DEGREE, 0, -1, dtype=real_dtype
    )

    def polish(_, state):
        low, high, low_value = state
        center = 0.5 * (low + high)
        center_value = jax.vmap(jnp.polyval)(coefficients, center)
        center_slope = jax.vmap(jnp.polyval)(derivative, center)
        newton = center - center_value / center_slope
        use_newton = (
            root_active
            & jnp.isfinite(newton)
            & (jnp.abs(center_slope) > 64.0 * jnp.finfo(real_dtype).eps)
            & (newton > low)
            & (newton < high)
        )
        candidate = jnp.where(use_newton, newton, center)
        candidate_value = jax.vmap(jnp.polyval)(coefficients, candidate)
        same_side = jnp.signbit(candidate_value) == jnp.signbit(low_value)
        low = jnp.where(root_active & same_side, candidate, low)
        low_value = jnp.where(root_active & same_side, candidate_value, low_value)
        high = jnp.where(root_active & ~same_side, candidate, high)
        return low, high, low_value

    root_lower, root_upper, _ = jax.lax.fori_loop(
        0,
        8,
        polish,
        (root_lower, root_upper, root_lower_value),
    )
    roots = 0.5 * (root_lower + root_upper)
    root_value = jax.vmap(jnp.polyval)(coefficients, roots)
    root_slope = jax.vmap(jnp.polyval)(derivative, roots)
    newton = roots - root_value / root_slope
    roots = jnp.where(
        root_active
        & jnp.isfinite(newton)
        & (jnp.abs(root_slope) > 64.0 * jnp.finfo(real_dtype).eps)
        & (newton > root_lower)
        & (newton < root_upper),
        newton,
        roots,
    )
    roots = _attach_implicit_root_jvp(coefficients, roots)
    residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    root_valid = root_active & jnp.isfinite(roots) & (residual <= 2.0e-7)
    ordered = jnp.sort(jnp.where(root_valid, roots, jnp.inf), axis=-1)
    root_count = jnp.sum(root_valid, axis=-1, dtype=jnp.int32)
    pair = jnp.arange(0, _SEXTIC_DEGREE, 2)
    interval_active = pair[None, :] + 1 < root_count[:, None]
    interval_lower = jnp.where(interval_active, ordered[:, pair], 0.0)
    interval_upper = jnp.where(interval_active, ordered[:, pair + 1], 0.0)
    invalid = (
        overflow.astype(jnp.int32)
        + (crossing_count > _SEXTIC_DEGREE).astype(jnp.int32)
        + jnp.mod(root_count, 2)
        + jnp.sum(root_active & ~root_valid, axis=-1, dtype=jnp.int32)
        + jnp.sum(unresolved, axis=-1, dtype=jnp.int32)
    )
    return interval_lower, interval_upper, interval_active, invalid


def _strip_profile_moments(
    abscissa: Array,
    coefficients: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    axis: Array,
    n_profile: int,
    root_mode: str = "companion",
    ordinate_bound: Array | None = None,
    bernstein_depth: int = 19,
    bernstein_capacity: int = 10,
) -> tuple[Array, Array, Array]:
    if root_mode == "companion":
        lower, upper, active, invalid = _companion_negative_intervals(coefficients)
    elif root_mode.startswith("ea_fixed"):
        if ordinate_bound is None:
            raise ValueError("fixed EA requires ordinate_bound")
        iteration_text = root_mode.removeprefix("ea_fixed")
        iterations = int(iteration_text) if iteration_text else 32
        roots = _fixed_independent_ea_roots(
            coefficients,
            ordinate_bound,
            iterations=iterations,
        )
        lower, upper, active, invalid = _root_negative_intervals(
            coefficients,
            roots,
        )
    elif root_mode == "bernstein":
        if ordinate_bound is None:
            raise ValueError("Bernstein isolation requires ordinate_bound")
        lower, upper, active, invalid = _bernstein_negative_intervals(
            coefficients,
            ordinate_bound,
            max_depth=bernstein_depth,
            capacity=bernstein_capacity,
        )
    else:
        raise ValueError(f"unknown root_mode: {root_mode}")
    if n_profile == 4:
        nodes, weights = _JACOBI4_X, _JACOBI4_W
    elif n_profile == 8:
        nodes, weights = _JACOBI8_X, _JACOBI8_W
    elif n_profile in (5, 6, 7):
        # Gauss--Chebyshev quadrature of the second kind is exactly the
        # Jacobi(alpha=beta=1/2) rule used for the limb profile.  Keep the
        # intermediate orders available for calibration without another
        # SciPy/runtime dependency.
        angle = np.pi * np.arange(n_profile, 0, -1) / (n_profile + 1)
        nodes = np.cos(angle)
        weights = np.pi * np.sin(angle) ** 2 / (n_profile + 1)
    else:
        raise ValueError("n_profile must be between 4 and 8")
    nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
    weights = jnp.asarray(weights, dtype=w_center.real.dtype)
    jacobi_weight = jnp.sqrt((1.0 - nodes) * (1.0 + nodes))
    lens = binary_geometry(s, q)

    def one_strip(x, lo, hi, interval_active):
        midpoint = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo)
        ordinate = midpoint[:, None] + half_width[:, None] * nodes[None, :]
        images = x * axis + ordinate * (1.0j * axis)
        centered = images - lens.shifted
        conjugate = jnp.conjugate(centered)
        mapped = (
            centered
            - lens.e1 / (conjugate - lens.a)
            - (1.0 - lens.e1) / (conjugate + lens.a)
            + lens.shifted
        )
        normalized = jnp.abs(mapped - w_center) / rho
        # Inactive algebraic intervals still pass through this fixed-shape
        # quadrature.  Evaluating ``sqrt(max(1-r**2, 0))`` on those lanes
        # gives the right primal value, but its JVP forms ``0 * inf`` at the
        # clipped square root.  This is especially visible at small q, where
        # the compact planetary intervals leave more inactive slots.  Give
        # every inactive/outside lane a benign radicand, then mask its value;
        # moving active interval endpoints carry the boundary derivative.
        strictly_inside = interval_active[:, None] & (normalized < 1.0)
        safe_radicand = jnp.where(
            strictly_inside,
            (1.0 - normalized) * (1.0 + normalized),
            1.0,
        )
        mu = jnp.where(strictly_inside, jnp.sqrt(safe_radicand), 0.0)
        smooth = mu / jacobi_weight[None, :]
        profile = half_width * jnp.sum(weights[None, :] * smooth, axis=-1)
        uniform = hi - lo
        return (
            jnp.sum(jnp.where(interval_active, uniform, 0.0)),
            jnp.sum(jnp.where(interval_active, profile, 0.0)),
        )

    uniform, profile = jax.vmap(one_strip)(abscissa, lower, upper, active)
    return uniform, profile, invalid


def _mag_limb_dark_cartesian_impl(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    n_slice: int = 8,
    n_profile: int = 4,
    n_limb: int = 64,
    axis: complex | Array = 1.0 + 0.0j,
    support=None,
    root_mode: str = "companion",
    bernstein_depth: int = 19,
    bernstein_capacity: int = 10,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Integrate uniform area and the LD radial profile in the same strips."""

    if n_slice <= 0 or n_limb <= 0:
        raise ValueError("quadrature and support sizes must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    u1 = jnp.asarray(u1, dtype=w_center.real.dtype)
    axis = _normalized_axis(axis, w_center.dtype)
    lens = binary_geometry(s, q)
    lens_radius = jnp.maximum(
        jnp.abs(lens.shifted - lens.a), jnp.abs(lens.shifted + lens.a)
    )
    source_bound = jnp.abs(w_center) + rho
    radial_offset = source_bound - lens_radius
    ordinate_bound = lens_radius + 0.5 * (
        radial_offset + jnp.sqrt(radial_offset**2 + 4.0)
    )
    ordinate_bound *= 1.0 + 32.0 * jnp.finfo(w_center.real.dtype).eps
    if support is None:
        support = _cartesian_support_cells(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=n_limb,
            axis=axis,
        )
    cells, active, topology, ghost, limb_topology = support
    nodes, weights = np.polynomial.legendre.leggauss(n_slice)
    nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
    weights = jnp.asarray(weights, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)

    def integrate_cell(cell_index, state):
        uniform_total, profile_total, invalid_total = state
        bounds = cells[cell_index]
        width = bounds[1] - bounds[0]
        abscissa = bounds[0] + width * jnp.sin(transform) ** 2
        strip_weights = weights * 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
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
        uniform, profile, invalid = _strip_profile_moments(
            abscissa,
            coefficients,
            w_center,
            rho,
            s=s,
            q=q,
            axis=axis,
            n_profile=n_profile,
            root_mode=root_mode,
            ordinate_bound=ordinate_bound,
            bernstein_depth=bernstein_depth,
            bernstein_capacity=bernstein_capacity,
        )
        return (
            uniform_total + jnp.sum(strip_weights * uniform),
            profile_total + jnp.sum(strip_weights * profile),
            invalid_total + jnp.sum(invalid, dtype=jnp.int32),
        )

    uniform, profile, invalid = jax.lax.fori_loop(
        jnp.int32(0),
        jnp.sum(active, dtype=jnp.int32),
        integrate_cell,
        (
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            jnp.int32(0),
        ),
    )
    normalization = 3.0 / (jnp.pi * rho**2 * (3.0 - u1))
    magnification = normalization * ((1.0 - u1) * uniform + u1 * profile)
    status = jnp.where(
        topology,
        jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
        jnp.int32(0),
    )
    result = CartesianMomentResult(
        magnification,
        jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
        jnp.int32(n_slice) * jnp.sum(active, dtype=jnp.int32),
        invalid,
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        status,
    )
    return result if return_info else result.magnification


def mag_limb_dark_cartesian_fixed(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    n_slice: int = 8,
    n_profile: int = 4,
    n_limb: int = 64,
    axis: complex | Array = 1.0 + 0.0j,
    return_info: bool = False,
) -> Array | CartesianMomentResult:
    """Integrate uniform area and the LD profile in one projection."""

    return _mag_limb_dark_cartesian_impl(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_slice=n_slice,
        n_profile=n_profile,
        n_limb=n_limb,
        axis=axis,
        return_info=return_info,
    )


def _mag_limb_dark_cartesian_gk15_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    u1: Array,
    axis: Array,
    support,
    root_mode: str = "bernstein",
    n_profile: int = 8,
    bernstein_depth: int = 19,
    bernstein_capacity: int = 8,
) -> tuple[CartesianMomentResult, CartesianMomentResult]:
    """Evaluate nested Kronrod-15/Gauss-7 LD strip moments once."""

    cells, active, topology, ghost, limb_topology = support
    axis = _normalized_axis(axis, w_center.dtype)
    nodes = jnp.asarray(GK15_X, dtype=w_center.real.dtype)
    fine_weights = jnp.asarray(GK15_W, dtype=w_center.real.dtype)
    coarse_weights = jnp.asarray(G7_W_ON_GK15, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)
    lens = binary_geometry(s, q)
    lens_radius = jnp.maximum(
        jnp.abs(lens.shifted - lens.a), jnp.abs(lens.shifted + lens.a)
    )
    source_bound = jnp.abs(w_center) + rho
    radial_offset = source_bound - lens_radius
    ordinate_bound = lens_radius + 0.5 * (
        radial_offset + jnp.sqrt(radial_offset**2 + 4.0)
    )
    ordinate_bound *= 1.0 + 32.0 * jnp.finfo(w_center.real.dtype).eps

    def integrate_cell(cell_index, state):
        fine_uniform, fine_profile, coarse_uniform, coarse_profile, invalid = state
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
        uniform, profile, node_invalid = _strip_profile_moments(
            abscissa,
            coefficients,
            w_center,
            rho,
            s=s,
            q=q,
            axis=axis,
            n_profile=n_profile,
            root_mode=root_mode,
            ordinate_bound=ordinate_bound,
            bernstein_depth=bernstein_depth,
            bernstein_capacity=bernstein_capacity,
        )
        return (
            fine_uniform + jnp.sum(fine_weights * jacobian * uniform),
            fine_profile + jnp.sum(fine_weights * jacobian * profile),
            coarse_uniform + jnp.sum(coarse_weights * jacobian * uniform),
            coarse_profile + jnp.sum(coarse_weights * jacobian * profile),
            invalid + jnp.sum(node_invalid, dtype=jnp.int32),
        )

    fine_uniform, fine_profile, coarse_uniform, coarse_profile, invalid = (
        jax.lax.fori_loop(
            jnp.int32(0),
            jnp.sum(active, dtype=jnp.int32),
            integrate_cell,
            (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
            ),
        )
    )
    normalization = 3.0 / (jnp.pi * rho**2 * (3.0 - u1))
    status = jnp.where(
        topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)
    )
    ghost_ratio = ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    n_active = jnp.sum(active, dtype=jnp.int32)

    def result(uniform, profile, order):
        return CartesianMomentResult(
            normalization * ((1.0 - u1) * uniform + u1 * profile),
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.int32(order) * n_active,
            invalid,
            ghost_ratio,
            limb_topology,
            status,
        )

    return result(fine_uniform, fine_profile, 15), result(
        coarse_uniform, coarse_profile, 7
    )


def _trace_limb_dark_cartesian_primary(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_limb: int,
):
    """Trace one source limb and build only the primary projection support."""

    safe_magnitude = jnp.maximum(
        jnp.abs(w_center), jnp.finfo(w_center.real.dtype).tiny
    )
    primary_axis = jnp.where(
        jnp.abs(w_center) > 0.0,
        1.0j * w_center / safe_magnitude,
        jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype),
    )
    scout_axis = primary_axis * jnp.exp(
        1.0j * jnp.deg2rad(jnp.asarray(45.0, dtype=w_center.real.dtype))
    )
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
    common = {
        "topology_uncertain": jnp.asarray(False),
        "minimum_ghost_residual": ghost,
        "limb_topology": limb_topology,
        "maximum_extrema": 20,
    }
    primary_support = _cartesian_support_cells_from_trace(
        image_limb,
        physical_mask,
        axis=primary_axis,
        neighbors=trace_neighbors,
        **common,
    )
    return (
        primary_axis,
        scout_axis,
        primary_support,
        image_limb,
        physical_mask,
        topology,
        ghost,
        limb_topology,
        trace_neighbors,
    )


def _complete_limb_dark_cartesian_pair(traced):
    """Build the rotated support lazily from an existing limb trace."""

    (
        primary_axis,
        scout_axis,
        primary_support,
        image_limb,
        physical_mask,
        _topology,
        ghost,
        limb_topology,
        trace_neighbors,
    ) = traced
    scout_support = _cartesian_support_cells_from_trace(
        image_limb,
        physical_mask,
        axis=scout_axis,
        topology_uncertain=jnp.asarray(False),
        minimum_ghost_residual=ghost,
        limb_topology=limb_topology,
        maximum_extrema=20,
        neighbors=trace_neighbors,
    )
    return primary_axis, scout_axis, primary_support, scout_support


def _prepare_limb_dark_cartesian_pair(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_limb: int,
):
    """Trace one source limb and build both projection supports."""

    traced = _trace_limb_dark_cartesian_primary(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
    )
    return _complete_limb_dark_cartesian_pair(traced)


def _mag_limb_dark_cartesian_pair(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    rtol: float | Array = 1.0e-3,
    primary_n_slice: int,
    scout_n_slice: int,
    certificate_fraction: float | Array,
    n_limb: int = 64,
    root_mode: str = "companion",
    n_profile: int = 8,
    scout_n_profile: int | None = None,
    bernstein_depth: int = 19,
    bernstein_capacity: int = 10,
    _prepared=None,
    _primary=None,
    _stage: int = 1,
) -> CartesianAdaptiveResult:
    """Evaluate and cross-certify one configurable Cartesian LD pair."""

    if root_mode == "bernstein-companion":
        primary_root_mode = "bernstein"
        scout_root_mode = "companion"
    elif root_mode in ("bernstein", "companion"):
        primary_root_mode = root_mode
        scout_root_mode = root_mode
    else:
        raise ValueError(f"unknown root_mode: {root_mode}")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    certificate_fraction = jnp.asarray(
        certificate_fraction, dtype=w_center.real.dtype
    )
    if scout_n_profile is None:
        scout_n_profile = n_profile
    if _prepared is None:
        prepared = _prepare_limb_dark_cartesian_pair(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=n_limb,
        )
    else:
        prepared = _prepared
    primary_axis, scout_axis, primary_support, scout_support = prepared
    if _primary is None:
        primary = _mag_limb_dark_cartesian_impl(
            w_center,
            rho,
            s=s,
            q=q,
            u1=u1,
            n_slice=primary_n_slice,
            n_profile=n_profile,
            n_limb=n_limb,
            axis=primary_axis,
            support=primary_support,
            root_mode=primary_root_mode,
            bernstein_depth=bernstein_depth,
            bernstein_capacity=bernstein_capacity,
            return_info=True,
        )
    else:
        primary = _primary
    scout = _mag_limb_dark_cartesian_impl(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_slice=scout_n_slice,
        n_profile=scout_n_profile,
        n_limb=n_limb,
        axis=scout_axis,
        support=scout_support,
        root_mode=scout_root_mode,
        bernstein_depth=bernstein_depth,
        bernstein_capacity=bernstein_capacity,
        return_info=True,
    )
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(primary.magnification), jnp.abs(scout.magnification)),
        1.0,
    )
    difference = jnp.abs(primary.magnification - scout.magnification)
    certified = (
        (primary.invalid_root_count == 0)
        & (scout.invalid_root_count == 0)
        & (primary.status == 0)
        & (scout.status == 0)
        & jnp.isfinite(primary.magnification)
        & jnp.isfinite(scout.magnification)
        & (difference <= certificate_fraction * rtol * scale)
    )
    return CartesianAdaptiveResult(
        primary.magnification,
        jnp.maximum(difference, 0.75 * rtol * scale),
        primary.n_slices + scout.n_slices,
        jnp.int32(_stage),
        jnp.where(
            certified,
            jnp.int32(0),
            jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
        ),
    )


def mag_limb_dark_cartesian_adaptive(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    rtol: float | Array = 1.0e-3,
    n_limb: int = 64,
    external_magnification: Array | None = None,
    return_info: bool = False,
) -> Array | CartesianAdaptiveResult:
    """Certify a fast LD profile moment with an independent rotated axis."""

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    scout_n_limb = min(n_limb, 64)

    def cross_schedule(prepared, traced, primary=None):
        stage1 = _mag_limb_dark_cartesian_pair(
            w_center,
            rho,
            s=s,
            q=q,
            u1=u1,
            rtol=rtol,
            primary_n_slice=7,
            scout_n_slice=6,
            certificate_fraction=0.10,
            n_limb=n_limb,
            root_mode="bernstein",
            bernstein_capacity=8,
            _prepared=prepared,
            _primary=primary,
        )
        def refine(_):
            if n_limb == scout_n_limb:
                refined_traced = traced
                refined_prepared = prepared
            else:
                refined_traced = _trace_limb_dark_cartesian_primary(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    n_limb=n_limb,
                )
                refined_prepared = _complete_limb_dark_cartesian_pair(
                    refined_traced
                )
            stage2 = _mag_limb_dark_cartesian_pair(
                w_center,
                rho,
                s=s,
                q=q,
                u1=u1,
                rtol=rtol,
                primary_n_slice=12,
                scout_n_slice=10,
                certificate_fraction=0.50,
                n_limb=n_limb,
                root_mode="companion",
                _prepared=refined_prepared,
                _stage=2,
            )

            def use_polar(_):
                (
                    _,
                    _,
                    _,
                    image_limb,
                    physical_mask,
                    topology,
                    ghost,
                    limb_topology,
                    trace_neighbors,
                ) = refined_traced
                angular_support = _angular_support_cells_from_trace(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    physical_limb=image_limb,
                    physical_mask=physical_mask,
                    topology_uncertain=topology,
                    minimum_ghost_residual=ghost,
                    limb_topology=limb_topology,
                    neighbors=trace_neighbors,
                )
                polar = mag_limb_dark_angular_moment_compact(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    u1=u1,
                    rtol=rtol,
                    _support=angular_support,
                    return_info=True,
                )
                return CartesianAdaptiveResult(
                    polar.magnification,
                    polar.estimated_error,
                    stage2.n_slices + polar.n_theta,
                    jnp.int32(3),
                    polar.status,
                )

            return jax.lax.cond(
                stage2.status != 0,
                use_polar,
                lambda _: stage2,
                operand=None,
            )

        return jax.lax.cond(
            stage1.status != 0,
            refine,
            lambda _: stage1,
            operand=None,
        )

    if external_magnification is None:
        traced = _trace_limb_dark_cartesian_primary(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=scout_n_limb,
        )
        prepared = _complete_limb_dark_cartesian_pair(traced)
        result = cross_schedule(prepared, traced)
    else:
        traced = _trace_limb_dark_cartesian_primary(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=scout_n_limb,
        )
        primary_axis, _, primary_support, *_ = traced
        primary = _mag_limb_dark_cartesian_impl(
            w_center,
            rho,
            s=s,
            q=q,
            u1=u1,
            n_slice=7,
            n_profile=8,
            n_limb=n_limb,
            axis=primary_axis,
            support=primary_support,
            root_mode="bernstein",
            bernstein_capacity=8,
            return_info=True,
        )
        external_magnification = jnp.asarray(
            external_magnification, dtype=w_center.real.dtype
        )
        external_scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(primary.magnification), jnp.abs(external_magnification)
            ),
            1.0,
        )
        external_difference = jnp.abs(
            primary.magnification - external_magnification
        )
        external_ok = (
            (primary.invalid_root_count == 0)
            & (primary.status == 0)
            & jnp.isfinite(primary.magnification)
            & jnp.isfinite(external_magnification)
            & (external_difference <= 0.05 * jnp.asarray(rtol) * external_scale)
        )
        result = jax.lax.cond(
            external_ok,
            lambda _: CartesianAdaptiveResult(
                primary.magnification,
                jnp.maximum(
                    external_difference,
                    0.5 * jnp.asarray(rtol) * external_scale,
                ),
                primary.n_slices,
                jnp.int32(1),
                jnp.int32(0),
            ),
            lambda _: cross_schedule(
                _complete_limb_dark_cartesian_pair(traced),
                traced,
                primary,
            ),
            operand=None,
        )
    return result if return_info else result.magnification


__all__ = [
    "mag_limb_dark_cartesian_adaptive",
    "mag_limb_dark_cartesian_fixed",
]
