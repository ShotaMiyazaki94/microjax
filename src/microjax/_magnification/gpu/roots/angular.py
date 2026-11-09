"""Resolution-free angular integration for binary and triple inverse ray shooting.

The production path solves the associated degree-six self-inversive polynomial
and validates every unit-circle root.  A conservative interval-classification
implementation is retained as an independent oracle and difficult-ring
fallback building block.  Neither path accepts a dense angular grid length.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from microjax.poly_solver import (
    poly_roots_self_inversive_fixed,
    poly_roots_self_inversive_robust_fixed,
)

from .level_set import binary_level_set_fourier, triple_level_set_fourier

Array = jnp.ndarray

ANGULAR_OK = 0
ANGULAR_CAPACITY = 1
ANGULAR_DEGENERATE = 2
ANGULAR_ROOT_FAILURE = 4

_INITIAL_CELLS = 16
_CELL_CAPACITY = 64
_MAX_DEPTH = 32
_BINARY_INSIDE_INTERVAL_CAPACITY = 4
_TRIPLE_INSIDE_INTERVAL_CAPACITY = 5


class AngularMeasure(NamedTuple):
    """Angular inside measure, its absolute bound, and a status bit mask."""

    measure: Array
    error: Array
    status: Array


class AngularIntervals(NamedTuple):
    """Fixed-shape inside intervals and boundary-root diagnostics."""

    intervals: Array
    n_intervals: Array
    error: Array
    status: Array


def evaluate_fourier(coefficients: Array, theta: Array) -> Array:
    """Evaluate a real trigonometric polynomial from positive modes."""

    modes = jnp.arange(1, coefficients.shape[0], dtype=theta.dtype)
    oscillatory = coefficients[1:] * jnp.exp(1j * theta[..., None] * modes)
    return jnp.real(coefficients[0]) + 2.0 * jnp.sum(jnp.real(oscillatory), axis=-1)


def evaluate_fourier_derivative(coefficients: Array, theta: Array) -> Array:
    """Evaluate the angular derivative of a real Fourier polynomial."""

    modes = jnp.arange(1, coefficients.shape[0], dtype=theta.dtype)
    oscillatory = coefficients[1:] * jnp.exp(1j * theta[..., None] * modes)
    return 2.0 * jnp.sum(jnp.real(1j * modes * oscillatory), axis=-1)


def evaluate_fourier_second_derivative(coefficients: Array, theta: Array) -> Array:
    """Evaluate the second angular derivative of a real Fourier polynomial."""

    modes = jnp.arange(1, coefficients.shape[0], dtype=theta.dtype)
    oscillatory = coefficients[1:] * jnp.exp(1j * theta[..., None] * modes)
    return -2.0 * jnp.sum(jnp.real(modes**2 * oscillatory), axis=-1)


def _conditioned_root_angle_error(
    residual: Array,
    padding: Array,
    derivative: Array,
    second_derivative: Array,
    eps: Array,
) -> Array:
    """Estimate root-angle error in both simple and tangent regimes.

    Use the linear condition estimate for simple roots and the positive
    quadratic solution near tangencies, where the linear estimate diverges.
    Vanishing curvature falls back to the conservative linear value.
    """

    uncertainty = residual + padding
    slope = jnp.abs(derivative)
    curvature = jnp.abs(second_derivative)
    linear = uncertainty / jnp.maximum(slope, jnp.sqrt(eps))
    discriminant = jnp.sqrt(slope**2 + 2.0 * curvature * uncertainty)
    quadratic = 2.0 * uncertainty / jnp.maximum(slope + discriminant, jnp.finfo(uncertainty.dtype).tiny)
    curvature_resolved = curvature > jnp.sqrt(eps)
    return jnp.where(curvature_resolved, jnp.minimum(quadratic, linear), linear)


def _reciprocal_pair_rescue(
    roots: Array,
    finite_roots: Array,
    unit_error: Array,
    residual_ok: Array,
    unit_candidate: Array,
    unit_tolerance: Array,
) -> Array:
    """Promote a straddling near-unit reciprocal pair atomically.

    A self-inversive polynomial has off-circle roots in pairs
    ``z_j = 1 / conj(z_i)``.  Near a double contact, roundoff in the root solve
    can put only one member inside the hard unit-circle threshold.  Accepting
    that member alone creates a spurious odd root count.  Rescue both members
    only when one already passed the ordinary threshold, both remain in a
    narrow near-unit guard band, both polished angles satisfy the Fourier
    level set, and the reciprocal relation is numerically resolved.
    """

    reciprocal_error = jnp.abs(
        roots[:, None] * jnp.conjugate(roots[None, :]) - 1.0
    )
    indices = jnp.arange(roots.shape[0])
    distinct = indices[:, None] != indices[None, :]
    finite_pair = finite_roots[:, None] & finite_roots[None, :]
    match_error = jnp.where(
        distinct & finite_pair,
        reciprocal_error,
        jnp.inf,
    )
    nearest_partner = jnp.argmin(match_error, axis=1)
    nearest_pair = indices[None, :] == nearest_partner[:, None]
    mutual_nearest_pair = nearest_pair & nearest_pair.T
    rescue_unit_tolerance = jnp.minimum(2.0 * unit_tolerance, 1e-3)
    # The same coefficient perturbation that moves a near-double root by
    # O(sqrt(eps)) can make the two independently converged roots imperfect
    # reciprocals by a comparable fraction of the accepted unit-circle band.
    # Tie the pair check to that band instead of a second, tighter threshold.
    # Both projected angles must still pass the original Fourier residual
    # below, so this does not admit arbitrary nearby off-circle roots.
    reciprocal_tolerance = jnp.minimum(0.25 * unit_tolerance, 1e-4)
    pair_ok = (
        mutual_nearest_pair
        & finite_pair
        & (unit_error[:, None] <= rescue_unit_tolerance)
        & (unit_error[None, :] <= rescue_unit_tolerance)
        & residual_ok[:, None]
        & residual_ok[None, :]
        & (unit_candidate[:, None] | unit_candidate[None, :])
        & (reciprocal_error <= reciprocal_tolerance)
    )
    return jnp.any(pair_ok, axis=1)


def _angular_intervals_fourier_roots(
    fourier,
    r: float,
    theta_min: float,
    theta_max: float,
    interval_capacity: int,
    robust_roots: bool = False,
    propagate_coefficient_padding: bool = True,
) -> AngularIntervals:
    """Classify a ring, skipping root solve when its sign is uniform."""

    c = fourier.coefficients
    oscillatory_bound = 2.0 * jnp.sum(jnp.abs(c[1:]))
    constant = jnp.real(c[0])
    entirely_inside = constant + oscillatory_bound + fourier.padding < 0.0
    entirely_outside = constant - oscillatory_bound - fourier.padding > 0.0
    classified = entirely_inside | entirely_outside | (theta_max <= theta_min)

    def constant_result(_):
        span_active = entirely_inside & (theta_max > theta_min)
        intervals = jnp.zeros((interval_capacity, 2), dtype=jnp.asarray(r).dtype)
        intervals = intervals.at[0].set(
            jnp.where(
                span_active,
                jnp.asarray([theta_min, theta_max]),
                jnp.asarray([0.0, 0.0]),
            )
        )
        return AngularIntervals(
            intervals,
            jnp.where(span_active, jnp.int32(1), jnp.int32(0)),
            jnp.asarray(0.0, dtype=jnp.asarray(r).dtype),
            jnp.int32(ANGULAR_OK),
        )

    def solve_roots(_):
        return _angular_intervals_fourier_roots_impl(
            fourier,
            r,
            theta_min,
            theta_max,
            interval_capacity,
            robust_roots,
            propagate_coefficient_padding,
        )

    return jax.lax.cond(classified, constant_result, solve_roots, operand=None)


def _angular_intervals_fourier_roots_impl(
    fourier,
    r: float,
    theta_min: float,
    theta_max: float,
    interval_capacity: int,
    robust_roots: bool = False,
    propagate_coefficient_padding: bool = True,
) -> AngularIntervals:
    """Classify intervals from a validated self-inversive Fourier polynomial."""

    c = fourier.coefficients
    polynomial = jnp.concatenate((c[:0:-1], c[:1], jnp.conjugate(c[1:])))
    root_solver = poly_roots_self_inversive_robust_fixed if robust_roots else poly_roots_self_inversive_fixed
    roots = root_solver(polynomial[None, :])[0]
    finite_roots = jnp.isfinite(roots.real) & jnp.isfinite(roots.imag)
    polynomial_residual = jnp.abs(jnp.polyval(polynomial, roots))
    polynomial_scale = jnp.polyval(jnp.abs(polynomial), jnp.abs(roots))
    relative_polynomial_residual = polynomial_residual / jnp.maximum(
        polynomial_scale, jnp.finfo(jnp.asarray(r).dtype).tiny
    )
    unit_error = jnp.abs(jnp.abs(roots) - 1.0)
    eps = jnp.finfo(jnp.asarray(r).dtype).eps
    # Near a radial topology extremum, a true double root on the unit circle is
    # ill-conditioned and the polynomial solver can split the pair slightly in
    # modulus while preserving its angle to high accuracy.  Projection is safe
    # only when the polished angle also satisfies the original Fourier level
    # set below, which rejects ordinary reciprocal off-circle roots.
    # Degree-eight triple roots need one extra factor of two at radial
    # tangencies. Binary keeps its established threshold unchanged.
    unit_tolerance_factor = 4096.0 if c.shape[0] == 5 else 2048.0
    unit_tolerance = jnp.minimum(unit_tolerance_factor * jnp.sqrt(eps), 1e-3)

    theta = jnp.mod(jnp.angle(roots), 2.0 * jnp.pi)

    def polish(_, angles):
        value = evaluate_fourier(c, angles)
        derivative = evaluate_fourier_derivative(c, angles)
        safe = jnp.abs(derivative) > 16.0 * eps
        step = jnp.where(safe, value / derivative, 0.0)
        step = jnp.clip(step, -0.25, 0.25)
        candidates = jnp.stack(
            (
                angles,
                angles - step,
                angles - 0.5 * step,
                angles - 0.25 * step,
            ),
            axis=0,
        )
        candidate_residual = jnp.abs(evaluate_fourier(c, candidates))
        best = jnp.argmin(candidate_residual, axis=0)
        return jnp.take_along_axis(candidates, best[None, :], axis=0)[0]

    theta = jax.lax.fori_loop(0, 6, polish, theta)
    residual = jnp.abs(evaluate_fourier(c, theta))
    # Near a tangency, two roots representing the same even-multiplicity
    # contact can reach the floating-point floor on opposite sides of one
    # rounding ulp.  A 4096*eps allowance admitted only one member of such a
    # pair on GPU for a valid rho=1e-4 case, making ``n_valid`` spuriously odd.
    # This remains an O(eps) root-certification allowance; the accepted root
    # residual is still propagated into the independent quadrature error.
    root_refinement_roundoff = 4096.0 * eps
    fourier_evaluation_roundoff = 64.0 * eps
    residual_tolerance = fourier.padding + root_refinement_roundoff + fourier_evaluation_roundoff
    residual_ok = residual <= residual_tolerance
    unit_candidate = finite_roots & (unit_error <= unit_tolerance)
    pair_rescued = _reciprocal_pair_rescue(
        roots,
        finite_roots,
        unit_error,
        residual_ok,
        unit_candidate,
        unit_tolerance,
    )
    validated_candidate = unit_candidate | pair_rescued
    valid = validated_candidate & residual_ok
    n_valid = jnp.sum(valid, dtype=jnp.int32)
    check_angles = 2.0 * jnp.pi * jnp.arange(16, dtype=jnp.asarray(r).dtype) / 16.0
    check_values = evaluate_fourier(c, check_angles)
    sign_changes = jnp.sum(
        (check_values <= 0.0) != (jnp.roll(check_values, -1) <= 0.0),
        dtype=jnp.int32,
    )
    leading_ok = jnp.abs(polynomial[0]) > 128.0 * eps
    polynomial_tolerance = jnp.maximum(1024.0 * eps, 1e-10)
    roots_ok = (
        ~fourier.degenerate
        & leading_ok
        & jnp.all(finite_roots)
        & jnp.all(
            jnp.where(
                validated_candidate,
                relative_polynomial_residual <= polynomial_tolerance,
                True,
            )
        )
        & jnp.all(jnp.isfinite(polynomial.real))
        & jnp.all(jnp.isfinite(polynomial.imag))
        & ((n_valid % 2) == 0)
        & (n_valid >= sign_changes)
    )
    two_pi = 2.0 * jnp.pi
    turns = jnp.ceil((theta_min - theta) / two_pi)
    theta_in_interval = theta + turns * two_pi
    in_interval = valid & (theta_in_interval >= theta_min) & (theta_in_interval <= theta_max)
    roots_sorted = jnp.sort(jnp.where(in_interval, theta_in_interval, theta_max))
    boundaries = jnp.concatenate((jnp.atleast_1d(theta_min), roots_sorted, jnp.atleast_1d(theta_max)))
    segment_lo = boundaries[:-1]
    segment_hi = boundaries[1:]
    segment_width = jnp.maximum(segment_hi - segment_lo, 0.0)
    segment_mid = 0.5 * (segment_lo + segment_hi)
    segment_inside = evaluate_fourier(c, segment_mid) <= 0.0
    segment_active = segment_inside & (segment_width > 0.0)
    # A projected near-tangent root can split one physical inside interval into
    # adjacent segments without changing the sign of H.  Treating every segment
    # as a separate interval both wastes fixed capacity and, more importantly,
    # attributes root uncertainty to contacts that do not move the boundary of
    # the integrated set.  Pack maximal contiguous inside runs instead.
    previous_active = jnp.concatenate((jnp.asarray([False]), segment_active[:-1]))
    next_active = jnp.concatenate((segment_active[1:], jnp.asarray([False])))
    interval_start = segment_active & ~previous_active
    interval_end = segment_active & ~next_active
    n_intervals_raw = jnp.sum(interval_start, dtype=jnp.int32)
    start_indices = jnp.nonzero(
        interval_start,
        size=interval_capacity,
        fill_value=segment_width.size - 1,
    )[0]
    end_indices = jnp.nonzero(
        interval_end,
        size=interval_capacity,
        fill_value=segment_width.size - 1,
    )[0]
    intervals = jnp.stack((segment_lo[start_indices], segment_hi[end_indices]), axis=1)
    n_intervals = jnp.minimum(n_intervals_raw, jnp.int32(interval_capacity))
    intervals = jnp.where(
        (jnp.arange(interval_capacity) < n_intervals)[:, None],
        intervals,
        0.0,
    )

    derivative = evaluate_fourier_derivative(c, theta)
    if propagate_coefficient_padding:
        root_error = (residual + fourier.padding) / jnp.maximum(jnp.abs(derivative), jnp.sqrt(eps))
    else:
        # ``fourier.padding`` is one correlated coefficient-construction
        # uncertainty for the complete ring family. Adding it independently at
        # every radial quadrature node makes the final area estimate grow with
        # the number of nodes and grossly over-counts x64 roundoff for tiny
        # sources. It remains part of root validation above. Only the measured
        # root-solve residual is propagated node by node; the binary boundary
        # integrator applies one tangent-aware global roundoff floor after area
        # normalization.
        second_derivative = evaluate_fourier_second_derivative(c, theta)
        root_error = _conditioned_root_angle_error(
            residual,
            jnp.asarray(0.0, dtype=residual.dtype),
            derivative,
            second_derivative,
            eps,
        )
    root_order = jnp.argsort(jnp.where(in_interval, theta_in_interval, theta_max))
    sorted_root_error = jnp.where(in_interval[root_order], root_error[root_order], 0.0)
    boundary_error = jnp.concatenate(
        (
            jnp.zeros(1, dtype=root_error.dtype),
            sorted_root_error,
            jnp.zeros(1, dtype=root_error.dtype),
        )
    )
    # Only roots forming the exterior endpoints of accepted inside runs can
    # perturb the angular measure.  Same-sign tangent contacts inside a run and
    # projected near-roots wholly in an outside region contribute no first-order
    # measure uncertainty.
    measure_error = jnp.sum(jnp.where(interval_start, boundary_error[:-1], 0.0)) + jnp.sum(
        jnp.where(interval_end, boundary_error[1:], 0.0)
    )
    error = jnp.minimum(
        measure_error,
        jnp.maximum(theta_max - theta_min, 0.0),
    )
    roots_ok = (roots_ok & jnp.isfinite(error)) | (theta_max <= theta_min)
    status = jnp.where(
        roots_ok,
        jnp.int32(ANGULAR_OK),
        jnp.int32(ANGULAR_ROOT_FAILURE),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            n_intervals_raw <= interval_capacity,
            jnp.int32(ANGULAR_OK),
            jnp.int32(ANGULAR_CAPACITY),
        ),
    )
    return AngularIntervals(intervals, n_intervals, error, status)


def angular_intervals_binary_roots(
    r: float,
    theta_min: float,
    theta_max: float,
    w_center_shifted: complex,
    rho: float,
    shifted: float,
    cell_tolerance: float,
    *,
    a: float,
    e1: float,
    robust_roots: bool = True,
    chart_center: complex = 0.0 + 0.0j,
    propagate_coefficient_padding: bool = True,
) -> AngularIntervals:
    """Return inside intervals from the degree-six boundary polynomial roots.

    The self-inversive polynomial is obtained directly from the degree-three
    Fourier coefficients. All roots are solved in a fixed shape, polished
    against the original Fourier level set, and validated before they
    partition the requested angular interval. ``cell_tolerance`` remains in
    the common interface but does not select a sampling resolution.
    """

    del cell_tolerance
    fourier = binary_level_set_fourier(
        r,
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
        chart_center=chart_center,
    )
    return _angular_intervals_fourier_roots(
        fourier,
        r,
        theta_min,
        theta_max,
        _BINARY_INSIDE_INTERVAL_CAPACITY,
        robust_roots,
        propagate_coefficient_padding,
    )


def angular_intervals_triple_roots(
    r: float,
    theta_min: float,
    theta_max: float,
    w_center_shifted: complex,
    rho: float,
    shifted: complex,
    cell_tolerance: float,
    *,
    a: float,
    e1: float,
    e2: float,
    r3_complex: complex,
    chart_center: complex = 0.0 + 0.0j,
) -> AngularIntervals:
    """Return inside intervals from the degree-eight triple boundary roots."""

    del cell_tolerance
    fourier = triple_level_set_fourier(
        r,
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
        e2=e2,
        r3_complex=r3_complex,
        chart_center=chart_center,
    )
    return _angular_intervals_fourier_roots(
        fourier,
        r,
        theta_min,
        theta_max,
        _TRIPLE_INSIDE_INTERVAL_CAPACITY,
    )


def angular_measure_binary_roots(
    r: float,
    theta_min: float,
    theta_max: float,
    w_center_shifted: complex,
    rho: float,
    shifted: float,
    cell_tolerance: float,
    *,
    a: float,
    e1: float,
    robust_roots: bool = True,
    chart_center: complex = 0.0 + 0.0j,
    propagate_coefficient_padding: bool = True,
) -> AngularMeasure:
    """Integrate angular width from validated boundary-root intervals."""

    result = angular_intervals_binary_roots(
        r,
        theta_min,
        theta_max,
        w_center_shifted,
        rho,
        shifted,
        cell_tolerance,
        a=a,
        e1=e1,
        robust_roots=robust_roots,
        chart_center=chart_center,
        propagate_coefficient_padding=propagate_coefficient_padding,
    )
    active = jnp.arange(_BINARY_INSIDE_INTERVAL_CAPACITY) < result.n_intervals
    widths = result.intervals[:, 1] - result.intervals[:, 0]
    measure = jnp.sum(jnp.where(active, widths, 0.0))
    return AngularMeasure(measure, result.error, result.status)


def angular_measure_triple_roots(
    r: float,
    theta_min: float,
    theta_max: float,
    w_center_shifted: complex,
    rho: float,
    shifted: complex,
    cell_tolerance: float,
    *,
    a: float,
    e1: float,
    e2: float,
    r3_complex: complex,
    chart_center: complex = 0.0 + 0.0j,
) -> AngularMeasure:
    """Integrate triple-lens angular width from validated boundary roots."""

    result = angular_intervals_triple_roots(
        r,
        theta_min,
        theta_max,
        w_center_shifted,
        rho,
        shifted,
        cell_tolerance,
        a=a,
        e1=e1,
        e2=e2,
        r3_complex=r3_complex,
        chart_center=chart_center,
    )
    active = jnp.arange(_TRIPLE_INSIDE_INTERVAL_CAPACITY) < result.n_intervals
    widths = result.intervals[:, 1] - result.intervals[:, 0]
    measure = jnp.sum(jnp.where(active, widths, 0.0))
    return AngularMeasure(measure, result.error, result.status)
