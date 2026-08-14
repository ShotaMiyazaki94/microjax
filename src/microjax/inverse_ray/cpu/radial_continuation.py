"""Radial-first polar ICRS for uniform-source CPU execution.

The production route traces the source limb once, turns its radial contacts
into conservative cells, prunes only cells proved outside by tensor-product
geometric contacts, and integrates each cell with fixed GK15 panels. At each
fixed radius it solves the cancellation-resistant degree-six angular level-set
polynomial independently.  Older continuation and atlas kernels remain in
this module as explicit research/reference routes; the public one-shot path
uses ``use_simple_support=True``, ignores the embedded G7 acceptance heuristic,
and never retries after inspecting a result.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from microjax.poly_solver import poly_roots_self_inversive_robust_fixed

from ..geometry.lens import binary_geometry
from ..geometry.topology import RADIAL_OK, RADIAL_TOLERANCE, define_radial_topology
from ..roots.angular import (
    ANGULAR_OK,
    ANGULAR_ROOT_FAILURE,
    _reciprocal_pair_rescue,
    evaluate_fourier,
    evaluate_fourier_derivative,
)
from ..roots.level_set import binary_level_set, binary_level_set_fourier
from .quadrature import (
    G7_W_ON_GL11,
    G7_W_ON_GK15,
    GK15_W,
    GK15_X,
    GL11_W,
    GL11_X,
)
from .polar_atlas import (
    POLAR_ATLAS_TOPOLOGY,
    PolarBranchAtlas,
    build_polar_branch_atlas,
    polar_atlas_angular_measure,
    polar_atlas_turning_radius_override,
    refine_polar_atlas_tangencies,
)
from .roots import companion_roots, real_companion_roots
from .simple_polar_support import build_simple_polar_topology
from .support import trace_binary_source_limb
from .refined_limb import trace_binary_source_limb_two_stage

Array = jnp.ndarray
_ANGULAR_CONTINUATION_FAILURE = 8


class RadialContinuationResult(NamedTuple):
    """Uniform-source value and diagnostics for the radial CPU prototype.

    The research/adaptive entry point includes an embedded-quadrature check.
    The production one-shot caller disables that check and uses this status
    only for structural diagnostics.
    """

    magnification: Array
    estimated_error: Array
    n_cells: Array
    n_full_root_solves: Array
    n_continuation_nodes: Array
    status: Array


class _ThetaRoots(NamedTuple):
    angles: Array
    active: Array
    radius: Array
    polynomial: Array
    polynomial_roots: Array
    status: Array


def _fourier(w_center: Array, rho: Array, radius: Array, *, s: Array, q: Array):
    lens = binary_geometry(s, q)
    return binary_level_set_fourier(
        radius,
        w_center - lens.shifted,
        rho,
        lens.shifted,
        a=lens.a,
        e1=lens.e1,
    )


def _initial_theta_roots(
    w_center: Array,
    rho: Array,
    radius: Array,
    *,
    s: Array,
    q: Array,
) -> _ThetaRoots:
    """Solve and validate all physical angular roots on one ring."""

    fourier = _fourier(w_center, rho, radius, s=s, q=q)
    return _initial_theta_roots_from_fourier(
        fourier.coefficients,
        fourier.padding,
        fourier.degenerate,
        radius,
    )


def _initial_theta_roots_from_fourier(
    coefficients: Array,
    padding: Array,
    degenerate: Array,
    radius: Array,
    *,
    use_companion: bool = False,
) -> _ThetaRoots:
    """Solve roots from Fourier data already constructed by the integrator."""

    polynomial = jnp.concatenate(
        (coefficients[:0:-1], coefficients[:1], jnp.conjugate(coefficients[1:]))
    )
    roots = (
        companion_roots(polynomial)
        if use_companion
        else poly_roots_self_inversive_robust_fixed(polynomial[None, :])[0]
    )
    finite = jnp.isfinite(roots.real) & jnp.isfinite(roots.imag)
    eps = jnp.finfo(radius.dtype).eps
    unit_error = jnp.abs(jnp.abs(roots) - 1.0)
    unit_tolerance = jnp.minimum(2048.0 * jnp.sqrt(eps), 1.0e-3)
    angles = jnp.angle(roots)

    def polish(_, current):
        value = evaluate_fourier(coefficients, current)
        derivative = evaluate_fourier_derivative(coefficients, current)
        safe = jnp.abs(derivative) > 16.0 * eps
        step = jnp.where(safe, value / derivative, 0.0)
        return current - jnp.clip(step, -0.25, 0.25)

    # The real companion solve is already close to machine precision.  Two
    # steps in the original Fourier equation remove the tan-half-angle
    # conditioning error; the residual check below remains the certificate.
    angles = jax.lax.fori_loop(0, 2, polish, angles)
    residual = jnp.abs(evaluate_fourier(coefficients, angles))
    residual_tolerance = padding + 8192.0 * eps
    residual_ok = residual <= residual_tolerance
    unit_candidate = finite & (unit_error <= unit_tolerance)
    rescued = _reciprocal_pair_rescue(
        roots,
        finite,
        unit_error,
        residual_ok,
        unit_candidate,
        unit_tolerance,
    )
    valid = (unit_candidate | rescued) & residual_ok
    n_valid = jnp.sum(valid, dtype=jnp.int32)
    sorted_angles = jnp.sort(
        jnp.where(valid, jnp.mod(angles, 2.0 * jnp.pi), 2.0 * jnp.pi)
    )
    active = jnp.arange(roots.size, dtype=jnp.int32) < n_valid
    status_ok = (
        ~degenerate
        & jnp.all(finite)
        & ((n_valid % 2) == 0)
        & jnp.all(jnp.isfinite(polynomial.real))
        & jnp.all(jnp.isfinite(polynomial.imag))
    )
    return _ThetaRoots(
        sorted_angles,
        active,
        radius,
        polynomial,
        roots,
        jnp.where(status_ok, jnp.int32(ANGULAR_OK), jnp.int32(ANGULAR_ROOT_FAILURE)),
    )


def _tangent_polynomial(coefficients: Array) -> tuple[Array, Array]:
    """Convert a degree-three Fourier level set to one real sextic.

    With ``t = tan(phi / 2)``, multiplying the real trigonometric polynomial
    by ``(1 + t**2)**3`` produces a degree-six real polynomial.  A discrete
    chart rotation places the tangent pole where the level set has the largest
    sampled magnitude, keeping the leading coefficient away from zero without
    making continuation or a sampled sign grid part of correctness.
    """

    dtype = coefficients.real.dtype
    degree = coefficients.shape[0] - 1
    rotations = 0.25 * jnp.pi * jnp.arange(8, dtype=dtype)
    modes = jnp.arange(degree + 1, dtype=dtype)
    rotated = coefficients[None, :] * jnp.exp(
        1.0j * rotations[:, None] * modes[None, :]
    )
    pole_angles = rotations + jnp.pi
    pole_values = evaluate_fourier(coefficients, pole_angles)
    selected = jax.lax.stop_gradient(jnp.argmax(jnp.abs(pole_values)))
    selected_coefficients = rotated[selected]

    constant = jnp.real(selected_coefficients[0])
    cosine = 2.0 * jnp.real(selected_coefficients[1:])
    sine = -2.0 * jnp.imag(selected_coefficients[1:])
    a1, a2, a3 = cosine
    b1, b2, b3 = sine
    ascending = jnp.stack(
        (
            constant + a1 + a2 + a3,
            2.0 * b1 + 4.0 * b2 + 6.0 * b3,
            3.0 * constant + a1 - 5.0 * a2 - 15.0 * a3,
            4.0 * b1 - 20.0 * b3,
            3.0 * constant - a1 - 5.0 * a2 + 15.0 * a3,
            2.0 * b1 - 4.0 * b2 + 6.0 * b3,
            constant - a1 + a2 - a3,
        )
    )
    return ascending[::-1], rotations[selected]


def _initial_theta_roots_tangent_from_fourier(
    coefficients: Array,
    padding: Array,
    degenerate: Array,
    radius: Array,
) -> _ThetaRoots:
    """Solve angular crossings through a rotated real tangent sextic."""

    tangent_polynomial, rotation = _tangent_polynomial(coefficients)
    polynomial_scale = jnp.maximum(
        jnp.max(jnp.abs(tangent_polynomial)),
        jnp.finfo(radius.dtype).tiny,
    )
    normalized_polynomial = tangent_polynomial / polynomial_scale
    tangent_roots = real_companion_roots(normalized_polynomial)
    finite = jnp.isfinite(tangent_roots.real) & jnp.isfinite(tangent_roots.imag)
    eps = jnp.finfo(radius.dtype).eps
    real_scale = jnp.maximum(jnp.abs(tangent_roots.real), 1.0)
    real_tolerance = jnp.minimum(2048.0 * jnp.sqrt(eps), 1.0e-3)
    nearly_real = jnp.abs(tangent_roots.imag) <= real_tolerance * real_scale
    angles = 2.0 * jnp.arctan(tangent_roots.real) + rotation

    def polish(_, current):
        value = evaluate_fourier(coefficients, current)
        derivative = evaluate_fourier_derivative(coefficients, current)
        safe = jnp.abs(derivative) > 16.0 * eps
        step = jnp.where(safe, value / derivative, 0.0)
        return current - jnp.clip(step, -0.25, 0.25)

    angles = jax.lax.fori_loop(0, 6, polish, angles)
    residual = jnp.abs(evaluate_fourier(coefficients, angles))
    residual_tolerance = padding + 8192.0 * eps
    valid = finite & nearly_real & (residual <= residual_tolerance)
    n_valid = jnp.sum(valid, dtype=jnp.int32)
    sorted_angles = jnp.sort(
        jnp.where(valid, jnp.mod(angles, 2.0 * jnp.pi), 2.0 * jnp.pi)
    )
    active = jnp.arange(tangent_roots.size, dtype=jnp.int32) < n_valid
    leading_ok = jnp.abs(normalized_polynomial[0]) > 128.0 * eps
    status_ok = (
        ~degenerate
        & leading_ok
        & jnp.all(finite)
        & ((n_valid % 2) == 0)
        & jnp.all(jnp.isfinite(normalized_polynomial))
    )
    self_inversive = jnp.concatenate(
        (
            coefficients[:0:-1],
            coefficients[:1],
            jnp.conjugate(coefficients[1:]),
        )
    )
    unit_roots = jnp.exp(1.0j * angles)
    return _ThetaRoots(
        sorted_angles,
        active,
        radius,
        self_inversive,
        unit_roots,
        jnp.where(
            status_ok,
            jnp.int32(ANGULAR_OK),
            jnp.int32(ANGULAR_ROOT_FAILURE),
        ),
    )


def _continue_theta_roots(
    previous: _ThetaRoots,
    coefficients: Array,
    padding: Array,
    degenerate: Array,
    radius: Array,
) -> tuple[_ThetaRoots, Array]:
    """Continue a fixed root set to an adjacent radial quadrature node."""

    eps = jnp.finfo(radius.dtype).eps

    polynomial = jnp.concatenate(
        (coefficients[:0:-1], coefficients[:1], jnp.conjugate(coefficients[1:]))
    )
    previous_polynomial = previous.polynomial
    derivative_coefficients = polynomial[:-1] * jnp.arange(
        polynomial.size - 1, 0, -1, dtype=radius.dtype
    )
    previous_derivative_coefficients = previous_polynomial[:-1] * jnp.arange(
        previous_polynomial.size - 1, 0, -1, dtype=radius.dtype
    )
    coefficient_step = polynomial - previous_polynomial
    predictor_denominator = jnp.polyval(
        previous_derivative_coefficients, previous.polynomial_roots
    )
    predictor_numerator = jnp.polyval(coefficient_step, previous.polynomial_roots)
    predictor_safe = jnp.abs(predictor_denominator) > 64.0 * eps
    predicted_roots = jnp.where(
        predictor_safe,
        previous.polynomial_roots - predictor_numerator / predictor_denominator,
        previous.polynomial_roots,
    )
    off_diagonal = ~jnp.eye(polynomial.size - 1, dtype=jnp.bool_)

    def ea_step(_, current):
        value = jnp.polyval(polynomial, current)
        derivative = jnp.polyval(derivative_coefficients, current)
        differences = current[:, None] - current[None, :]
        safe_difference = off_diagonal & (jnp.abs(differences) > 64.0 * eps)
        guarded_difference = jnp.where(safe_difference, differences, 1.0 + 0.0j)
        reciprocal_sum = jnp.sum(
            jnp.where(
                safe_difference,
                1.0 / guarded_difference,
                0.0 + 0.0j,
            ),
            axis=1,
        )
        denominator = derivative - value * reciprocal_sum
        safe = jnp.abs(denominator) > 64.0 * eps
        updated = jnp.where(safe, current - value / denominator, current)
        return jnp.where(jnp.isfinite(updated), updated, current)

    polynomial_roots = jax.lax.fori_loop(0, 16, ea_step, predicted_roots)
    finite = jnp.isfinite(polynomial_roots.real) & jnp.isfinite(polynomial_roots.imag)
    unit_error = jnp.abs(jnp.abs(polynomial_roots) - 1.0)
    unit_tolerance = jnp.minimum(2048.0 * jnp.sqrt(eps), 1.0e-3)
    angles = jnp.angle(polynomial_roots)

    def polish_angle(_, current):
        value = evaluate_fourier(coefficients, current)
        derivative = evaluate_fourier_derivative(coefficients, current)
        safe = jnp.abs(derivative) > 16.0 * eps
        step = jnp.where(safe, value / derivative, 0.0)
        return current - jnp.clip(step, -0.25, 0.25)

    angles = jax.lax.fori_loop(0, 5, polish_angle, angles)
    residual = jnp.abs(evaluate_fourier(coefficients, angles))
    residual_tolerance = padding + jnp.maximum(
        8192.0 * eps,
        jnp.asarray(1.0e-11, dtype=radius.dtype),
    )
    residual_ok = residual <= residual_tolerance
    unit_candidate = finite & (unit_error <= unit_tolerance)
    rescued = _reciprocal_pair_rescue(
        polynomial_roots,
        finite,
        unit_error,
        residual_ok,
        unit_candidate,
        unit_tolerance,
    )
    valid = (unit_candidate | rescued) & residual_ok
    n_valid = jnp.sum(valid, dtype=jnp.int32)
    root_residual = jnp.abs(jnp.polyval(polynomial, polynomial_roots))
    powers = jnp.arange(polynomial.size - 1, -1, -1, dtype=radius.dtype)
    root_scale = jnp.sum(
        jnp.abs(polynomial)[None, :]
        * jnp.abs(polynomial_roots)[:, None] ** powers[None, :],
        axis=1,
    )
    near_unit_circle = jnp.abs(
        jnp.log(jnp.maximum(jnp.abs(polynomial_roots), eps))
    ) <= jnp.log(jnp.asarray(2.0, dtype=radius.dtype))
    polynomial_roots_ok = jnp.all(
        ~near_unit_circle
        | (root_residual <= 8192.0 * eps * jnp.maximum(root_scale, eps))
    )
    roots_ok = jnp.all(finite) & ((n_valid % 2) == 0) & polynomial_roots_ok
    sorted_angles = jnp.sort(
        jnp.where(valid, jnp.mod(angles, 2.0 * jnp.pi), 2.0 * jnp.pi)
    )

    # The algebraic residual is authoritative.  This phase audit is an
    # additional guard against a co-converged pair, not an absence proof.
    audit_angles = 2.0 * jnp.pi * jnp.arange(32, dtype=radius.dtype) / 32.0
    audit_inside = evaluate_fourier(coefficients, audit_angles) <= 0.0
    sign_changes = jnp.sum(audit_inside != jnp.roll(audit_inside, -1), dtype=jnp.int32)
    represented = n_valid
    topology_ok = sign_changes <= represented
    status_ok = roots_ok & topology_ok & ~degenerate
    diagnostic_status = jnp.bitwise_or(
        jnp.where(
            roots_ok & ~degenerate,
            jnp.int32(0),
            jnp.int32(ANGULAR_ROOT_FAILURE),
        ),
        jnp.where(
            topology_ok,
            jnp.int32(0),
            jnp.int32(_ANGULAR_CONTINUATION_FAILURE),
        ),
    )
    current = _ThetaRoots(
        sorted_angles,
        jnp.arange(polynomial_roots.size, dtype=jnp.int32) < n_valid,
        radius,
        polynomial,
        polynomial_roots,
        jnp.bitwise_or(
            previous.status,
            jnp.where(status_ok, jnp.int32(ANGULAR_OK), diagnostic_status),
        ),
    )
    return current, _angular_measure(coefficients, current)


def _angular_measure(coefficients: Array, roots: _ThetaRoots) -> Array:
    """Return total inside angle from a sorted fixed-capacity root set."""

    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=roots.angles.dtype)
    boundaries = jnp.concatenate(
        (jnp.zeros(1, dtype=roots.angles.dtype), roots.angles, two_pi[None])
    )
    lower = boundaries[:-1]
    upper = boundaries[1:]
    width = jnp.maximum(upper - lower, 0.0)
    inside = evaluate_fourier(coefficients, 0.5 * (lower + upper)) <= 0.0
    return jnp.sum(jnp.where(inside, width, 0.0))


def _integrate_cell(
    w_center: Array,
    rho: Array,
    bounds: Array,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array, Array]:
    """Apply one endpoint-transformed GL11/G7 pair with root continuation."""

    return _integrate_cell_rule(
        w_center,
        rho,
        bounds,
        nodes=GL11_X,
        high_weights=GL11_W,
        low_weights=G7_W_ON_GL11,
        s=s,
        q=q,
    )


def _integrate_cell_rule(
    w_center: Array,
    rho: Array,
    bounds: Array,
    *,
    nodes,
    high_weights,
    low_weights,
    s: Array,
    q: Array,
) -> tuple[Array, Array, Array]:
    """Apply one endpoint transform and a nested fixed quadrature pair."""

    dtype = bounds.dtype
    nodes = jnp.asarray(nodes, dtype=dtype)
    angle = 0.25 * jnp.pi * (nodes + 1.0)
    width = bounds[1] - bounds[0]
    radii = bounds[0] + width * jnp.sin(angle) ** 2
    jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * angle)
    fouriers = jax.vmap(lambda radius: _fourier(w_center, rho, radius, s=s, q=q))(radii)
    oscillatory_bound = 2.0 * jnp.sum(jnp.abs(fouriers.coefficients[:, 1:]), axis=1)
    constant = jnp.real(fouriers.coefficients[:, 0])
    entirely_inside = constant + oscillatory_bound + fouriers.padding < 0.0
    entirely_outside = constant - oscillatory_bound - fouriers.padding > 0.0
    mixed = ~(entirely_inside | entirely_outside | fouriers.degenerate)
    # The coefficient triangle bound can remain inconclusive on an entirely
    # outside ring.  Initialising at the geometric cell midpoint then misses a
    # short image interval confined near one radial endpoint.  A 32-phase
    # level-set audit is much cheaper than one polynomial solve; initialise at
    # the ring with the deepest sampled interior instead.
    audit_angles = 2.0 * jnp.pi * jnp.arange(32, dtype=dtype) / 32.0
    audit_values = jax.vmap(
        lambda coefficients: evaluate_fourier(coefficients, audit_angles)
    )(fouriers.coefficients)
    anchor_score = jnp.min(audit_values, axis=1)
    # The phase audit chooses a well-conditioned anchor but is not an absence
    # certificate: an arbitrarily narrow negative arc can fall between all
    # sampled phases.  Every ring left inconclusive by the exact coefficient
    # triangle bound therefore remains in the root-continuation schedule.
    needs_roots = mixed
    n_root_nodes = jnp.sum(needs_roots, dtype=jnp.int32)
    anchor = jnp.argmin(jnp.where(needs_roots, anchor_score, jnp.inf))
    anchor = jnp.where(n_root_nodes > 0, anchor, nodes.size // 2)

    direct_measure = jnp.where(entirely_inside, 2.0 * jnp.pi, 0.0)

    def initialize(_):
        roots = _initial_theta_roots_from_fourier(
            fouriers.coefficients[anchor],
            fouriers.padding[anchor],
            fouriers.degenerate[anchor],
            radii[anchor],
        )
        measure = _angular_measure(fouriers.coefficients[anchor], roots)
        return roots, measure

    def initialize_uniform(_):
        return (
            _ThetaRoots(
                jnp.zeros(6, dtype=dtype),
                jnp.zeros(6, dtype=jnp.bool_),
                radii[anchor],
                jnp.zeros(7, dtype=jnp.result_type(dtype, jnp.complex64)),
                jnp.zeros(
                    6,
                    dtype=(jnp.complex128 if dtype == jnp.float64 else jnp.complex64),
                ),
                jnp.int32(ANGULAR_OK),
            ),
            jnp.where(entirely_inside[anchor], 2.0 * jnp.pi, 0.0),
        )

    initial, initial_measure = jax.lax.cond(
        n_root_nodes > 0, initialize, initialize_uniform, operand=None
    )
    continued, continued_measure = jax.vmap(
        lambda coefficients, padding, degenerate, radius: _continue_theta_roots(
            initial,
            coefficients,
            padding,
            degenerate,
            radius,
        )
    )(
        fouriers.coefficients,
        fouriers.padding,
        fouriers.degenerate,
        radii,
    )
    measures = jnp.where(needs_roots, continued_measure, direct_measure)
    measures = measures.at[anchor].set(initial_measure)
    statuses = jnp.where(needs_roots, continued.status, jnp.int32(ANGULAR_OK))
    statuses = statuses.at[anchor].set(initial.status)
    values = radii * measures
    kronrod = jnp.sum(jnp.asarray(high_weights, dtype=dtype) * jacobian * values)
    gauss = jnp.sum(jnp.asarray(low_weights, dtype=dtype) * jacobian * values)
    return kronrod, jnp.abs(kronrod - gauss), jnp.bitwise_or.reduce(statuses)


def _integrate_cell_independent(
    w_center: Array,
    rho: Array,
    bounds: Array,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array, Array]:
    """Reference GL11/G7 cell that solves every radial ring independently."""

    dtype = bounds.dtype
    nodes = jnp.asarray(GL11_X, dtype=dtype)
    angle = 0.25 * jnp.pi * (nodes + 1.0)
    width = bounds[1] - bounds[0]
    radii = bounds[0] + width * jnp.sin(angle) ** 2
    jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * angle)

    fouriers = jax.vmap(
        lambda radius: _fourier(w_center, rho, radius, s=s, q=q)
    )(radii)
    oscillatory_bound = 2.0 * jnp.sum(
        jnp.abs(fouriers.coefficients[:, 1:]), axis=1
    )
    constant = jnp.real(fouriers.coefficients[:, 0])
    entirely_inside = constant + oscillatory_bound + fouriers.padding < 0.0
    entirely_outside = constant - oscillatory_bound - fouriers.padding > 0.0
    mixed = ~(entirely_inside | entirely_outside | fouriers.degenerate)

    def solve(coefficients, padding, degenerate, radius):
        roots = _initial_theta_roots_from_fourier(
            coefficients,
            padding,
            degenerate,
            radius,
            use_companion=False,
        )
        return (
            _angular_measure(coefficients, roots),
            roots.status,
        )

    root_measure, root_status = jax.vmap(solve)(
        fouriers.coefficients,
        fouriers.padding,
        fouriers.degenerate,
        radii,
    )
    direct_measure = jnp.where(entirely_inside, 2.0 * jnp.pi, 0.0)
    measures = jnp.where(mixed, root_measure, direct_measure)
    statuses = jnp.where(mixed, root_status, jnp.int32(ANGULAR_OK))
    values = radii * measures
    kronrod = jnp.sum(jnp.asarray(GL11_W, dtype=dtype) * jacobian * values)
    gauss = jnp.sum(jnp.asarray(G7_W_ON_GL11, dtype=dtype) * jacobian * values)
    return kronrod, jnp.abs(kronrod - gauss), jnp.bitwise_or.reduce(statuses)


def _integrate_cell_limb(
    w_center: Array,
    rho: Array,
    bounds: Array,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array, Array]:
    """Integrate one cell with the same fixed radial rule in every geometry.

    The cell uses one nested GK15/G7 rule after a ``sin**2`` radial transform. The topology
    builder supplies two fixed numerical panels per geometric cell. The transform
    resolves the square-root endpoint behaviour at source-limb contacts while
    every fixed-radius angular boundary is solved from the same degree-six
    level-set polynomial.
    """

    dtype = bounds.dtype
    lower, upper = bounds
    width = upper - lower
    def integrate_cell(_):
        def integrate_polynomial(_):
            nodes = jnp.asarray(GK15_X, dtype=dtype)
            angle = 0.25 * jnp.pi * (nodes + 1.0)
            panel_lower = jnp.stack((lower, lower + 0.5 * width))
            panel_width = 0.5 * width
            radii = (
                panel_lower[:, None]
                + panel_width * jnp.sin(angle)[None, :] ** 2
            ).reshape(-1)
            jacobian = (
                0.25
                * jnp.pi
                * panel_width
                * jnp.broadcast_to(jnp.sin(2.0 * angle), (2, nodes.size))
            ).reshape(-1)
            fouriers = jax.vmap(
                lambda radius: _fourier(w_center, rho, radius, s=s, q=q)
            )(radii)
            oscillatory_bound = 2.0 * jnp.sum(
                jnp.abs(fouriers.coefficients[:, 1:]), axis=1
            )
            constant = jnp.real(fouriers.coefficients[:, 0])
            entirely_inside = (
                constant + oscillatory_bound + fouriers.padding < 0.0
            )
            entirely_outside = (
                constant - oscillatory_bound - fouriers.padding > 0.0
            )
            mixed = ~(
                entirely_inside | entirely_outside | fouriers.degenerate
            )

            def solve(coefficients, padding, degenerate, radius):
                roots = _initial_theta_roots_tangent_from_fourier(
                    coefficients,
                    padding,
                    degenerate,
                    radius,
                )
                return _angular_measure(coefficients, roots), roots.status

            root_measure, root_status = jax.vmap(solve)(
                fouriers.coefficients,
                fouriers.padding,
                fouriers.degenerate,
                radii,
            )
            measures = jnp.where(
                mixed,
                root_measure,
                jnp.where(entirely_inside, 2.0 * jnp.pi, 0.0),
            )
            statuses = jnp.where(
                mixed, root_status, jnp.int32(ANGULAR_OK)
            )
            values = radii * measures
            high = jnp.sum(
                jnp.tile(jnp.asarray(GK15_W, dtype=dtype), 2)
                * jacobian
                * values
            )
            low = jnp.sum(
                jnp.tile(jnp.asarray(G7_W_ON_GK15, dtype=dtype), 2)
                * jacobian
                * values
            )
            return high, jnp.abs(high - low), jnp.bitwise_or.reduce(statuses)

        # The cancellation-resistant level-set coefficients remain scaled at
        # low q, including rings through a lens-pole radius.  Use the same
        # fixed-r angular polynomial on every crossed cell.  The traced limb
        # supplies conservative radial support and breakpoints; it need not
        # launch a large Newton seed pool at every quadrature node.
        return integrate_polynomial(None)

    # Do not classify a whole radial cell from finitely sampled limb segments.
    # A narrow angular image can lie between those samples and make an
    # apparently constant outside cell carry finite area.  Every cell follows
    # the same fixed rule; the coefficient bounds inside that rule still avoid
    # angular root solves for rigorously full or empty *rings*.
    return integrate_cell(None)


def _integrate_cell_atlas(
    w_center: Array,
    rho: Array,
    bounds: Array,
    atlas: PolarBranchAtlas,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array, Array]:
    """Apply GL11/G7 after seeding every mixed ring from the limb atlas."""

    dtype = bounds.dtype
    nodes = jnp.asarray(GL11_X, dtype=dtype)
    angle = 0.25 * jnp.pi * (nodes + 1.0)
    width = bounds[1] - bounds[0]
    radii = bounds[0] + width * jnp.sin(angle) ** 2
    jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * angle)
    fouriers = jax.vmap(lambda radius: _fourier(w_center, rho, radius, s=s, q=q))(radii)
    oscillatory_bound = 2.0 * jnp.sum(jnp.abs(fouriers.coefficients[:, 1:]), axis=1)
    constant = jnp.real(fouriers.coefficients[:, 0])
    entirely_inside = constant + oscillatory_bound + fouriers.padding < 0.0
    entirely_outside = constant - oscillatory_bound - fouriers.padding > 0.0
    mixed = ~(entirely_inside | entirely_outside | fouriers.degenerate)

    atlas_measure = jax.vmap(
        lambda coefficients, padding, degenerate, radius: polar_atlas_angular_measure(
            atlas,
            coefficients,
            padding,
            degenerate,
            radius,
        )
    )(
        fouriers.coefficients,
        fouriers.padding,
        fouriers.degenerate,
        radii,
    )
    direct_measure = jnp.where(entirely_inside, 2.0 * jnp.pi, 0.0)
    measures = jnp.where(mixed, atlas_measure.measure, direct_measure)
    statuses = jnp.where(mixed, atlas_measure.status, jnp.int32(0))
    count_consistent = jnp.all(
        atlas_measure.n_crossings == atlas_measure.n_crossings[0]
    )
    statuses = jnp.bitwise_or(
        statuses,
        jnp.where(
            count_consistent,
            jnp.int32(0),
            jnp.int32(POLAR_ATLAS_TOPOLOGY),
        ),
    )
    values = radii * measures
    high = jnp.sum(jnp.asarray(GL11_W, dtype=dtype) * jacobian * values)
    low = jnp.sum(jnp.asarray(G7_W_ON_GL11, dtype=dtype) * jacobian * values)
    return high, jnp.abs(high - low), jnp.bitwise_or.reduce(statuses)


def mag_uniform_radial_continuation_cpu(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    n_limb: int = 64,
    independent_roots: bool = False,
    use_polar_atlas: bool = False,
    use_simple_support: bool = False,
    use_refined_limb: bool | None = None,
) -> RadialContinuationResult:
    """Evaluate uniform magnification with radial topology and theta continuation."""

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    if use_refined_limb is None:
        # The production simple-support experiment uses the fixed-budget
        # 32+32 trace.  Legacy atlas/continuation research routes retain their
        # uniform phases because their interpolation assumes equal spacing.
        use_refined_limb = use_simple_support
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    s = jnp.asarray(s, dtype=w_center.real.dtype)
    q = jnp.asarray(q, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    limb_phases = None
    if use_refined_limb:
        if n_limb % 2:
            raise ValueError("two-stage limb refinement requires even n_limb")
        trace = trace_binary_source_limb_two_stage(
            w_center, rho, s=s, q=q, n_coarse=n_limb // 2
        )
        image_limb = trace.image_limb
        physical_mask = trace.physical_mask
        limb_phases = trace.phases
    else:
        image_limb, physical_mask = trace_binary_source_limb(
            w_center, rho, s=s, q=q, n_limb=n_limb, include_all_roots=False
        )
    return _mag_uniform_radial_from_trace(
        w_center,
        rho,
        s=s,
        q=q,
        rtol=rtol,
        image_limb=image_limb,
        physical_mask=physical_mask,
        independent_roots=independent_roots,
        use_polar_atlas=use_polar_atlas,
        use_simple_support=use_simple_support,
        limb_phases=limb_phases,
    )


def _mag_uniform_radial_from_trace(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    rtol: Array,
    image_limb: Array,
    physical_mask: Array,
    independent_roots: bool,
    use_polar_atlas: bool = False,
    use_simple_support: bool = False,
    limb_phases: Array | None = None,
    check_tolerance: bool = True,
) -> RadialContinuationResult:
    """Evaluate the radial rule from an already available source-limb trace."""

    lens = binary_geometry(s, q)
    atlas = None
    if use_polar_atlas:
        atlas = refine_polar_atlas_tangencies(
            build_polar_branch_atlas(
                image_limb,
                physical_mask,
            ),
            w_center,
            rho,
            s=s,
            q=q,
        )
    origin_inside = (
        binary_level_set(
            jnp.asarray(0.0 + 0.0j, dtype=w_center.dtype),
            w_center - lens.shifted,
            rho,
            lens.shifted,
            a=lens.a,
            e1=lens.e1,
        )
        <= 0.0
    )
    if use_simple_support:
        topology = build_simple_polar_topology(
            image_limb,
            physical_mask,
            w_center,
            rho,
            s=s,
            q=q,
            origin_inside=origin_inside,
            phases=limb_phases,
        )
    else:
        topology = define_radial_topology(
            image_limb,
            physical_mask,
            rho,
            margin_r=0.5,
            origin_inside=origin_inside,
            track_roots=False,
            binary_margin_parameters=(lens.shifted, lens.a, lens.e1),
            filter_roundoff_turning_points=True,
            turning_radii_override=(
                polar_atlas_turning_radius_override(atlas, image_limb)
                if use_polar_atlas
                else None
            ),
        )
    normalization = jnp.pi * rho**2

    def integrate_one(index, state):
        value, error, status = state
        if use_polar_atlas:
            cell_value, cell_error, cell_status = _integrate_cell_atlas(
                w_center,
                rho,
                topology.intervals[index],
                atlas,
                s=s,
                q=q,
            )
        elif use_simple_support:
            cell_value, cell_error, cell_status = _integrate_cell_limb(
                w_center,
                rho,
                topology.intervals[index],
                s=s,
                q=q,
            )
        elif independent_roots:
            cell_value, cell_error, cell_status = _integrate_cell_independent(
                w_center, rho, topology.intervals[index], s=s, q=q
            )
        else:
            cell_value, cell_error, cell_status = _integrate_cell(
                w_center, rho, topology.intervals[index], s=s, q=q
            )
        return (
            value + cell_value,
            error + cell_error,
            jnp.bitwise_or(status, cell_status),
        )

    value, error, status = jax.lax.fori_loop(
        jnp.int32(0),
        topology.n_intervals,
        integrate_one,
        (
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            jnp.asarray(0.0, dtype=w_center.real.dtype),
            topology.status,
        ),
    )
    magnification = value / normalization
    estimated_error = error / normalization
    scale = jnp.maximum(jnp.abs(magnification), 1.0)
    # Embedded radial differences are acceptance heuristics rather than strict
    # error bounds.  Tighten the direct-limb acceptance when the tracked limb
    # contains partial topology; dense audits expose shared support bias beyond
    # the embedded radial difference.  This changes only status, not the value
    # or execution schedule.
    if use_polar_atlas:
        # The embedded radial difference measures quadrature error but not the
        # residual source-limb interpolation error.  The dense non-resonant
        # q=1e-6 audit found a 6.77x underestimate at one fold-normal point;
        # a 0.15 acceptance factor rejects it without changing the fixed rule.
        error_safety = 0.15
    elif use_simple_support:
        # The embedded pair measures radial convergence but can underestimate
        # the value error near a close contact even when every physical limb
        # branch is present.  The dense planetary audit requires 0.27 to reject
        # the worst shared-bias pair while retaining the independent fragmented
        # fold regression.  Use this one fixed factor for every geometric cell;
        # acceptance must not depend on a topology-specific routing label.
        error_safety = jnp.asarray(0.27, dtype=scale.dtype)
    else:
        error_safety = 0.5 if independent_roots else 1.0
    if check_tolerance:
        accepted = (
            (status == RADIAL_OK)
            & jnp.isfinite(magnification)
            & jnp.isfinite(estimated_error)
            & (estimated_error <= error_safety * rtol * scale)
        )
        status = jnp.where(
            accepted,
            jnp.int32(RADIAL_OK),
            jnp.bitwise_or(status, jnp.int32(RADIAL_TOLERANCE)),
        )
    return RadialContinuationResult(
        magnification,
        estimated_error,
        topology.n_intervals,
        jnp.where(
            use_simple_support,
            (2 * GK15_X.size) * topology.n_intervals,
            jnp.where(
                independent_roots,
                11 * topology.n_intervals,
                jnp.where(use_polar_atlas, 0, topology.n_intervals),
            ),
        ),
        jnp.where(
            use_simple_support | independent_roots,
            0,
            11 * topology.n_intervals,
        ),
        status,
    )


__all__ = ["RadialContinuationResult", "mag_uniform_radial_continuation_cpu"]
