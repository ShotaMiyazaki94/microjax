"""Angle-first radial-moment integration for uniform binary sources."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from microjax.point_source import lens_eq

from ..geometry.lens import binary_geometry
from .sentinel import hidden_caustic_candidate
from .support import trace_binary_source_limb, tracked_limb_neighbors
from .quadrature import G7_W_ON_GK15, GK15_W, GK15_X
from .roots import (
    batched_companion_roots,
    batched_fixed_ea_roots,
    batched_polished_real_companion_roots,
)

Array = jnp.ndarray
ANGULAR_MOMENT_TOPOLOGY = 1 << 17
ANGULAR_MOMENT_EXHAUSTED = 1 << 18
ANGULAR_MOMENT_SUPPORT = 1 << 19


def _radial_image_bound(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> Array:
    """Return a finite image-radius bound for radial polynomial scaling."""

    lens = binary_geometry(s, q)
    real_dtype = w_center.real.dtype
    lens_radius = jnp.maximum(
        jnp.abs(lens.shifted - lens.a),
        jnp.abs(lens.shifted + lens.a),
    )
    source_bound = jnp.abs(w_center) + rho
    radial_offset = source_bound - lens_radius
    image_bound = lens_radius + 0.5 * (
        radial_offset + jnp.sqrt(radial_offset**2 + 4.0)
    )
    return image_bound * (1.0 + 32.0 * jnp.finfo(real_dtype).eps)


class AngularMomentResult(NamedTuple):
    """Fixed-order angular-moment value and root diagnostics."""

    magnification: Array
    estimated_error: Array
    n_theta: Array
    invalid_root_count: Array
    ghost_residual_ratio: Array
    limb_topology: Array
    status: Array


class AngularSupport(NamedTuple):
    """Angular cells and the diagnostics required to certify them."""

    cells: Array
    active: Array
    topology_uncertain: Array
    minimum_ghost_residual: Array
    limb_topology: Array
    tangencies_valid: Array


def _clenshaw_curtis_rule(n_intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the nested ``n_intervals + 1`` rule on ``[-1, 1]``."""

    if n_intervals < 2 or n_intervals % 2:
        raise ValueError("n_intervals must be an even integer >= 2")
    theta = np.pi * np.arange(n_intervals + 1) / n_intervals
    nodes = np.cos(theta)
    weights = np.zeros(n_intervals + 1)
    interior = np.arange(1, n_intervals)
    accumulator = np.ones(n_intervals - 1)
    if n_intervals % 2 == 0:
        weights[0] = weights[-1] = 1.0 / (n_intervals**2 - 1.0)
        for harmonic in range(1, n_intervals // 2):
            accumulator -= (
                2.0
                * np.cos(2.0 * harmonic * theta[interior])
                / (4.0 * harmonic**2 - 1.0)
            )
        accumulator -= np.cos(n_intervals * theta[interior]) / (n_intervals**2 - 1.0)
    weights[interior] = 2.0 * accumulator / n_intervals
    return nodes, weights


def binary_radial_level_set_coefficients(
    theta: Array,
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
) -> Array:
    """Return real degree-six coefficients of ``H(r exp(i theta))``."""

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    rho = jnp.asarray(rho, dtype=real_dtype)
    lens = binary_geometry(s, q)
    unit = jnp.exp(1.0j * jnp.asarray(theta, dtype=real_dtype))
    unit_bar = jnp.conjugate(unit)
    shifted = jnp.asarray(lens.shifted, dtype=real_dtype)

    denominator = jnp.asarray(
        [shifted**2 - lens.a**2, -2.0 * shifted * unit_bar, unit_bar**2],
        dtype=w_center.dtype,
    )
    deflection_constant = -shifted + lens.a * (2.0 * lens.e1 - 1.0)
    numerator = jnp.asarray(
        [
            -w_center * denominator[0] - deflection_constant,
            unit * denominator[0] - w_center * denominator[1] - unit_bar,
            unit * denominator[1] - w_center * denominator[2],
            unit * denominator[2],
        ],
        dtype=w_center.dtype,
    )
    numerator_square = jnp.convolve(numerator, jnp.conjugate(numerator))
    denominator_square = jnp.convolve(denominator, jnp.conjugate(denominator))
    padded_denominator = jnp.pad(denominator_square, (0, 2))
    ascending = jnp.real(numerator_square - rho**2 * padded_denominator)
    descending = ascending[::-1]
    scale = jnp.max(jnp.abs(descending))
    return descending / jnp.maximum(scale, jnp.finfo(real_dtype).tiny)


def _stable_radial_level_value(
    angles: Array,
    radii: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> Array:
    """Evaluate ``|N| - rho |D|`` without squared-level-set cancellation."""

    lens = binary_geometry(s, q)
    unit = jnp.exp(1.0j * angles)[..., None]
    z = radii * unit - lens.shifted
    zbar = jnp.conjugate(z)
    d_plus = zbar - lens.a
    d_minus = zbar + lens.a
    denominator = d_plus * d_minus
    source_shifted = w_center - lens.shifted
    numerator = (
        (z - source_shifted) * denominator
        - lens.e1 * d_minus
        - (1.0 - lens.e1) * d_plus
    )
    numerator_abs = jnp.abs(numerator)
    denominator_abs = rho * jnp.abs(denominator)
    return numerator_abs - denominator_abs


def _ray_intervals_from_roots(
    coefficients: Array,
    roots: Array,
    stable_context: tuple[Array, Array, Array, Array, Array] | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Select positive real intervals from an already solved root batch."""

    real = roots.real
    scale = jnp.maximum(1.0, jnp.abs(real))
    nearly_real = jnp.abs(roots.imag) <= 2.0e-7 * scale
    positive = real >= 0.0
    residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    valid = nearly_real & positive & (residual <= 2.0e-7)
    radii = jnp.sort(jnp.where(valid, real, jnp.inf), axis=-1)
    maximum = jnp.max(jnp.where(valid, real, 0.0), axis=-1)
    outer = jnp.maximum(maximum + 1.0, 1.0)
    clean = jnp.where(jnp.isfinite(radii), radii, outer[:, None])
    boundaries = jnp.concatenate(
        (jnp.zeros_like(outer[:, None]), clean, outer[:, None]), axis=-1
    )
    lower = boundaries[:, :-1]
    upper = boundaries[:, 1:]
    midpoint = 0.5 * (lower + upper)
    suspicious = (~valid) & nearly_real & positive
    suspicious_count = jnp.sum(suspicious, axis=-1, dtype=jnp.int32)

    if stable_context is None:
        values = jax.vmap(jax.vmap(jnp.polyval, in_axes=(None, 0)))(
            coefficients,
            midpoint,
        )
        inside = values <= 0.0
    else:
        angles, w_center, rho, s, q = stable_context
        stable_level = _stable_radial_level_value(
            angles,
            midpoint,
            w_center,
            rho,
            s=s,
            q=q,
        )
        inside = stable_level <= 0.0

    return lower, upper, inside, suspicious_count


def _ray_intervals_complex(
    coefficients: Array,
) -> tuple[Array, Array, Array, Array]:
    """Preserve the historical complex solver for limb-darkened profiles."""

    return _ray_intervals_from_roots(
        coefficients,
        batched_companion_roots(coefficients),
    )


def _ray_intervals(
    coefficients: Array,
    *,
    root_mode: str = "companion",
    ordinate_bound: Array | None = None,
    stable_context: tuple[Array, Array, Array, Array, Array] | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Return positive radial intervals using a fixed root schedule."""

    ea_uncertain = None
    if root_mode.startswith("ea_auto"):
        try:
            iterations = int(root_mode.removeprefix("ea_auto"))
        except ValueError as exc:
            raise ValueError(f"invalid radial EA iteration count: {root_mode}") from exc
        if iterations <= 0:
            raise ValueError(f"invalid radial EA iteration count: {iterations}")
        # The EA arithmetic amortizes only once the batch is large.  Polar
        # support certification uses a large midpoint batch, while each
        # quadrature cell is deliberately a tiny 8/12/24/32 batch.  Keep the
        # latter on the faster companion path automatically.
        root_mode = (
            f"ea_fixed{iterations}"
            if coefficients.shape[0] >= 64
            else "companion"
        )

    if root_mode == "companion":
        # Radial level-set coefficients are real.  The polished real companion
        # solver preserves the existing residual checks and implicit root JVP.
        trial_roots = batched_polished_real_companion_roots(coefficients)
        trial_real = trial_roots.real
        trial_scale = jnp.maximum(1.0, jnp.abs(trial_real))
        trial_nearly_real = jnp.abs(trial_roots.imag) <= 2.0e-7 * trial_scale
        trial_positive = trial_real >= 0.0
        trial_residual = jnp.abs(
            jax.vmap(jnp.polyval)(coefficients, trial_roots)
        )
        # Preserve the former complex-solver coverage on numerically marginal
        # rays. A batch-wide fallback keeps ordinary rays cheap.
        needs_complex_fallback = (
            jnp.any(
                ~jnp.isfinite(trial_roots.real)
                | ~jnp.isfinite(trial_roots.imag)
            )
            | jnp.any(
                trial_nearly_real
                & trial_positive
                & (trial_residual > 2.0e-7)
            )
        )
        roots = jax.lax.cond(
            jax.lax.stop_gradient(needs_complex_fallback),
            lambda _: batched_companion_roots(coefficients),
            lambda _: trial_roots,
            operand=None,
        )
    elif root_mode.startswith("ea_fixed"):
        if ordinate_bound is None:
            raise ValueError("fixed radial EA requires ordinate_bound")
        try:
            iterations = int(root_mode.removeprefix("ea_fixed"))
        except ValueError as exc:
            raise ValueError(f"invalid radial EA root_mode: {root_mode}") from exc
        if iterations <= 0:
            raise ValueError(f"invalid radial EA iteration count: {iterations}")
        trial_roots = batched_fixed_ea_roots(
            coefficients,
            ordinate_bound,
            iterations,
        )
        trial_scale = jnp.maximum(1.0, jnp.abs(trial_roots.real))
        trial_residual = jnp.abs(
            jax.vmap(jnp.polyval)(coefficients, trial_roots)
        )
        degree = trial_roots.shape[-1]
        off_diagonal = ~jnp.eye(degree, dtype=bool)
        root_separation = jnp.abs(
            trial_roots[..., :, None] - trial_roots[..., None, :]
        )
        # Keep the imaginary part out of this gate: an under-polished
        # conjugate pair can still have a small positive real part while its
        # imaginary split is much larger than the acceptance tolerance.
        relevant = trial_roots.real >= 0.0
        # EA is ill-conditioned at nearly multiple roots.  In that regime a
        # tiny residual does not imply an accurate split of the roots, and
        # the interval area can move by much more than the residual.  Preserve
        # the established companion result for that rare case.  The test is a
        # single batch-wide guard, matching the existing complex fallback and
        # avoiding per-ray control-flow in the hot path.
        clustered = jnp.any(
            off_diagonal
            & relevant[..., :, None]
            & relevant[..., None, :]
            & (
                root_separation
                <= 4.0e-5 * trial_scale[..., :, None]
            ),
            axis=(-1, -2),
        )
        ea_uncertain = (
            jnp.any(
                ~jnp.isfinite(trial_roots.real)
                | ~jnp.isfinite(trial_roots.imag)
                | (trial_residual > 2.0e-7),
                axis=-1,
            )
            | clustered
        )
        needs_companion_fallback = jnp.any(
            ea_uncertain
        )
        roots = jax.lax.cond(
            jax.lax.stop_gradient(needs_companion_fallback),
            lambda _: batched_polished_real_companion_roots(coefficients),
            lambda _: trial_roots,
            operand=None,
        )
    else:
        raise ValueError(f"unknown radial root_mode: {root_mode}")
    lower, upper, inside, suspicious = _ray_intervals_from_roots(
        coefficients,
        roots,
        stable_context,
    )
    if ea_uncertain is not None:
        # Companion is only a legacy-value fallback here, not a proof that the
        # near-multiple split is correct.  Preserve fail-closed semantics by
        # carrying the EA uncertainty into the invalid-root count.
        suspicious = suspicious + ea_uncertain.astype(jnp.int32)
    return lower, upper, inside, suspicious


def _ray_moments(
    coefficients: Array,
    *,
    root_mode: str = "companion",
    ordinate_bound: Array | None = None,
    stable_context: tuple[Array, Array, Array, Array, Array] | None = None,
) -> tuple[Array, Array]:
    lower, upper, inside, suspicious = _ray_intervals(
        coefficients,
        root_mode=root_mode,
        ordinate_bound=ordinate_bound,
        stable_context=stable_context,
    )
    moment = 0.5 * jnp.sum(
        jnp.where(inside, (upper - lower) * (upper + lower), 0.0), axis=-1
    )
    return moment, suspicious


def _endpoint_preserving_union_mask(
    cells: Array,
    physical_limb: Array,
    physical_mask: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    support_uncertain: Array,
    root_mode: str = "companion",
    ordinate_bound: Array | None = None,
) -> Array:
    """Return the occupied atomic cells without merging their endpoints.

    The source-limb extrema already partition ``[0, 2 pi]`` at every detected
    angular support endpoint.  On a complete partition, image occupancy cannot
    change inside one cell, so one independent radial all-root solve at its
    midpoint identifies the union member.  The original endpoints are kept:
    neighbouring occupied intervals are deliberately *not* merged before the
    quadrature rule is applied.

    A physical limb sample forces its containing cell to remain active.  This
    preserves a component represented by only one or two samples near a fold,
    even when the cell midpoint lies just outside that very short component.
    If the trace or any required tangency is uncertain, no cell is removed;
    support reduction is an optimisation and is never allowed to become a
    correctness assumption.
    """

    nonempty = cells[:, 1] > cells[:, 0]

    def reduce_support(_):
        midpoint = 0.5 * (cells[:, 0] + cells[:, 1])
        coefficients = jax.vmap(
            lambda angle: binary_radial_level_set_coefficients(
                angle,
                w_center,
                rho,
                s=s,
                q=q,
            )
        )(midpoint)
        moments, suspicious = _ray_moments(
            coefficients,
            root_mode=root_mode,
            ordinate_bound=ordinate_bound,
            stable_context=(midpoint, w_center, rho, s, q),
        )
        occupied = (moments > 0.0) | (suspicious > 0) | ~jnp.isfinite(moments)

        sample_angles = jnp.mod(jnp.angle(physical_limb), 2.0 * jnp.pi).reshape(-1)
        sample_mask = physical_mask.reshape(-1) & jnp.isfinite(sample_angles)
        half_width = 0.5 * (cells[:, 1] - cells[:, 0])
        circular_distance = jnp.abs(
            jnp.angle(
                jnp.exp(
                    1.0j * (sample_angles[None, :] - midpoint[:, None])
                )
            )
        )
        endpoint_tolerance = 64.0 * jnp.finfo(cells.dtype).eps
        contains_limb_sample = jnp.any(
            sample_mask[None, :]
            & (circular_distance <= half_width[:, None] + endpoint_tolerance),
            axis=1,
        )
        return nonempty & (occupied | contains_limb_sample)

    return jax.lax.cond(
        jnp.asarray(support_uncertain),
        lambda _: nonempty,
        reduce_support,
        operand=None,
    )


def _split_overlapping_angular_support(
    support: AngularSupport,
    physical_limb: Array,
    physical_mask: Array,
    *,
    parts: int,
) -> AngularSupport:
    """Split the one active cell containing overlapping traced image branches.

    A single continuously traced image branch contributes at most ``n_limb``
    samples to one angular cell.  A larger count therefore identifies a cell
    where multiple image branches overlap in angle.  Those cells are the ones
    in which the radial moment can change too sharply for one fixed angular
    panel, even though the support endpoints themselves are complete.

    The split reuses inactive entries in the fixed-size support array.  It
    performs no lens-equation solve and leaves all support diagnostics and
    endpoints unchanged.  If the support array has no spare entries, the
    original support is returned unchanged and its existing structural status
    remains authoritative.
    """

    if parts < 2:
        raise ValueError("parts must be at least 2")

    cells = support.cells
    active = support.active
    sample_angles = jnp.mod(jnp.angle(physical_limb), 2.0 * jnp.pi).reshape(-1)
    sample_mask = physical_mask.reshape(-1) & jnp.isfinite(sample_angles)
    endpoint_tolerance = 64.0 * jnp.finfo(cells.dtype).eps
    sample_count = jnp.sum(
        sample_mask[None, :]
        & (sample_angles[None, :] >= cells[:, 0, None] - endpoint_tolerance)
        & (sample_angles[None, :] <= cells[:, 1, None] + endpoint_tolerance),
        axis=1,
        dtype=jnp.int32,
    )
    sample_count = jnp.where(active, sample_count, jnp.int32(0))
    selected = jnp.argmax(sample_count)

    spare = jnp.nonzero(~active, size=parts - 1, fill_value=-1)[0]
    can_split = (
        (sample_count[selected] > physical_limb.shape[-1])
        & jnp.all(spare >= 0)
    )
    safe_spare = jnp.maximum(spare, 0)

    def split(_):
        lower, upper = cells[selected]
        boundaries = jnp.linspace(lower, upper, parts + 1)
        subcells = jnp.stack((boundaries[:-1], boundaries[1:]), axis=-1)
        split_cells = cells.at[selected].set(subcells[0])
        split_cells = split_cells.at[safe_spare].set(subcells[1:])
        split_active = active.at[safe_spare].set(True)
        return support._replace(cells=split_cells, active=split_active)

    return jax.lax.cond(can_split, split, lambda _: support, operand=None)


def _split_angular_support_at_angle(
    support: AngularSupport,
    angle: Array,
    *,
    enabled: Array | bool,
) -> tuple[AngularSupport, Array]:
    """Insert one exact structural angle into a fixed-capacity support."""

    cells = support.cells
    active = support.active
    angle = jnp.mod(jnp.asarray(angle, dtype=cells.dtype), 2.0 * jnp.pi)
    tolerance = 64.0 * jnp.finfo(cells.dtype).eps
    contains = (
        active
        & (angle > cells[:, 0] + tolerance)
        & (angle < cells[:, 1] - tolerance)
    )
    selected = jnp.argmax(contains)
    spare = jnp.nonzero(~active, size=1, fill_value=-1)[0][0]
    applied = jnp.asarray(enabled) & jnp.any(contains) & (spare >= 0)
    safe_spare = jnp.maximum(spare, 0)

    def split(_):
        lower, upper = cells[selected]
        split_cells = cells.at[selected].set(jnp.stack((lower, angle)))
        split_cells = split_cells.at[safe_spare].set(jnp.stack((angle, upper)))
        split_active = active.at[safe_spare].set(True)
        return support._replace(cells=split_cells, active=split_active)

    return (
        jax.lax.cond(applied, split, lambda _: support, operand=None),
        applied,
    )


def _angular_support_cells(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_limb: int,
    root_mode: str = "companion",
) -> AngularSupport:
    # ``include_all_roots`` only overrides the returned mask; it does not
    # change the quintic coefficients or the tracked roots.  Keep the
    # algebraic roots from this single trace instead of solving the identical
    # quintic sequence a second time merely to expose ghost slots.
    image_limb, physical_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        include_all_roots=False,
    )
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
    trace_finite = jnp.all(jnp.isfinite(image_limb))
    topology_uncertain = topology_uncertain | (minimum_ghost_residual <= 4.0 * rho)
    topology_uncertain = topology_uncertain | ~trace_finite
    return _angular_support_cells_from_trace(
        w_center,
        rho,
        s=s,
        q=q,
        physical_limb=image_limb,
        physical_mask=physical_mask,
        topology_uncertain=topology_uncertain,
        minimum_ghost_residual=minimum_ghost_residual,
        limb_topology=limb_topology,
        root_mode=root_mode,
    )


def _angular_support_cells_from_trace(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    physical_limb: Array,
    physical_mask: Array,
    topology_uncertain: Array,
    minimum_ghost_residual: Array,
    limb_topology: Array,
    neighbors: tuple[Array, Array, Array, Array] | None = None,
    root_mode: str = "companion",
) -> AngularSupport:
    """Build polar support from an already traced source boundary.

    Cartesian and polar moments are two coordinate charts of the same image
    region. Accepting a shared trace here avoids a second boundary-root solve
    when a poorly conditioned Cartesian projection needs an independent
    angular certificate.
    """

    if neighbors is None:
        neighbors = tracked_limb_neighbors(physical_limb, physical_mask)
    previous_limb, following_limb, previous_mask, following_mask = neighbors
    previous_step = jnp.angle(physical_limb * jnp.conjugate(previous_limb))
    following_step = jnp.angle(following_limb * jnp.conjugate(physical_limb))
    sign_change = previous_step * following_step <= 0.0
    previous_endpoint = physical_mask & ~previous_mask
    following_endpoint = physical_mask & ~following_mask
    physical_endpoint = previous_endpoint | following_endpoint
    candidate_mask = physical_mask & (sign_change | physical_endpoint)

    center = jnp.angle(physical_limb)
    minus = center - previous_step
    plus = center + following_step
    curvature = minus - 2.0 * center + plus
    safe_curvature = jnp.where(
        jnp.abs(curvature) > 64.0 * jnp.finfo(center.dtype).eps,
        curvature,
        1.0,
    )
    raw_offset = 0.5 * (minus - plus) / safe_curvature
    offset = jnp.clip(raw_offset, -1.0, 1.0)
    fitted = center + 0.5 * (plus - minus) * offset + 0.5 * curvature * offset**2
    fitted = jnp.where(previous_mask & following_mask, fitted, center)
    fitted = jnp.mod(fitted, 2.0 * jnp.pi)

    maximum_extrema = 24
    flat_mask = candidate_mask.reshape(-1)
    isolated_endpoint = previous_endpoint & following_endpoint
    previous_transition_count = jnp.sum(previous_endpoint, axis=0)
    following_transition_count = jnp.sum(following_endpoint, axis=0)
    previous_transition = jnp.sum(
        jnp.where(previous_endpoint, physical_limb, 0.0), axis=0
    ) / jnp.maximum(previous_transition_count, 1)
    following_transition = jnp.sum(
        jnp.where(following_endpoint, physical_limb, 0.0), axis=0
    ) / jnp.maximum(following_transition_count, 1)
    from_previous_transition = jnp.angle(
        physical_limb * jnp.conjugate(previous_transition[None, :])
    )
    to_following_transition = jnp.angle(
        following_transition[None, :] * jnp.conjugate(physical_limb)
    )
    isolated_turn = (
        isolated_endpoint
        & (previous_transition_count[None, :] == 2)
        & (following_transition_count[None, :] == 2)
        & (from_previous_transition * to_following_transition <= 0.0)
        & (jnp.abs(from_previous_transition) <= 0.5)
        & (jnp.abs(to_following_transition) <= 0.5)
    )
    # A one-sample branch needs an extra tangency only when its image angle
    # turns outside the angles of the two co-transitioning root pairs.  Same-
    # signed steps describe a monotone short branch and must not manufacture an
    # extremum. This is a local continuity certificate, independent of trace
    # density.
    flat_requires_tangency = (
        candidate_mask & (~physical_endpoint | isolated_turn)
    ).reshape(-1)
    # ``trace_binary_source_limb`` carries every quintic root forward as the
    # next solve's initial value. Opposite signed angular increments on one
    # continuously tracked branch therefore bracket a real support extremum,
    # even when the 2-D tangency Newton system is singular. Keep this as an
    # independent support certificate, guarded against root-slot jumps.
    previous_jump = jnp.abs(physical_limb - previous_limb)
    following_jump = jnp.abs(following_limb - physical_limb)
    image_scale = 1.0 + jnp.abs(physical_limb)
    branch_bracketed = (
        candidate_mask
        & ~physical_endpoint
        & previous_mask
        & following_mask
        & jnp.isfinite(previous_limb)
        & jnp.isfinite(physical_limb)
        & jnp.isfinite(following_limb)
        & (jnp.abs(previous_step) <= 0.5)
        & (jnp.abs(following_step) <= 0.5)
        & (previous_jump <= 0.5 * image_scale)
        & (following_jump <= 0.5 * image_scale)
    ).reshape(-1)
    indices = jnp.nonzero(flat_mask, size=maximum_extrema, fill_value=-1)[0]
    safe_indices = jnp.maximum(indices, 0)
    selected = fitted.reshape(-1)[safe_indices]
    selected = jnp.where(indices >= 0, selected, 2.0 * jnp.pi)
    selected_images = physical_limb.reshape(-1)[safe_indices]
    sampled_angles = jnp.mod(center, 2.0 * jnp.pi).reshape(-1)[safe_indices]
    sampled_angles = jnp.where(indices >= 0, sampled_angles, 2.0 * jnp.pi)
    requires_tangency = flat_requires_tangency[safe_indices] & (indices >= 0)
    bracketed = branch_bracketed[safe_indices] & (indices >= 0)

    selected, tangencies_valid = _refine_angular_tangencies(
        w_center,
        rho,
        s=s,
        q=q,
        initial_images=selected_images,
        initial_angles=sampled_angles,
        fallback_angles=selected,
        active=indices >= 0,
        required=requires_tangency,
        bracketed=bracketed,
    )
    n_extrema = jnp.sum(flat_mask, dtype=jnp.int32)
    topology_uncertain = topology_uncertain | (n_extrema > maximum_extrema)
    trace_finite = jnp.all(jnp.isfinite(physical_limb))
    topology_uncertain = topology_uncertain | ~trace_finite | (n_extrema == 0)
    # A malformed trace must not be downgraded to a soft topology warning:
    # otherwise the fallback full-circle cell could agree at two quadrature
    # orders and be certified as a zero/incorrect support.  Reuse the existing
    # support-validity channel so every angular caller fails closed.
    trace_support_valid = trace_finite & (n_extrema > 0)
    tangencies_valid = jnp.where(
        trace_support_valid,
        tangencies_valid,
        jnp.zeros_like(tangencies_valid),
    )
    endpoints = jnp.sort(
        jnp.concatenate(
            (
                jnp.asarray([0.0, 2.0 * jnp.pi], dtype=w_center.real.dtype),
                selected,
            )
        )
    )
    cells = jnp.stack((endpoints[:-1], endpoints[1:]), axis=-1)
    support_uncertain = topology_uncertain | ~jnp.all(tangencies_valid)
    active = _endpoint_preserving_union_mask(
        cells,
        physical_limb,
        physical_mask,
        w_center,
        rho,
        s=s,
        q=q,
        support_uncertain=support_uncertain,
        root_mode=root_mode,
        ordinate_bound=(
            _radial_image_bound(w_center, rho, s=s, q=q)
            if root_mode.startswith(("ea_fixed", "ea_auto"))
            else None
        ),
    )
    return AngularSupport(
        cells,
        active,
        topology_uncertain,
        minimum_ghost_residual,
        limb_topology,
        tangencies_valid,
    )


def _refine_angular_tangencies(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    initial_images: Array,
    initial_angles: Array,
    fallback_angles: Array,
    active: Array,
    required: Array,
    bracketed: Array,
) -> tuple[Array, Array]:
    """Project traced extrema onto exact radial level-set tangencies.

    An angular Gauss rule converges rapidly only when its cells terminate at
    the square-root singularities of the radial moment.  A sampled source limb
    locates those singularities, but even a dense trace leaves a phase error
    large enough for several Gauss orders to miss the same narrow fold.  The
    exact endpoints satisfy ``H(r, theta) = dH/dr = 0``; six damped Newton
    steps remove that trace-spacing error without densifying the support.

    Endpoint motion contributes no boundary term because the radial moment is
    zero at a tangency, so stopping its derivative keeps forward AD compact
    while preserving the differentiated integral.
    """

    lens = binary_geometry(s, q)
    real_dtype = w_center.real.dtype
    rho_scale = jnp.maximum(rho**2, jnp.finfo(real_dtype).tiny)

    def refine_one(
        image,
        initial_angle,
        fallback_angle,
        is_bracketed,
    ):
        initial_state = jnp.stack((jnp.abs(image), initial_angle))

        def level_set(radius, angle):
            image_point = radius * jnp.exp(1.0j * angle)
            mapped = (
                lens_eq(
                    image_point - lens.shifted,
                    nlenses=2,
                    a=lens.a,
                    e1=lens.e1,
                )
                + lens.shifted
            )
            return (jnp.abs(mapped - w_center) ** 2 - rho**2) / rho_scale

        radial_derivative = jax.grad(level_set, argnums=0)

        def equations(state):
            radius, angle = state
            return jnp.stack(
                (
                    level_set(radius, angle),
                    radial_derivative(radius, angle),
                )
            )

        equation_jacobian = jax.jacfwd(equations)

        def newton_step(_, state):
            residual = equations(state)
            jacobian = equation_jacobian(state)
            determinant = (
                jacobian[0, 0] * jacobian[1, 1]
                - jacobian[0, 1] * jacobian[1, 0]
            )
            safe_determinant = jnp.where(
                jnp.abs(determinant) > 128.0 * jnp.finfo(real_dtype).eps,
                determinant,
                1.0,
            )
            radial_step = (
                jacobian[1, 1] * residual[0]
                - jacobian[0, 1] * residual[1]
            ) / safe_determinant
            angular_step = (
                -jacobian[1, 0] * residual[0]
                + jacobian[0, 0] * residual[1]
            ) / safe_determinant
            radial_limit = 0.25 * jnp.maximum(jnp.abs(state[0]), 1.0e-3)
            step = jnp.stack(
                (
                    jnp.clip(radial_step, -radial_limit, radial_limit),
                    jnp.clip(angular_step, -0.25, 0.25),
                )
            )
            candidate = state - step
            candidate_valid = (
                jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(residual))
                & (jnp.abs(determinant) > 128.0 * jnp.finfo(real_dtype).eps)
                & (candidate[0] > 0.0)
            )
            return jnp.where(candidate_valid, candidate, state)

        refined = jax.lax.fori_loop(0, 6, newton_step, initial_state)
        final_residual = equations(refined)
        angular_displacement = jnp.abs(
            jnp.angle(jnp.exp(1.0j * (refined[1] - initial_angle)))
        )
        converged = (
            jnp.all(jnp.isfinite(final_residual))
            & (jnp.max(jnp.abs(final_residual)) <= 1.0e-7)
            & (angular_displacement <= 0.35)
            & (refined[0] > 0.0)
        )
        angle = jnp.where(converged, refined[1], fallback_angle)
        return jnp.mod(angle, 2.0 * jnp.pi), converged | is_bracketed

    # Compact extrema before Newton. A vectorized 24-slot solve made every
    # polar point pay for padding and branch endpoints; the sequential CPU
    # chart needs exact tangencies only at true, continuously tracked extrema.
    required = required & active
    n_required = jnp.sum(required, dtype=jnp.int32)
    indices = jnp.nonzero(
        required,
        size=required.size,
        fill_value=0,
    )[0]
    initial_refined = jnp.mod(fallback_angles, 2.0 * jnp.pi)
    initial_valid = ~required

    def refine_selected(compact_index, state):
        refined_angles, valid = state
        index = indices[compact_index]
        angle, is_valid = refine_one(
            initial_images[index],
            initial_angles[index],
            fallback_angles[index],
            bracketed[index],
        )
        return refined_angles.at[index].set(angle), valid.at[index].set(is_valid)

    refined, valid = jax.lax.fori_loop(
        jnp.int32(0),
        n_required,
        refine_selected,
        (initial_refined, initial_valid),
    )
    return jax.lax.stop_gradient(refined), valid


def _uniform_result_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_theta: int,
    cells: Array,
    active: Array,
    topology_uncertain: Array,
    minimum_ghost_residual: Array,
    limb_topology: Array,
    tangencies_valid: Array | bool = True,
    root_mode: str = "companion",
) -> AngularMomentResult:
    """Integrate one angular rule on an already constructed support."""

    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
    weights = jnp.asarray(weights, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)
    lower = cells[:, 0]
    width = cells[:, 1] - cells[:, 0]
    ordinate_bound = (
        _radial_image_bound(w_center, rho, s=s, q=q)
        if root_mode.startswith(("ea_fixed", "ea_auto"))
        else None
    )

    def integrate_cell(inputs):
        cell_lower, cell_width, cell_active = inputs

        def evaluate(_):
            theta = cell_lower + cell_width * jnp.sin(transform) ** 2
            angular_weights = (
                weights * 0.25 * jnp.pi * cell_width * jnp.sin(2.0 * transform)
            )
            coefficients = jax.vmap(
                lambda angle: binary_radial_level_set_coefficients(
                    angle, w_center, rho, s=s, q=q
                )
            )(theta)
            moments, invalid = _ray_moments(
                coefficients,
                root_mode=root_mode,
                ordinate_bound=ordinate_bound,
                stable_context=(theta, w_center, rho, s, q),
            )
            return jnp.sum(angular_weights * moments), jnp.sum(invalid, dtype=jnp.int32)

        return jax.lax.cond(
            cell_active,
            evaluate,
            lambda _: (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
            ),
            operand=None,
        )

    cell_areas, cell_invalid = jax.lax.map(
        integrate_cell,
        (lower, width, active),
    )
    area = jnp.sum(cell_areas)
    return AngularMomentResult(
        area / (jnp.pi * rho**2),
        jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
        jnp.int32(n_theta) * jnp.sum(active, dtype=jnp.int32),
        jnp.sum(cell_invalid, dtype=jnp.int32),
        minimum_ghost_residual / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        jnp.bitwise_or(
            jnp.bitwise_or(
                jnp.where(
                    jnp.any(~jnp.isfinite(cell_areas)), jnp.int32(1), jnp.int32(0)
                ),
                jnp.where(
                    topology_uncertain,
                    jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
                    jnp.int32(0),
                ),
            ),
            jnp.where(
                jnp.all(tangencies_valid),
                jnp.int32(0),
                jnp.int32(ANGULAR_MOMENT_SUPPORT),
            ),
        ),
    )


def _uniform_cell_areas_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_theta: int,
    cells: Array,
    active: Array,
    root_mode: str = "companion",
) -> tuple[Array, Array]:
    """Return per-cell areas and invalid-root counts for sparse refinement."""

    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
    weights = jnp.asarray(weights, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)
    ordinate_bound = (
        _radial_image_bound(w_center, rho, s=s, q=q)
        if root_mode.startswith(("ea_fixed", "ea_auto"))
        else None
    )

    def integrate_cell(inputs):
        bounds, cell_active = inputs

        def evaluate(_):
            width = bounds[1] - bounds[0]
            theta = bounds[0] + width * jnp.sin(transform) ** 2
            angular_weights = weights * 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
            coefficients = jax.vmap(
                lambda angle: binary_radial_level_set_coefficients(
                    angle, w_center, rho, s=s, q=q
                )
            )(theta)
            moments, invalid = _ray_moments(
                coefficients,
                root_mode=root_mode,
                ordinate_bound=ordinate_bound,
                stable_context=(theta, w_center, rho, s, q),
            )
            return jnp.sum(angular_weights * moments), jnp.sum(invalid, dtype=jnp.int32)

        return jax.lax.cond(
            cell_active,
            evaluate,
            lambda _: (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
            ),
            operand=None,
        )

    return jax.lax.map(integrate_cell, (cells, active))


def _uniform_result_from_support_flat(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_theta: int,
    cells: Array,
    active: Array,
    topology_uncertain: Array,
    minimum_ghost_residual: Array,
    limb_topology: Array,
    root_mode: str = "companion",
) -> AngularMomentResult:
    """A/B kernel that flattens all angular cells into one root batch."""

    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
    weights = jnp.asarray(weights, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)
    lower = cells[:, 0]
    width = jnp.where(active, cells[:, 1] - cells[:, 0], 0.0)
    theta = lower[:, None] + width[:, None] * jnp.sin(transform)[None, :] ** 2
    angular_weights = (
        weights[None, :]
        * 0.25
        * jnp.pi
        * width[:, None]
        * jnp.sin(2.0 * transform)[None, :]
    )
    ordinate_bound = (
        _radial_image_bound(w_center, rho, s=s, q=q)
        if root_mode.startswith(("ea_fixed", "ea_auto"))
        else None
    )
    flat_theta = theta.reshape(-1)
    coefficients = jax.vmap(
        lambda angle: binary_radial_level_set_coefficients(
            angle, w_center, rho, s=s, q=q
        )
    )(flat_theta)
    moments, invalid = _ray_moments(
        coefficients,
        root_mode=root_mode,
        ordinate_bound=ordinate_bound,
        stable_context=(flat_theta, w_center, rho, s, q),
    )
    moments = moments.reshape(theta.shape)
    invalid = invalid.reshape(theta.shape)
    cell_areas = jnp.sum(angular_weights * moments, axis=1)
    cell_invalid = jnp.sum(invalid, axis=1, dtype=jnp.int32)
    area = jnp.sum(jnp.where(active, cell_areas, 0.0))
    status = jnp.bitwise_or(
        jnp.where(
            jnp.any(~jnp.isfinite(jnp.where(active, cell_areas, 0.0))),
            jnp.int32(1),
            jnp.int32(0),
        ),
        jnp.where(
            topology_uncertain,
            jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
            jnp.int32(0),
        ),
    )
    return AngularMomentResult(
        area / (jnp.pi * rho**2),
        jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
        jnp.int32(n_theta) * jnp.sum(active, dtype=jnp.int32),
        jnp.sum(jnp.where(active, cell_invalid, 0), dtype=jnp.int32),
        minimum_ghost_residual / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        status,
    )


def _uniform_nested_cc_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    fine_intervals: int,
    cells: Array,
    active: Array,
    topology_uncertain: Array,
    minimum_ghost_residual: Array,
    limb_topology: Array,
    root_mode: str = "companion",
) -> tuple[AngularMomentResult, AngularMomentResult]:
    """Evaluate nested Clenshaw--Curtis fine/coarse angular moments once."""

    if fine_intervals < 4 or fine_intervals % 4:
        raise ValueError("fine_intervals must be a positive multiple of four")
    fine_nodes, fine_weights = _clenshaw_curtis_rule(fine_intervals)
    _, coarse_weights = _clenshaw_curtis_rule(fine_intervals // 2)
    # Endpoints have an exactly zero Jacobian after the sine-squared map.  Do
    # not solve their tangent radial polynomials; evaluate only the shared
    # interior fine nodes and gather the odd-indexed subset for the coarse
    # rule.
    transformed = 0.25 * np.pi * (fine_nodes[1:-1] + 1.0)
    transformed = jnp.asarray(transformed, dtype=w_center.real.dtype)
    fine_weights = jnp.asarray(fine_weights[1:-1], dtype=w_center.real.dtype)
    coarse_weights = jnp.asarray(coarse_weights[1:-1], dtype=w_center.real.dtype)
    coarse_indices = jnp.arange(1, fine_intervals - 1, 2, dtype=jnp.int32)
    ordinate_bound = (
        _radial_image_bound(w_center, rho, s=s, q=q)
        if root_mode.startswith(("ea_fixed", "ea_auto"))
        else None
    )

    def integrate_cell(inputs):
        bounds, cell_active = inputs

        def evaluate(_):
            width = bounds[1] - bounds[0]
            theta = bounds[0] + width * jnp.sin(transformed) ** 2
            jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * transformed)
            coefficients = jax.vmap(
                lambda angle: binary_radial_level_set_coefficients(
                    angle, w_center, rho, s=s, q=q
                )
            )(theta)
            moments, invalid = _ray_moments(
                coefficients,
                root_mode=root_mode,
                ordinate_bound=ordinate_bound,
                stable_context=(theta, w_center, rho, s, q),
            )
            transformed_moments = jacobian * moments
            fine_area = jnp.sum(fine_weights * transformed_moments)
            coarse_area = jnp.sum(coarse_weights * transformed_moments[coarse_indices])
            return (
                fine_area,
                coarse_area,
                jnp.sum(invalid, dtype=jnp.int32),
                jnp.sum(invalid[coarse_indices], dtype=jnp.int32),
            )

        return jax.lax.cond(
            cell_active,
            evaluate,
            lambda _: (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
                jnp.int32(0),
            ),
            operand=None,
        )

    fine_areas, coarse_areas, fine_invalid, coarse_invalid = jax.lax.map(
        integrate_cell, (cells, active)
    )
    normalization = jnp.pi * rho**2
    ghost_ratio = minimum_ghost_residual / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    base_status = jnp.where(
        topology_uncertain,
        jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
        jnp.int32(0),
    )

    def make_result(areas, invalid, interior_nodes):
        status = jnp.bitwise_or(
            base_status,
            jnp.where(jnp.any(~jnp.isfinite(areas)), jnp.int32(1), jnp.int32(0)),
        )
        return AngularMomentResult(
            jnp.sum(areas) / normalization,
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.int32(interior_nodes) * jnp.sum(active, dtype=jnp.int32),
            jnp.sum(invalid, dtype=jnp.int32),
            ghost_ratio,
            limb_topology,
            status,
        )

    return (
        make_result(fine_areas, fine_invalid, fine_intervals - 1),
        make_result(coarse_areas, coarse_invalid, fine_intervals // 2 - 1),
    )


def _uniform_gk15_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    cells: Array,
    active: Array,
    topology_uncertain: Array,
    minimum_ghost_residual: Array,
    limb_topology: Array,
    root_mode: str = "companion",
) -> tuple[AngularMomentResult, AngularMomentResult]:
    """Evaluate nested Kronrod-15 and Gauss-7 angular moments once."""

    nodes = jnp.asarray(GK15_X, dtype=w_center.real.dtype)
    fine_weights = jnp.asarray(GK15_W, dtype=w_center.real.dtype)
    coarse_weights = jnp.asarray(G7_W_ON_GK15, dtype=w_center.real.dtype)
    transformed = 0.25 * jnp.pi * (nodes + 1.0)
    ordinate_bound = (
        _radial_image_bound(w_center, rho, s=s, q=q)
        if root_mode.startswith(("ea_fixed", "ea_auto"))
        else None
    )

    def integrate_cell(inputs):
        bounds, cell_active = inputs

        def evaluate(_):
            width = bounds[1] - bounds[0]
            theta = bounds[0] + width * jnp.sin(transformed) ** 2
            jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * transformed)
            coefficients = jax.vmap(
                lambda angle: binary_radial_level_set_coefficients(
                    angle, w_center, rho, s=s, q=q
                )
            )(theta)
            moments, invalid = _ray_moments(
                coefficients,
                root_mode=root_mode,
                ordinate_bound=ordinate_bound,
                stable_context=(theta, w_center, rho, s, q),
            )
            transformed_moments = jacobian * moments
            return (
                jnp.sum(fine_weights * transformed_moments),
                jnp.sum(coarse_weights * transformed_moments),
                jnp.sum(invalid, dtype=jnp.int32),
                jnp.sum(
                    jnp.where(coarse_weights != 0.0, invalid, 0),
                    dtype=jnp.int32,
                ),
            )

        return jax.lax.cond(
            cell_active,
            evaluate,
            lambda _: (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
                jnp.int32(0),
            ),
            operand=None,
        )

    fine_areas, coarse_areas, fine_invalid, coarse_invalid = jax.lax.map(
        integrate_cell, (cells, active)
    )
    normalization = jnp.pi * rho**2
    ghost_ratio = minimum_ghost_residual / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    base_status = jnp.where(
        topology_uncertain,
        jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
        jnp.int32(0),
    )

    def make_result(areas, invalid, nodes_per_cell):
        status = jnp.bitwise_or(
            base_status,
            jnp.where(jnp.any(~jnp.isfinite(areas)), jnp.int32(1), jnp.int32(0)),
        )
        return AngularMomentResult(
            jnp.sum(areas) / normalization,
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.int32(nodes_per_cell) * jnp.sum(active, dtype=jnp.int32),
            jnp.sum(invalid, dtype=jnp.int32),
            ghost_ratio,
            limb_topology,
            status,
        )

    return (
        make_result(fine_areas, fine_invalid, 15),
        make_result(coarse_areas, coarse_invalid, 7),
    )


def mag_uniform_angular_moment_fixed(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_theta: int = 32,
    n_limb: int = 16,
    return_info: bool = False,
) -> Array | AngularMomentResult:
    """Evaluate one global-chart Gauss-Legendre angular-moment tier."""

    if n_theta <= 0:
        raise ValueError("n_theta must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    support = _angular_support_cells(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
    )
    result = _uniform_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=n_theta,
        cells=support.cells,
        active=support.active,
        topology_uncertain=support.topology_uncertain,
        minimum_ghost_residual=support.minimum_ghost_residual,
        limb_topology=support.limb_topology,
        tangencies_valid=support.tangencies_valid,
    )
    return result if return_info else result.magnification


def mag_uniform_angular_moment(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    return_info: bool = False,
) -> Array | AngularMomentResult:
    """Evaluate and certify the calibrated angle-first CPU hierarchy."""

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    # The source-limb trace locates angular support extrema; it is not an
    # accuracy quadrature.  Although 32 samples place regular-source values
    # within 7e-6 of VBBL, a stratified caustic sweep found topology misses.
    # Keep 128 here for topology safety; angular order is optimized separately.
    support = _angular_support_cells(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=128,
    )
    result = _mag_uniform_angular_moment_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        rtol=rtol,
        support=support,
    )
    return result if return_info else result.magnification


def _mag_uniform_angular_moment_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    rtol: Array,
    support: AngularSupport,
) -> AngularMomentResult:
    """Evaluate the primary angular hierarchy on precomputed support."""

    result, _, _ = _mag_uniform_angular_moment_workspace(
        w_center,
        rho,
        s=s,
        q=q,
        rtol=rtol,
        support=support,
    )
    return result


def _mag_uniform_angular_moment_workspace(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    rtol: Array,
    support: AngularSupport,
) -> tuple[AngularMomentResult, Array, Array]:
    """Return the primary result and reusable 6/4-point cell areas."""

    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    fine_areas, fine_invalid = _uniform_cell_areas_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=6,
        cells=cells,
        active=active,
    )
    normalization = jnp.pi * rho**2
    fine_status = jnp.bitwise_or(
        jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        jnp.where(
            jnp.all(tangencies_valid),
            jnp.int32(0),
            jnp.int32(ANGULAR_MOMENT_SUPPORT),
        ),
    )
    fine = AngularMomentResult(
        jnp.sum(fine_areas) / normalization,
        jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
        jnp.int32(6) * jnp.sum(active, dtype=jnp.int32),
        jnp.sum(fine_invalid, dtype=jnp.int32),
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        fine_status,
    )
    needs_coarse_check = (fine.status == 0) | (
        (fine.status == ANGULAR_MOMENT_TOPOLOGY) & (fine.ghost_residual_ratio > 5.0)
    )
    coarse_areas = jax.lax.cond(
        needs_coarse_check,
        lambda _: _uniform_cell_areas_from_support(
            w_center,
            rho,
            s=s,
            q=q,
            n_theta=4,
            cells=cells,
            active=active,
        )[0],
        lambda _: fine_areas,
        operand=None,
    )
    coarse_magnification = jnp.sum(coarse_areas) / normalization
    scale = jnp.maximum(jnp.abs(fine.magnification), 1.0)
    tier_difference = jnp.abs(fine.magnification - coarse_magnification)
    relative_difference = tier_difference / scale
    regular_certificate = (fine.status == 0) & (relative_difference <= 2.0 * rtol)
    topology_certificate = (
        (fine.status == ANGULAR_MOMENT_TOPOLOGY)
        & (fine.ghost_residual_ratio > 5.0)
        & (relative_difference <= 0.8 * rtol)
    )
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
    result = fine._replace(
        estimated_error=estimated_error,
        status=jnp.where(
            certified,
            jnp.int32(0),
            jnp.bitwise_or(fine.status, jnp.int32(ANGULAR_MOMENT_EXHAUSTED)),
        ),
    )
    return result, fine_areas, coarse_areas


def _mag_uniform_sparse_refined_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    rtol: Array,
    support: AngularSupport,
    fine_areas: Array,
    coarse_areas: Array,
    capacity: int = 2,
) -> AngularMomentResult:
    """Refine only the fixed number of largest-error angular cells."""

    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    scores = jnp.where(active, jnp.abs(fine_areas - coarse_areas), -jnp.inf)
    _, indices = jax.lax.top_k(jax.lax.stop_gradient(scores), capacity)
    selected_active = active[indices]
    high_areas, high_invalid = _uniform_cell_areas_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=12,
        cells=cells[indices],
        active=selected_active,
    )
    selected_fine = fine_areas[indices]
    selected_coarse = coarse_areas[indices]
    selected_low_error = jnp.sum(
        jnp.where(selected_active, jnp.abs(selected_fine - selected_coarse), 0.0)
    )
    total_low_error = jnp.sum(
        jnp.where(active, jnp.abs(fine_areas - coarse_areas), 0.0)
    )
    remaining_error = jnp.maximum(0.0, total_low_error - selected_low_error)
    refined_error = jnp.sum(
        jnp.where(selected_active, jnp.abs(high_areas - selected_fine), 0.0)
    )
    normalization = jnp.pi * rho**2
    area = jnp.sum(fine_areas) + jnp.sum(
        jnp.where(selected_active, high_areas - selected_fine, 0.0)
    )
    magnification = area / normalization
    scale = jnp.maximum(jnp.abs(magnification), 1.0)
    estimated_error = jnp.maximum(
        2.0 * (remaining_error + refined_error) / normalization,
        0.75 * rtol * scale,
    )
    structural_status = jnp.bitwise_or(
        jnp.where(jnp.any(high_invalid != 0), jnp.int32(1), jnp.int32(0)),
        jnp.where(
            jnp.all(tangencies_valid),
            jnp.int32(0),
            jnp.int32(ANGULAR_MOMENT_SUPPORT),
        ),
    )
    certified = (
        (structural_status == 0)
        & jnp.isfinite(magnification)
        & (estimated_error <= rtol * scale)
    )
    status = jnp.where(
        certified,
        jnp.int32(0),
        jnp.bitwise_or(
            jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
            jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
        ),
    )
    return AngularMomentResult(
        magnification,
        estimated_error,
        jnp.int32(6) * jnp.sum(active, dtype=jnp.int32)
        + jnp.int32(12) * jnp.sum(selected_active, dtype=jnp.int32),
        jnp.sum(high_invalid, dtype=jnp.int32),
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        status,
    )


def _uniform_high_order_refinement(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    rtol: Array,
    support: AngularSupport,
    preceding_n_theta: Array,
) -> AngularMomentResult:
    """Certify a difficult fitted-endpoint support with a 24/32 pair."""

    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    # Near a small-q resonant caustic the angular moment can converge
    # non-monotonically at low order.  In addition, an isolated one-sample
    # limb branch can leave only a fitted (rather than Newton-refined)
    # tangency endpoint.  A lazy, substantially higher 24/32 pair resolves
    # the square-root onset without retracing the source limb.
    medium = _uniform_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=24,
        cells=cells,
        active=active,
        topology_uncertain=topology,
        minimum_ghost_residual=ghost,
        limb_topology=limb_topology,
        tangencies_valid=tangencies_valid,
    )
    high = _uniform_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=32,
        cells=cells,
        active=active,
        topology_uncertain=topology,
        minimum_ghost_residual=ghost,
        limb_topology=limb_topology,
        tangencies_valid=tangencies_valid,
    )
    high_scale = jnp.maximum(jnp.abs(high.magnification), 1.0)
    high_difference = jnp.abs(high.magnification - medium.magnification)
    high_error = jnp.maximum(
        2.0 * high_difference,
        0.75 * rtol * high_scale,
    )
    # Topology uncertainty is expected on a caustic crossing.  A failed exact
    # tangency refinement is also soft here because both high-order rules
    # integrate the same fitted endpoint with an endpoint-clustering
    # transform.  Root/non-finite failures remain hard errors.
    soft_status = jnp.bitwise_or(
        jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
        jnp.int32(ANGULAR_MOMENT_SUPPORT),
    )
    hard_status = jnp.bitwise_and(
        high.status,
        jnp.bitwise_not(soft_status),
    )
    # At most one short-lived fold pair may rely on fitted endpoints.  More
    # unverified tangencies indicate globally unreliable support and must
    # remain fail-closed even if two quadrature orders happen to agree.
    unverified_tangencies = jnp.sum(
        ~tangencies_valid,
        dtype=jnp.int32,
    )
    certified = (
        (hard_status == 0)
        & (unverified_tangencies <= 2)
        & (medium.invalid_root_count == 0)
        & (high.invalid_root_count == 0)
        & jnp.isfinite(medium.magnification)
        & jnp.isfinite(high.magnification)
        & jnp.isfinite(high_error)
        & (high_error <= rtol * high_scale)
    )
    return high._replace(
        estimated_error=high_error,
        n_theta=preceding_n_theta + medium.n_theta + high.n_theta,
        status=jnp.where(
            certified,
            jnp.int32(0),
            jnp.bitwise_or(
                high.status,
                jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
            ),
        ),
    )


def _uniform_contact_refinement(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    rtol: Array,
    support: AngularSupport,
    capacity: int = 2,
    root_mode: str = "companion",
) -> AngularMomentResult:
    """Refine only the angular cells responsible for origin contact."""

    if capacity <= 0:
        raise ValueError("contact refinement capacity must be positive")
    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    low_areas, low_invalid = _uniform_cell_areas_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=8,
        cells=cells,
        active=active,
        root_mode=root_mode,
    )
    medium_areas, medium_invalid = _uniform_cell_areas_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=12,
        cells=cells,
        active=active,
        root_mode=root_mode,
    )
    low_difference = jnp.abs(medium_areas - low_areas)
    scores = jnp.where(active, low_difference, -jnp.inf)
    _, indices = jax.lax.top_k(jax.lax.stop_gradient(scores), capacity)
    selected_active = active[indices]
    high24_areas, high24_invalid = _uniform_cell_areas_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=24,
        cells=cells[indices],
        active=selected_active,
        root_mode=root_mode,
    )
    high32_areas, high32_invalid = _uniform_cell_areas_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=32,
        cells=cells[indices],
        active=selected_active,
        root_mode=root_mode,
    )
    selected_medium = medium_areas[indices]
    area = jnp.sum(jnp.where(active, medium_areas, 0.0)) + jnp.sum(
        jnp.where(selected_active, high32_areas - selected_medium, 0.0)
    )
    normalization = jnp.pi * rho**2
    magnification = area / normalization
    selected_low_error = jnp.sum(
        jnp.where(selected_active, low_difference[indices], 0.0)
    )
    total_low_error = jnp.sum(jnp.where(active, low_difference, 0.0))
    remaining_low_error = jnp.maximum(0.0, total_low_error - selected_low_error)
    selected_high_error = jnp.sum(
        jnp.where(
            selected_active,
            jnp.abs(high32_areas - high24_areas),
            0.0,
        )
    )
    scale = jnp.maximum(jnp.abs(magnification), 1.0)
    estimated_error = jnp.maximum(
        2.0 * (remaining_low_error + selected_high_error) / normalization,
        0.75 * rtol * scale,
    )
    invalid_root_count = (
        jnp.sum(jnp.where(active, low_invalid + medium_invalid, 0), dtype=jnp.int32)
        + jnp.sum(
            jnp.where(
                selected_active,
                high24_invalid + high32_invalid,
                0,
            ),
            dtype=jnp.int32,
        )
    )
    unverified_tangencies = jnp.sum(~tangencies_valid, dtype=jnp.int32)
    finite = (
        jnp.isfinite(magnification)
        & jnp.isfinite(estimated_error)
        & jnp.all(jnp.isfinite(jnp.where(active, medium_areas, 0.0)))
        & jnp.all(
            jnp.isfinite(jnp.where(selected_active, high32_areas, 0.0))
        )
    )
    certified = (
        finite
        & (invalid_root_count == 0)
        & (unverified_tangencies <= 2)
        & (estimated_error <= rtol * scale)
    )
    failed_status = jnp.bitwise_or(
        jnp.where(
            topology,
            jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
            jnp.int32(0),
        ),
        jnp.where(
            unverified_tangencies <= 2,
            jnp.int32(0),
            jnp.int32(ANGULAR_MOMENT_SUPPORT),
        ),
    )
    failed_status = jnp.bitwise_or(
        failed_status,
        jnp.int32(ANGULAR_MOMENT_EXHAUSTED),
    )
    return AngularMomentResult(
        magnification,
        estimated_error,
        (
            jnp.int32(8 + 12) * jnp.sum(active, dtype=jnp.int32)
            + jnp.int32(24 + 32) * jnp.sum(selected_active, dtype=jnp.int32)
        ),
        invalid_root_count,
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        jnp.where(certified, jnp.int32(0), failed_status),
    )


def mag_uniform_angular_moment_compact(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    _support: AngularSupport | None = None,
    return_info: bool = False,
) -> Array | AngularMomentResult:
    """Cross-certify a compact polar hierarchy, normally with an 8/12 pair.

    This is the independent chart used after Cartesian projections expose a
    directional singularity.  The optional support is deliberately private:
    the CPU scheduler supplies support reconstructed from its existing limb
    trace, so chart arbitration does not repeat boundary root solves.
    """

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    support = (
        _angular_support_cells(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=128,
        )
        if _support is None
        else _support
    )
    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    coarse = _uniform_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=8,
        cells=cells,
        active=active,
        topology_uncertain=topology,
        minimum_ghost_residual=ghost,
        limb_topology=limb_topology,
        tangencies_valid=tangencies_valid,
    )
    fine = _uniform_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=12,
        cells=cells,
        active=active,
        topology_uncertain=topology,
        minimum_ghost_residual=ghost,
        limb_topology=limb_topology,
        tangencies_valid=tangencies_valid,
    )
    scale = jnp.maximum(jnp.abs(fine.magnification), 1.0)
    difference = jnp.abs(fine.magnification - coarse.magnification)
    structural_status = jnp.bitwise_and(
        fine.status,
        jnp.bitwise_not(jnp.int32(ANGULAR_MOMENT_TOPOLOGY)),
    )
    estimated_error = jnp.maximum(
        2.0 * difference,
        0.75 * rtol * scale,
    )
    certified = (
        (structural_status == 0)
        & (coarse.invalid_root_count == 0)
        & (fine.invalid_root_count == 0)
        & jnp.isfinite(fine.magnification)
        & jnp.isfinite(estimated_error)
        & (estimated_error <= rtol * scale)
    )
    result = fine._replace(
        estimated_error=estimated_error,
        status=jnp.where(
            certified,
            jnp.int32(0),
            jnp.bitwise_or(fine.status, jnp.int32(ANGULAR_MOMENT_EXHAUSTED)),
        ),
    )

    def refine_difficult_point(_):
        return _uniform_high_order_refinement(
            w_center,
            rho,
            s=s,
            q=q,
            rtol=rtol,
            support=support,
            preceding_n_theta=coarse.n_theta + fine.n_theta,
        )

    result = jax.lax.cond(
        certified,
        lambda _: result,
        refine_difficult_point,
        operand=None,
    )
    return result if return_info else result.magnification


def mag_uniform_angular_moment_refined(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    coarse_magnification: Array | None = None,
    _support: AngularSupport | None = None,
    return_info: bool = False,
) -> Array | AngularMomentResult:
    """Resolve sharp angular peaks in the refined polar coordinate chart."""

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    support = (
        _angular_support_cells(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=128,
        )
        if _support is None
        else _support
    )
    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    medium = _uniform_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_theta=12,
        cells=cells,
        active=active,
        topology_uncertain=topology,
        minimum_ghost_residual=ghost,
        limb_topology=limb_topology,
        tangencies_valid=tangencies_valid,
    )
    if coarse_magnification is None:
        coarse_magnification = _uniform_result_from_support(
            w_center,
            rho,
            s=s,
            q=q,
            n_theta=8,
            cells=cells,
            active=active,
            topology_uncertain=topology,
            minimum_ghost_residual=ghost,
            limb_topology=limb_topology,
            tangencies_valid=tangencies_valid,
        ).magnification
    medium_scale = jnp.maximum(jnp.abs(medium.magnification), 1.0)
    medium_difference = jnp.abs(medium.magnification - coarse_magnification)
    medium_structural = jnp.bitwise_and(
        medium.status,
        jnp.bitwise_not(jnp.int32(ANGULAR_MOMENT_TOPOLOGY)),
    )
    medium_error = jnp.maximum(
        2.0 * medium_difference,
        0.75 * rtol * medium_scale,
    )
    medium_certified = (
        (medium_structural == 0)
        & (medium.invalid_root_count == 0)
        & jnp.isfinite(medium.magnification)
        & jnp.isfinite(medium_error)
        & (medium_error <= rtol * medium_scale)
    )

    def use_medium(_):
        return medium._replace(estimated_error=medium_error, status=jnp.int32(0))

    def use_high(_):
        high = _uniform_result_from_support(
            w_center,
            rho,
            s=s,
            q=q,
            n_theta=16,
            cells=cells,
            active=active,
            topology_uncertain=topology,
            minimum_ghost_residual=ghost,
            limb_topology=limb_topology,
            tangencies_valid=tangencies_valid,
        )
        scale = jnp.maximum(jnp.abs(high.magnification), 1.0)
        tier_difference = jnp.abs(high.magnification - medium.magnification)
        structural_status = jnp.bitwise_and(
            high.status,
            jnp.bitwise_not(jnp.int32(ANGULAR_MOMENT_TOPOLOGY)),
        )
        estimated_error = jnp.maximum(2.0 * tier_difference, 0.75 * rtol * scale)
        certified = (
            (structural_status == 0)
            & (high.invalid_root_count == 0)
            & jnp.isfinite(high.magnification)
            & jnp.isfinite(estimated_error)
            & (estimated_error <= rtol * scale)
        )
        high_result = high._replace(
            estimated_error=estimated_error,
            status=jnp.where(
                certified,
                jnp.int32(0),
                jnp.bitwise_or(high.status, jnp.int32(ANGULAR_MOMENT_EXHAUSTED)),
            ),
        )
        return jax.lax.cond(
            certified,
            lambda _: high_result,
            lambda _: _uniform_high_order_refinement(
                w_center,
                rho,
                s=s,
                q=q,
                rtol=rtol,
                support=support,
                preceding_n_theta=medium.n_theta + high.n_theta,
            ),
            operand=None,
        )

    result = jax.lax.cond(medium_certified, use_medium, use_high, operand=None)
    return result if return_info else result.magnification


__all__ = [
    "AngularMomentResult",
    "ANGULAR_MOMENT_TOPOLOGY",
    "ANGULAR_MOMENT_EXHAUSTED",
    "binary_radial_level_set_coefficients",
    "mag_uniform_angular_moment",
    "mag_uniform_angular_moment_compact",
    "mag_uniform_angular_moment_fixed",
    "mag_uniform_angular_moment_refined",
]
