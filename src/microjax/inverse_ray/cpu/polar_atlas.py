"""Topology seeds from the two-dimensional source-limb image trace.

The radial-first CPU chart used to project every traced image branch onto a
one-dimensional interval and then rediscover all angular boundary roots from
scratch.  A polar branch atlas retains the missing information: each adjacent
pair of limb images supplies a short segment in ``(r, theta)``.  Intersections
of those segments with a requested radius are inexpensive angular seeds.  A
few Newton steps on the exact Fourier level set remove the interpolation error,
so the polyline is a topology guide rather than an approximation to the area.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..geometry.lens import binary_geometry
from ..roots.angular import (
    evaluate_fourier,
    evaluate_fourier_derivative,
    evaluate_fourier_second_derivative,
)
from ..roots.level_set import binary_level_set_fourier
from .support import tracked_limb_neighbors

Array = jnp.ndarray

POLAR_ATLAS_OK = 0
POLAR_ATLAS_CAPACITY = 1
POLAR_ATLAS_ROOT_FAILURE = 2
POLAR_ATLAS_TOPOLOGY = 4
_MAX_BINARY_ANGULAR_ROOTS = 6
_TURNING_CAPACITY = 32


class PolarBranchAtlas(NamedTuple):
    """Fixed-shape polar segments of all continuously tracked image limbs."""

    radius_start: Array
    radius_end: Array
    angle_start: Array
    angle_step: Array
    active: Array
    cap_radius_sample: Array
    cap_radius_vertex: Array
    cap_angle_center: Array
    cap_angle_linear: Array
    cap_angle_quadratic: Array
    cap_phase_offset: Array
    cap_radial_quadratic: Array
    cap_angle_vertex: Array
    cap_source_index: Array
    cap_active: Array
    status: Array


class PolarAtlasMeasure(NamedTuple):
    """Exact angular measure seeded by a polar branch atlas."""

    measure: Array
    angles: Array
    active: Array
    n_crossings: Array
    maximum_residual: Array
    status: Array


def build_polar_branch_atlas(
    image_limb: Array,
    physical_mask: Array,
) -> PolarBranchAtlas:
    """Retain both polar coordinates of every valid source-limb segment.

    ``trace_binary_source_limb`` already continues the five algebraic roots
    around the source.  Only the closing edge needs the explicit monodromy
    assignment performed by :func:`tracked_limb_neighbors`.  Segments with an
    unphysical endpoint are not extrapolated through a fold birth; those are
    reported later as a topology mismatch instead of being silently invented.
    """

    if image_limb.ndim != 2 or physical_mask.shape != image_limb.shape:
        raise ValueError("image_limb and physical_mask must have the same 2D shape")
    previous, following, previous_mask, following_mask = tracked_limb_neighbors(
        image_limb, physical_mask
    )
    radius_start = jnp.abs(image_limb)
    radius_end = jnp.abs(following)
    angle_start = jnp.angle(image_limb)
    angle_step = jnp.angle(following * jnp.conjugate(image_limb))
    finite = (
        jnp.isfinite(radius_start)
        & jnp.isfinite(radius_end)
        & jnp.isfinite(angle_start)
        & jnp.isfinite(angle_step)
    )
    active = physical_mask & following_mask & finite

    # At a source-limb/caustic crossing two physical image branches meet at
    # one critical point.  The two samples closest to that point live in
    # different root slots, so ordinary same-slot continuation leaves the
    # traced preimage boundary open.  Join the pair on the physical side of
    # each 3<->5 transition.  The bridge is only an angular seed: evaluation
    # and area still use the exact Fourier level set.
    disappearing = physical_mask & ~following_mask
    appearing = ~physical_mask & following_mask
    n_disappearing = jnp.sum(disappearing, axis=0, dtype=jnp.int32)
    n_appearing = jnp.sum(appearing, axis=0, dtype=jnp.int32)
    use_disappearing = n_disappearing == 2
    transition_mask = jnp.where(use_disappearing[None, :], disappearing, appearing)
    transition_points = jnp.where(use_disappearing[None, :], image_limb, following)
    pair_order = jnp.argsort(~transition_mask, axis=0)[:2]
    pair = jnp.take_along_axis(transition_points, pair_order, axis=0)
    bridge_start = pair[0]
    bridge_end = pair[1]
    bridge_radius_start = jnp.abs(bridge_start)
    bridge_radius_end = jnp.abs(bridge_end)
    bridge_angle_start = jnp.angle(bridge_start)
    bridge_angle_step = jnp.angle(bridge_end * jnp.conjugate(bridge_start))
    bridge_active = (
        (use_disappearing | (n_appearing == 2))
        & jnp.isfinite(bridge_radius_start)
        & jnp.isfinite(bridge_radius_end)
        & jnp.isfinite(bridge_angle_start)
        & jnp.isfinite(bridge_angle_step)
    )

    # The radial topology places a sampled extremum at the vertex of its local
    # three-point parabola.  Linear atlas segments stop at the sampled value
    # and would leave the thin interval between that value and the fitted
    # vertex empty.  Store an analytic two-sided cap only for that interval.
    radius_previous = jnp.abs(previous)
    radial_left = radius_start - radius_previous
    radial_right = radius_end - radius_start
    radial_scale = jnp.maximum(
        jnp.max(jnp.where(physical_mask, radius_start, 0.0)), 1.0
    )
    slope_floor = jnp.sqrt(jnp.finfo(radius_start.dtype).eps) * radial_scale
    turning = (
        physical_mask
        & previous_mask
        & following_mask
        & (radial_left * radial_right <= 0.0)
        & ((jnp.abs(radial_left) + jnp.abs(radial_right)) > slope_floor)
    )
    radial_curvature = radius_previous - 2.0 * radius_start + radius_end
    safe_curvature = jnp.where(
        jnp.abs(radial_curvature) > slope_floor,
        radial_curvature,
        1.0,
    )
    phase_offset = 0.5 * (radius_previous - radius_end) / safe_curvature
    cap_active = turning & (jnp.abs(phase_offset) <= 1.0)
    radius_vertex = radius_start + 0.25 * (radius_end - radius_previous) * phase_offset
    previous_step = jnp.angle(image_limb * jnp.conjugate(previous))
    following_step = jnp.angle(following * jnp.conjugate(image_limb))
    angle_linear = 0.5 * (previous_step + following_step)
    angle_quadratic = 0.5 * (following_step - previous_step)
    angle_vertex = (
        angle_start + angle_linear * phase_offset + angle_quadratic * phase_offset**2
    )
    cap_finite = (
        jnp.isfinite(radius_vertex)
        & jnp.isfinite(phase_offset)
        & jnp.isfinite(angle_linear)
        & jnp.isfinite(angle_quadratic)
        & jnp.isfinite(angle_vertex)
    )
    cap_active = cap_active & cap_finite
    flat_cap_active = cap_active.reshape(-1)
    n_caps = jnp.sum(flat_cap_active, dtype=jnp.int32)
    cap_indices = jnp.nonzero(
        flat_cap_active,
        size=_TURNING_CAPACITY,
        fill_value=0,
    )[0]
    compact_cap_active = jnp.arange(_TURNING_CAPACITY, dtype=jnp.int32) < jnp.minimum(
        n_caps, _TURNING_CAPACITY
    )
    atlas_status = jnp.where(
        n_caps <= _TURNING_CAPACITY,
        jnp.int32(POLAR_ATLAS_OK),
        jnp.int32(POLAR_ATLAS_CAPACITY),
    )
    return PolarBranchAtlas(
        jnp.concatenate((radius_start.reshape(-1), bridge_radius_start)),
        jnp.concatenate((radius_end.reshape(-1), bridge_radius_end)),
        jnp.concatenate((angle_start.reshape(-1), bridge_angle_start)),
        jnp.concatenate((angle_step.reshape(-1), bridge_angle_step)),
        jnp.concatenate((active.reshape(-1), bridge_active)),
        radius_start.reshape(-1)[cap_indices],
        radius_vertex.reshape(-1)[cap_indices],
        angle_start.reshape(-1)[cap_indices],
        angle_linear.reshape(-1)[cap_indices],
        angle_quadratic.reshape(-1)[cap_indices],
        phase_offset.reshape(-1)[cap_indices],
        (0.5 * radial_curvature).reshape(-1)[cap_indices],
        angle_vertex.reshape(-1)[cap_indices],
        cap_indices,
        compact_cap_active,
        atlas_status,
    )


def polar_atlas_turning_radius_override(
    atlas: PolarBranchAtlas,
    image_limb: Array,
) -> Array:
    """Return exact cap radii in the original limb-sample layout."""

    output = jnp.full(image_limb.size, jnp.nan, dtype=image_limb.real.dtype)
    n_active = jnp.sum(atlas.cap_active, dtype=jnp.int32)

    def insert(index, values):
        return values.at[atlas.cap_source_index[index]].set(
            atlas.cap_radius_vertex[index]
        )

    output = jax.lax.fori_loop(jnp.int32(0), n_active, insert, output)
    return output.reshape(image_limb.shape)


def refine_polar_atlas_tangencies(
    atlas: PolarBranchAtlas,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> PolarBranchAtlas:
    """Project sampled turning caps onto ``H = dH/dtheta = 0``.

    Source-limb samples locate the relevant extrema.  The exact binary Fourier
    level set then determines their radii and angles without increasing limb
    sampling.  These refined radii can be reused as radial-cell breakpoints,
    keeping support construction and angular seeding geometrically identical.
    """

    lens = binary_geometry(s, q)
    real_dtype = w_center.real.dtype
    eps = jnp.finfo(real_dtype).eps

    def equations(state):
        radius, angle = state
        fourier = binary_level_set_fourier(
            radius,
            w_center - lens.shifted,
            rho,
            lens.shifted,
            a=lens.a,
            e1=lens.e1,
        )
        return jnp.stack(
            (
                evaluate_fourier(fourier.coefficients, angle),
                evaluate_fourier_derivative(fourier.coefficients, angle),
            )
        )

    equation_jacobian = jax.jacfwd(equations)

    def refine_one(radius, angle, sample_radius):
        initial = jnp.stack((radius, angle))
        radial_scale = jnp.maximum(
            4.0 * jnp.abs(sample_radius - radius),
            1024.0 * eps * jnp.maximum(jnp.abs(radius), 1.0),
        )

        def newton_step(_, state):
            residual = equations(state)
            jacobian = equation_jacobian(state)
            determinant = (
                jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
            )
            safe = jnp.abs(determinant) > 128.0 * eps
            safe_determinant = jnp.where(safe, determinant, 1.0)
            radial_step = (
                jacobian[1, 1] * residual[0] - jacobian[0, 1] * residual[1]
            ) / safe_determinant
            angular_step = (
                -jacobian[1, 0] * residual[0] + jacobian[0, 0] * residual[1]
            ) / safe_determinant
            step = jnp.stack(
                (
                    jnp.clip(radial_step, -radial_scale, radial_scale),
                    jnp.clip(angular_step, -0.25, 0.25),
                )
            )
            candidate = state - step
            candidate_ok = (
                safe
                & jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(residual))
                & (candidate[0] > 0.0)
            )
            return jnp.where(candidate_ok, candidate, state)

        refined = jax.lax.fori_loop(0, 6, newton_step, initial)
        residual = equations(refined)
        displacement = jnp.abs(jnp.angle(jnp.exp(1.0j * (refined[1] - angle))))
        converged = (
            jnp.all(jnp.isfinite(residual))
            & (jnp.max(jnp.abs(residual)) <= 1.0e-7)
            & (displacement <= 0.35)
            & (jnp.abs(refined[0] - radius) <= radial_scale)
            & (refined[0] > 0.0)
        )
        return refined, converged

    n_active = jnp.sum(atlas.cap_active, dtype=jnp.int32)

    def refine_selected(index, state):
        radii, angles, valid = state
        refined, converged = refine_one(
            radii[index], angles[index], atlas.cap_radius_sample[index]
        )
        return (
            radii.at[index].set(jnp.where(converged, refined[0], radii[index])),
            angles.at[index].set(jnp.where(converged, refined[1], angles[index])),
            valid.at[index].set(converged),
        )

    refined_radii, refined_angles, valid = jax.lax.fori_loop(
        jnp.int32(0),
        n_active,
        refine_selected,
        (
            atlas.cap_radius_vertex,
            atlas.cap_angle_vertex,
            ~atlas.cap_active,
        ),
    )
    all_valid = jnp.all(valid | ~atlas.cap_active)
    status = jnp.bitwise_or(
        atlas.status,
        jnp.where(
            all_valid,
            jnp.int32(POLAR_ATLAS_OK),
            jnp.int32(POLAR_ATLAS_ROOT_FAILURE),
        ),
    )
    return atlas._replace(
        cap_radius_vertex=jax.lax.stop_gradient(refined_radii),
        cap_angle_vertex=jax.lax.stop_gradient(refined_angles),
        status=status,
    )


def _inside_measure(coefficients: Array, angles: Array, n_angles: Array) -> Array:
    """Classify the intervals between sorted exact boundary angles."""

    dtype = angles.dtype
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
    active = jnp.arange(angles.size, dtype=jnp.int32) < n_angles
    sorted_angles = jnp.sort(jnp.where(active, jnp.mod(angles, two_pi), two_pi))
    boundaries = jnp.concatenate(
        (jnp.zeros(1, dtype=dtype), sorted_angles, two_pi[None])
    )
    lower = boundaries[:-1]
    upper = boundaries[1:]
    width = jnp.maximum(upper - lower, 0.0)
    inside = evaluate_fourier(coefficients, 0.5 * (lower + upper)) <= 0.0
    return jnp.sum(jnp.where(inside, width, 0.0)), sorted_angles, active


def polar_atlas_angular_measure(
    atlas: PolarBranchAtlas,
    coefficients: Array,
    padding: Array,
    degenerate: Array,
    radius: Array,
) -> PolarAtlasMeasure:
    """Intersect an atlas with one ring and polish the resulting angles.

    The half-open crossing convention assigns a sampled vertex to exactly one
    adjacent segment.  For a binary level set there can be at most six exact
    angular boundary roots.  More polyline crossings therefore indicate an
    unresolved sampled turn and are reported as capacity/topology failure.
    """

    r0 = atlas.radius_start
    r1 = atlas.radius_end
    increasing = (r0 <= radius) & (radius < r1)
    decreasing = (r1 <= radius) & (radius < r0)
    crossing = atlas.active & (increasing | decreasing)
    linear_denominator = r1 - r0
    safe_linear_denominator = jnp.where(
        jnp.abs(linear_denominator) > jnp.finfo(radius.dtype).tiny,
        linear_denominator,
        1.0,
    )
    linear_fraction = jnp.clip((radius - r0) / safe_linear_denominator, 0.0, 1.0)
    linear_angles = atlas.angle_start + linear_fraction * atlas.angle_step

    cap_lo = jnp.minimum(atlas.cap_radius_sample, atlas.cap_radius_vertex)
    cap_hi = jnp.maximum(atlas.cap_radius_sample, atlas.cap_radius_vertex)
    cap_crossing = atlas.cap_active & (cap_lo <= radius) & (radius < cap_hi)
    safe_cap_quadratic = jnp.where(
        jnp.abs(atlas.cap_radial_quadratic) > jnp.finfo(radius.dtype).tiny,
        atlas.cap_radial_quadratic,
        1.0,
    )
    cap_displacement = jnp.sqrt(
        jnp.maximum(
            (radius - atlas.cap_radius_vertex) / safe_cap_quadratic,
            0.0,
        )
    )
    cap_phase_minus = atlas.cap_phase_offset - cap_displacement
    cap_phase_plus = atlas.cap_phase_offset + cap_displacement

    def cap_angle(phase):
        approximate = (
            atlas.cap_angle_center
            + atlas.cap_angle_linear * phase
            + atlas.cap_angle_quadratic * phase**2
        )
        return (
            approximate
            + atlas.cap_angle_vertex
            - (
                atlas.cap_angle_center
                + atlas.cap_angle_linear * atlas.cap_phase_offset
                + atlas.cap_angle_quadratic * atlas.cap_phase_offset**2
            )
        )

    candidate_angles = jnp.concatenate(
        (linear_angles, cap_angle(cap_phase_minus), cap_angle(cap_phase_plus))
    )
    candidate_mask = jnp.concatenate((crossing, cap_crossing, cap_crossing))
    n_crossings = jnp.sum(candidate_mask, dtype=jnp.int32)
    indices = jnp.nonzero(
        candidate_mask,
        size=_MAX_BINARY_ANGULAR_ROOTS,
        fill_value=0,
    )[0]
    selected = jnp.arange(_MAX_BINARY_ANGULAR_ROOTS, dtype=jnp.int32) < jnp.minimum(
        n_crossings, _MAX_BINARY_ANGULAR_ROOTS
    )
    angles = candidate_angles[indices]

    eps = jnp.finfo(radius.dtype).eps

    def polish(_, current):
        value = evaluate_fourier(coefficients, current)
        derivative = evaluate_fourier_derivative(coefficients, current)
        second_derivative = evaluate_fourier_second_derivative(
            coefficients, current
        )
        newton_safe = jnp.abs(derivative) > 16.0 * eps
        newton_step = jnp.where(newton_safe, value / derivative, 0.0)
        halley_denominator = (
            2.0 * derivative**2 - value * second_derivative
        )
        halley_safe = jnp.abs(halley_denominator) > 32.0 * eps
        halley_step = jnp.where(
            halley_safe,
            2.0 * value * derivative / halley_denominator,
            newton_step,
        )
        step = jnp.where(newton_safe, halley_step, 0.0)
        step = jnp.clip(step, -0.35, 0.35)
        candidates = jnp.stack(
            (current, current - step, current - 0.5 * step, current - 0.25 * step)
        )
        candidate_residual = jnp.abs(evaluate_fourier(coefficients, candidates))
        best = jnp.argmin(candidate_residual, axis=0)
        return jnp.take_along_axis(candidates, best[None, :], axis=0)[0]

    angles = jax.lax.fori_loop(0, 6, polish, angles)
    residual = jnp.abs(evaluate_fourier(coefficients, angles))
    residual_tolerance = padding + jnp.maximum(
        8192.0 * eps,
        jnp.asarray(1.0e-11, dtype=radius.dtype),
    )
    selected_residual_ok = jnp.all(~selected | (residual <= residual_tolerance))
    finite = jnp.all(~selected | jnp.isfinite(angles))
    even = (n_crossings % 2) == 0
    within_capacity = n_crossings <= _MAX_BINARY_ANGULAR_ROOTS

    measure, sorted_angles, sorted_active = _inside_measure(
        coefficients, angles, jnp.minimum(n_crossings, _MAX_BINARY_ANGULAR_ROOTS)
    )
    # A low-cost sign audit catches a missed narrow component whenever one of
    # its sign changes is sampled.  It supplements, but does not replace, the
    # source-limb crossing-count certificate.
    audit_angles = (
        2.0
        * jnp.pi
        * jnp.arange(32, dtype=radius.dtype)
        / jnp.asarray(32.0, dtype=radius.dtype)
    )
    audit_inside = evaluate_fourier(coefficients, audit_angles) <= 0.0
    sign_changes = jnp.sum(audit_inside != jnp.roll(audit_inside, -1), dtype=jnp.int32)
    represented = jnp.minimum(n_crossings, _MAX_BINARY_ANGULAR_ROOTS)
    topology_ok = sign_changes <= represented
    status = jnp.bitwise_or(
        atlas.status,
        jnp.bitwise_or(
            jnp.where(
                within_capacity,
                jnp.int32(POLAR_ATLAS_OK),
                jnp.int32(POLAR_ATLAS_CAPACITY),
            ),
            jnp.where(
                finite & selected_residual_ok & ~degenerate & even,
                jnp.int32(POLAR_ATLAS_OK),
                jnp.int32(POLAR_ATLAS_ROOT_FAILURE),
            ),
        ),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            topology_ok,
            jnp.int32(POLAR_ATLAS_OK),
            jnp.int32(POLAR_ATLAS_TOPOLOGY),
        ),
    )
    maximum_residual = jnp.max(jnp.where(selected, residual, 0.0))
    return PolarAtlasMeasure(
        measure,
        sorted_angles,
        sorted_active,
        n_crossings,
        maximum_residual,
        status,
    )


__all__ = [
    "POLAR_ATLAS_CAPACITY",
    "POLAR_ATLAS_OK",
    "POLAR_ATLAS_ROOT_FAILURE",
    "POLAR_ATLAS_TOPOLOGY",
    "PolarAtlasMeasure",
    "PolarBranchAtlas",
    "build_polar_branch_atlas",
    "polar_atlas_angular_measure",
    "polar_atlas_turning_radius_override",
    "refine_polar_atlas_tangencies",
]
