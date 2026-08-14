"""Uniform polar-support construction for the radial CPU ICRS.

This module deliberately has one geometric rule for every binary-lens state:

1. trace the circular source limb once;
2. differentiate each physical image branch along the limb;
3. project smooth radial extrema onto ``H = H_theta = 0``;
4. retain each traced branch's sampled radial minimum and maximum as a
   conservative bracket; and
5. split at those extrema and at physical branch endpoints.

The limb angle is therefore retained, but it is used only to locate radial
contacts.  Angular boundary roots on an integration ring are still solved from
the exact degree-three Fourier level set.  There are no fold bridges, turning
caps, root-slot-specific rules, or result-dependent rescue passes here.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from microjax.point_source import lens_eq

from ..geometry.lens import binary_geometry
from ..geometry.topology import RADIAL_CAPACITY, RADIAL_OK, RADIAL_TOPOLOGY
from ..roots.level_set import (
    binary_level_set_fourier,
)
from .support import tracked_limb_neighbors

Array = jnp.ndarray

_INTERVAL_CAPACITY = 128
# Five tracked binary-lens branches contribute two branch extrema each; the
# remaining slots cover local turns and physical segment endpoints.  Overflow
# is reported rather than silently truncated.
_TANGENCY_CAPACITY = 32
_BASE_INTERVAL_CAPACITY = 48


class SimplePolarTopology(NamedTuple):
    """Fixed-shape radial cells from the uniform limb/contact construction."""

    intervals: Array
    n_intervals: Array
    status: Array
    n_tangencies: Array


class LimbTangencies(NamedTuple):
    """Radial contacts seeded by sign changes along the traced source limb."""

    radius: Array
    angle: Array
    active: Array
    residual: Array
    capacity_exceeded: Array
    maximum_motion_ratio: Array


def _refine_limb_endpoints(
    image_limb: Array,
    endpoint_mask: Array,
    radial_step: Array,
    angular_step: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array]:
    """Project caustic branch endpoints onto ``H=0`` and ``det J=0``."""

    lens = binary_geometry(s, q)
    real_dtype = image_limb.real.dtype
    eps = jnp.finfo(real_dtype).eps
    level_scale = jnp.maximum(rho**2, jnp.finfo(real_dtype).tiny)

    def equations(state):
        current_radius, current_angle = state
        image = current_radius * jnp.exp(1.0j * current_angle)
        midpoint = image - lens.shifted
        mapped = (
            lens_eq(midpoint, nlenses=2, a=lens.a, e1=lens.e1)
            + lens.shifted
        )
        difference = mapped - w_center
        shear = (
            lens.e1 / (jnp.conjugate(midpoint) - lens.a) ** 2
            + (1.0 - lens.e1) / (jnp.conjugate(midpoint) + lens.a) ** 2
        )
        level = (
            jnp.real(difference * jnp.conjugate(difference)) - rho**2
        ) / level_scale
        shear_abs = jnp.abs(shear)
        determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
        return jnp.stack((level, determinant))

    equation_jacobian = jax.jacfwd(equations)

    def project_one(initial_radius, initial_angle, dr, dtheta, active):
        initial = jnp.stack((initial_radius, initial_angle))
        radial_limit = jnp.maximum(
            4.0 * dr,
            4096.0 * eps * jnp.maximum(initial_radius, 1.0),
        )
        angular_limit = jnp.maximum(4.0 * dtheta, 1.0e-3)

        def step(_, state):
            residual = equations(state)
            jacobian = equation_jacobian(state)
            determinant = (
                jacobian[0, 0] * jacobian[1, 1]
                - jacobian[0, 1] * jacobian[1, 0]
            )
            safe = jnp.abs(determinant) > 128.0 * eps
            safe_determinant = jnp.where(safe, determinant, 1.0)
            delta = jnp.stack(
                (
                    (
                        jacobian[1, 1] * residual[0]
                        - jacobian[0, 1] * residual[1]
                    )
                    / safe_determinant,
                    (
                        -jacobian[1, 0] * residual[0]
                        + jacobian[0, 0] * residual[1]
                    )
                    / safe_determinant,
                )
            )
            delta = delta.at[0].set(
                jnp.clip(delta[0], -radial_limit, radial_limit)
            )
            delta = delta.at[1].set(
                jnp.clip(delta[1], -angular_limit, angular_limit)
            )
            candidate = state - delta
            valid = (
                active
                & safe
                & (candidate[0] > 0.0)
                & jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(residual))
            )
            return jnp.where(valid, candidate, state)

        refined = jax.lax.fori_loop(0, 18, step, initial)
        residual = jnp.max(jnp.abs(equations(refined)))
        converged = (
            active
            & jnp.isfinite(residual)
            & (residual <= 2.0e-11)
            & (jnp.abs(refined[0] - initial_radius) <= 4.0 * radial_limit)
            & (refined[0] > 0.0)
        )
        return refined[0], converged

    flat_mask = endpoint_mask.reshape(-1)
    n_endpoints = jnp.sum(flat_mask, dtype=jnp.int32)
    indices = jnp.nonzero(
        flat_mask, size=_TANGENCY_CAPACITY, fill_value=0
    )[0]
    selected = (
        jnp.arange(_TANGENCY_CAPACITY, dtype=jnp.int32)
        < jnp.minimum(n_endpoints, _TANGENCY_CAPACITY)
    )
    flat_image = image_limb.reshape(-1)[indices]
    refined_radius, converged = jax.vmap(project_one)(
        jnp.abs(flat_image),
        jnp.angle(flat_image),
        radial_step.reshape(-1)[indices],
        angular_step.reshape(-1)[indices],
        selected,
    )
    endpoint_radius = jnp.where(converged, refined_radius, jnp.abs(flat_image))
    return jax.lax.stop_gradient(endpoint_radius), selected


def _fourier(
    radius: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
):
    lens = binary_geometry(s, q)
    return binary_level_set_fourier(
        radius,
        w_center - lens.shifted,
        rho,
        lens.shifted,
        a=lens.a,
        e1=lens.e1,
    )


def refine_limb_tangencies(
    image_limb: Array,
    physical_mask: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    phases: Array | None = None,
) -> LimbTangencies:
    """Project sampled limb phases onto the radial-contact equations.

    Sign changes in the exact implicit radial derivative seed the projection.
    Every selected point then uses the identical two-equation projection; no
    fold pairing, cap model, or root-slot-specific rule is introduced.  A
    caustic branch endpoint is not a smooth tangency and is handled separately
    by :func:`build_simple_polar_topology`.
    """

    previous, following, previous_mask, following_mask = tracked_limb_neighbors(
        image_limb, physical_mask
    )
    radius = jnp.abs(image_limb)
    previous_radius = jnp.abs(previous)
    following_radius = jnp.abs(following)
    previous_angle_step = jnp.abs(jnp.angle(image_limb * jnp.conjugate(previous)))
    following_angle_step = jnp.abs(jnp.angle(following * jnp.conjugate(image_limb)))
    radial_neighbor_step = jnp.maximum(
        jnp.where(previous_mask, jnp.abs(radius - previous_radius), 0.0),
        jnp.where(following_mask, jnp.abs(following_radius - radius), 0.0),
    )
    angular_neighbor_step = jnp.maximum(
        jnp.where(previous_mask, previous_angle_step, 0.0),
        jnp.where(following_mask, following_angle_step, 0.0),
    )
    real_dtype = radius.dtype
    eps = jnp.finfo(real_dtype).eps

    lens = binary_geometry(s, q)
    level_scale = jnp.maximum(rho**2, jnp.finfo(real_dtype).tiny)

    # Differentiate the lens equation along the uniformly parameterised
    # source limb.  A radial extremum is a zero of this derivative; using its
    # sign change avoids inferring contacts from finite radial differences.
    if phases is None:
        phases = (
            2.0
            * jnp.pi
            * jnp.arange(image_limb.shape[1], dtype=real_dtype)
            / image_limb.shape[1]
        )
    else:
        phases = jnp.asarray(phases, dtype=real_dtype)
        if phases.ndim != 1 or phases.shape[0] != image_limb.shape[1]:
            raise ValueError("phases must match the traced source limb")
    source_phase_derivative = 1.0j * rho * jnp.exp(1.0j * phases)
    image_midpoint = image_limb - lens.shifted
    shear = (
        lens.e1 / (jnp.conjugate(image_midpoint) - lens.a) ** 2
        + (1.0 - lens.e1)
        / (jnp.conjugate(image_midpoint) + lens.a) ** 2
    )
    shear_abs = jnp.abs(shear)
    determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
    derivative_safe = jnp.abs(determinant) > jnp.sqrt(eps)
    image_phase_derivative = (
        source_phase_derivative[None, :]
        - shear * jnp.conjugate(source_phase_derivative)[None, :]
    ) / jnp.where(derivative_safe, determinant, 1.0)
    radial_phase_derivative = jnp.real(
        jnp.conjugate(image_limb) * image_phase_derivative
    ) / jnp.maximum(radius, jnp.finfo(real_dtype).tiny)
    radial_phase_derivative = jnp.where(
        physical_mask & derivative_safe,
        radial_phase_derivative,
        jnp.nan,
    )
    following_phase = jnp.roll(phases, -1).at[-1].set(2.0 * jnp.pi)
    phase_step = following_phase - phases
    # A source-limb trace is not a reliable support certificate when the
    # implicit first-order image displacement crosses more than two local
    # image scales in a single phase interval.  The factor two retains smooth
    # high-magnification arcs whose first-order predictor is conservative.  The
    # two-stage trace deliberately
    # moves a sample toward this singular motion, turning a hidden near-cusp
    # bias into a fail-closed topology status without a retry or a lens-specific
    # parameter gate.
    motion_ratio = (
        jnp.abs(image_phase_derivative)
        * phase_step[None, :]
        / jnp.maximum(radius, 1.0)
    )
    maximum_motion_ratio = jnp.max(
        jnp.where(physical_mask & derivative_safe, motion_ratio, 0.0)
    )

    def equations(state):
        current_radius, current_angle = state
        image = current_radius * jnp.exp(1.0j * current_angle)
        image_midpoint = image - lens.shifted
        mapped = (
            lens_eq(
                image_midpoint,
                nlenses=2,
                a=lens.a,
                e1=lens.e1,
            )
            + lens.shifted
        )
        difference = mapped - w_center
        shear = (
            lens.e1 / (jnp.conjugate(image_midpoint) - lens.a) ** 2
            + (1.0 - lens.e1)
            / (jnp.conjugate(image_midpoint) + lens.a) ** 2
        )
        image_theta = 1.0j * image
        mapped_theta = image_theta + shear * jnp.conjugate(image_theta)
        level = (
            jnp.real(difference * jnp.conjugate(difference)) - rho**2
        ) / level_scale
        level_theta = (
            2.0 * jnp.real(jnp.conjugate(difference) * mapped_theta)
        ) / level_scale
        return jnp.stack((level, level_theta))

    equation_jacobian = jax.jacfwd(equations)

    def project_one(initial_radius, initial_angle, radial_step, angular_step, active):
        initial = jnp.stack((initial_radius, initial_angle))
        radial_limit = jnp.maximum(
            4.0 * radial_step,
            4096.0 * eps * jnp.maximum(initial_radius, 1.0),
        )
        angular_limit = jnp.maximum(4.0 * angular_step, 1.0e-3)

        def newton_step(_, state):
            residual = equations(state)
            jacobian = equation_jacobian(state)
            determinant = (
                jacobian[0, 0] * jacobian[1, 1]
                - jacobian[0, 1] * jacobian[1, 0]
            )
            safe = jnp.abs(determinant) > 128.0 * eps
            safe_determinant = jnp.where(safe, determinant, 1.0)
            radial_delta = (
                jacobian[1, 1] * residual[0]
                - jacobian[0, 1] * residual[1]
            ) / safe_determinant
            angular_delta = (
                -jacobian[1, 0] * residual[0]
                + jacobian[0, 0] * residual[1]
            ) / safe_determinant
            radial_delta = jnp.clip(radial_delta, -radial_limit, radial_limit)
            angular_delta = jnp.clip(angular_delta, -angular_limit, angular_limit)
            candidate = state - jnp.stack((radial_delta, angular_delta))
            candidate_ok = (
                active
                & safe
                & (candidate[0] > 0.0)
                & jnp.all(jnp.isfinite(candidate))
                & jnp.all(jnp.isfinite(residual))
            )
            return jnp.where(candidate_ok, candidate, state)

        refined = jax.lax.fori_loop(0, 18, newton_step, initial)
        residual = equations(refined)
        radial_displacement = jnp.abs(refined[0] - initial_radius)
        angular_displacement = jnp.abs(
            jnp.angle(jnp.exp(1.0j * (refined[1] - initial_angle)))
        )
        residual_norm = jnp.max(jnp.abs(residual))
        converged = (
            active
            & jnp.all(jnp.isfinite(refined))
            & jnp.isfinite(residual_norm)
            & (residual_norm <= 2.0e-11)
            & (radial_displacement <= 4.0 * radial_limit)
            & (angular_displacement <= 4.0 * angular_limit)
            & (refined[0] > 0.0)
        )
        return refined, converged, residual_norm
    following_radial_derivative = jnp.roll(radial_phase_derivative, -1, axis=1)
    radial_scale = jnp.maximum(
        jnp.max(jnp.where(physical_mask, radius, 0.0)),
        1.0,
    )
    motion_floor = jnp.sqrt(eps) * radial_scale
    sampled_turning = (
        physical_mask
        & following_mask
        & jnp.isfinite(radial_phase_derivative)
        & jnp.isfinite(following_radial_derivative)
        & (radial_phase_derivative * following_radial_derivative <= 0.0)
        & (
            jnp.abs(radial_phase_derivative)
            + jnp.abs(following_radial_derivative)
            > motion_floor
        )
    )
    seed_mask = physical_mask & sampled_turning
    following_image = jnp.roll(image_limb, -1, axis=1)
    choose_following = (
        jnp.abs(following_radial_derivative)
        < jnp.abs(radial_phase_derivative)
    )
    seed_image = jnp.where(choose_following, following_image, image_limb)
    seed_radius = jnp.abs(seed_image)
    seed_angle = jnp.angle(seed_image)
    seed_radial_step = jnp.maximum(
        radial_neighbor_step,
        jnp.roll(radial_neighbor_step, -1, axis=1),
    )
    seed_angular_step = jnp.maximum(
        angular_neighbor_step,
        jnp.roll(angular_neighbor_step, -1, axis=1),
    )
    n_seeds = jnp.sum(seed_mask, dtype=jnp.int32)
    seed_indices = jnp.nonzero(
        seed_mask.reshape(-1),
        size=_TANGENCY_CAPACITY,
        fill_value=0,
    )[0]
    selected = (
        jnp.arange(_TANGENCY_CAPACITY, dtype=jnp.int32)
        < jnp.minimum(n_seeds, _TANGENCY_CAPACITY)
    )
    flat_radius = seed_radius.reshape(-1)[seed_indices]
    flat_angle = seed_angle.reshape(-1)[seed_indices]
    refined, converged, residual = jax.vmap(project_one)(
        flat_radius,
        flat_angle,
        seed_radial_step.reshape(-1)[seed_indices],
        seed_angular_step.reshape(-1)[seed_indices],
        selected,
    )
    return LimbTangencies(
        jax.lax.stop_gradient(refined[:, 0]),
        jax.lax.stop_gradient(refined[:, 1]),
        jax.lax.stop_gradient(converged),
        residual,
        n_seeds > _TANGENCY_CAPACITY,
        jax.lax.stop_gradient(maximum_motion_ratio),
    )


def build_simple_polar_topology(
    image_limb: Array,
    physical_mask: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    origin_inside: Array = False,
    phases: Array | None = None,
) -> SimplePolarTopology:
    """Build conservative radial cells without topology-specific cases."""

    previous, following, previous_mask, following_mask = tracked_limb_neighbors(
        image_limb, physical_mask
    )
    radii = jnp.abs(image_limb)
    previous_radii = jnp.abs(previous)
    following_radii = jnp.abs(following)
    valid = physical_mask & jnp.isfinite(radii)
    branch_active = jnp.any(valid, axis=1)
    branch_minimum = jnp.min(jnp.where(valid, radii, jnp.inf), axis=1)
    branch_maximum = jnp.max(jnp.where(valid, radii, -jnp.inf), axis=1)
    previous_step = jnp.where(
        valid & previous_mask,
        jnp.abs(radii - previous_radii),
        0.0,
    )
    following_step = jnp.where(
        valid & following_mask,
        jnp.abs(following_radii - radii),
        0.0,
    )
    previous_angle_step = jnp.where(
        valid & previous_mask,
        jnp.abs(jnp.angle(image_limb * jnp.conjugate(previous))),
        0.0,
    )
    following_angle_step = jnp.where(
        valid & following_mask,
        jnp.abs(jnp.angle(following * jnp.conjugate(image_limb))),
        0.0,
    )
    # A fold endpoint can lie between the last physical and first ghost limb
    # samples.  For square-root motion the unsampled remainder is below
    # 1/(sqrt(2)-1) times the preceding equal-phase step; four steps retain a
    # simple conservative margin without identifying the fold explicitly.
    branch_margin = 4.0 * jnp.max(
        jnp.maximum(previous_step, following_step), axis=1
    )
    branch_margin = jnp.maximum(branch_margin, 0.25 * rho)
    lower = jnp.where(
        branch_active,
        jnp.maximum(branch_minimum - branch_margin, 0.0),
        0.0,
    )
    upper = jnp.where(
        branch_active,
        branch_maximum + branch_margin,
        0.0,
    )
    innermost = jnp.argmin(jnp.where(branch_active, lower, jnp.inf))
    lower = lower.at[innermost].set(
        jnp.where(origin_inside & jnp.any(branch_active), 0.0, lower[innermost])
    )
    # Do not union root tracks independently.  When the source encloses a lens,
    # the inner and outer boundaries of one filled image annulus can live in
    # different algebraic slots; a branchwise union would delete the annular
    # interior between them.  One global conservative envelope cannot make that
    # mistake.  The extrema and physical branch endpoints below provide the
    # only internal breakpoints.
    global_lower = jnp.min(jnp.where(branch_active, lower, jnp.inf))
    global_upper = jnp.max(jnp.where(branch_active, upper, -jnp.inf))
    any_branch = jnp.any(branch_active)
    regions = jnp.zeros((lower.size, 2), dtype=radii.dtype).at[0].set(
        jnp.where(
            any_branch,
            jnp.stack((global_lower, global_upper)),
            jnp.zeros((2,), dtype=radii.dtype),
        )
    )
    n_regions = jnp.where(any_branch, jnp.int32(1), jnp.int32(0))

    tangencies = refine_limb_tangencies(
        image_limb,
        physical_mask,
        w_center,
        rho,
        s=s,
        q=q,
        phases=phases,
    )
    tangent_radius = tangencies.radius
    tangent_active = tangencies.active
    region_active = jnp.arange(regions.shape[0], dtype=jnp.int32) < n_regions
    tangent_covered = jnp.any(
        tangent_active[:, None]
        & region_active[None, :]
        & (tangent_radius[:, None] >= regions[None, :, 0])
        & (tangent_radius[:, None] <= regions[None, :, 1]),
        axis=1,
    )
    tangent_active = tangent_active & tangent_covered

    # A physical branch can begin or end where the source limb crosses a
    # caustic.  Such a singular endpoint need not satisfy the smooth tangency
    # equations, so retain its radius directly instead of projecting it onto a
    # nearby extremum.
    endpoint_mask = valid & ~(previous_mask & following_mask)
    refined_endpoint_radius, refined_endpoint_active = _refine_limb_endpoints(
        image_limb,
        endpoint_mask,
        jnp.maximum(previous_step, following_step),
        jnp.maximum(previous_angle_step, following_angle_step),
        w_center,
        rho,
        s=s,
        q=q,
    )
    sampled_endpoint_radius = radii.reshape(-1)
    sampled_endpoint_active = endpoint_mask.reshape(-1)


    values = jnp.concatenate(
        (
            tangent_radius,
            refined_endpoint_radius,
            sampled_endpoint_radius,
            branch_minimum,
            branch_maximum,
            regions.reshape(-1),
        )
    )
    values_active = jnp.concatenate(
        (
            tangent_active,
            refined_endpoint_active,
            sampled_endpoint_active,
            branch_active,
            branch_active,
            jnp.repeat(region_active, 2),
        )
    )
    sorted_values = jnp.sort(jnp.where(values_active, values, jnp.inf))
    finite = jnp.isfinite(sorted_values)
    previous_value = jnp.concatenate((jnp.asarray([-jnp.inf]), sorted_values[:-1]))
    # Near a very low-mass lens, many starting samples converge to the same
    # ill-conditioned contact with a radial spread of a few 1e-9.  Contacts
    # closer than the square-root machine scale cannot be distinguished
    # reliably by the subsequent angular polynomial either, so coalesce them
    # once here instead of creating dozens of numerically empty cells.
    unique_tolerance = jnp.maximum(
        jnp.sqrt(jnp.finfo(radii.dtype).eps)
        * jnp.maximum(jnp.abs(sorted_values), 1.0),
        1.0e-4 * rho,
    )
    unique = finite & ((sorted_values - previous_value) > unique_tolerance)
    unique_indices = jnp.nonzero(
        unique,
        size=values.size,
        fill_value=values.size - 1,
    )[0]
    n_unique = jnp.sum(unique, dtype=jnp.int32)
    unique_values = sorted_values[unique_indices]
    candidate_bounds = jnp.stack((unique_values[:-1], unique_values[1:]), axis=1)
    candidate_slots = jnp.arange(candidate_bounds.shape[0], dtype=jnp.int32)
    candidate_active = (
        (candidate_slots + 1 < n_unique)
        & (candidate_bounds[:, 1] > candidate_bounds[:, 0])
    )
    midpoint = 0.5 * (candidate_bounds[:, 0] + candidate_bounds[:, 1])
    covered = jnp.any(
        region_active[None, :]
        & (midpoint[:, None] >= regions[None, :, 0])
        & (midpoint[:, None] <= regions[None, :, 1]),
        axis=1,
    )
    candidate_active = candidate_active & covered
    n_base_raw = jnp.sum(candidate_active, dtype=jnp.int32)
    base_indices = jnp.nonzero(
        candidate_active,
        size=_BASE_INTERVAL_CAPACITY,
        fill_value=0,
    )[0]
    base_bounds = candidate_bounds[base_indices]
    n_intervals = jnp.minimum(n_base_raw, _INTERVAL_CAPACITY)
    intervals = jnp.zeros(
        (_INTERVAL_CAPACITY, 2), dtype=base_bounds.dtype
    ).at[:_BASE_INTERVAL_CAPACITY].set(base_bounds)
    intervals = jnp.where(
        (jnp.arange(_INTERVAL_CAPACITY, dtype=jnp.int32) < n_intervals)[:, None],
        intervals,
        0.0,
    )
    overflow = (
        tangencies.capacity_exceeded
        | (n_base_raw > _BASE_INTERVAL_CAPACITY)
    )
    status = jnp.where(
        overflow,
        jnp.int32(RADIAL_CAPACITY),
        jnp.int32(RADIAL_OK),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            tangencies.maximum_motion_ratio > 2.0,
            jnp.int32(RADIAL_TOPOLOGY),
            jnp.int32(RADIAL_OK),
        ),
    )
    # The radial chart must not certify an empty/partial algebraic trace as a
    # zero-area source.  Ghost roots are part of the quintic support
    # certificate too, so any non-finite slot (or no finite physical branch)
    # is a hard topology warning even when the surviving branchwise brackets
    # look numerically well behaved.
    trace_valid = jnp.all(jnp.isfinite(image_limb)) & any_branch
    status = jnp.bitwise_or(
        status,
        jnp.where(trace_valid, jnp.int32(RADIAL_OK), jnp.int32(RADIAL_TOPOLOGY)),
    )
    return SimplePolarTopology(
        intervals,
        n_intervals,
        status,
        jnp.sum(tangent_active, dtype=jnp.int32),
    )


__all__ = [
    "LimbTangencies",
    "SimplePolarTopology",
    "build_simple_polar_topology",
    "refine_limb_tangencies",
]
