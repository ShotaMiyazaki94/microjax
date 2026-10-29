"""Linear limb-darkening radial-slice kernel for binary CPU ICRS."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax, vmap

from ..geometry.lens import binary_geometry
from ..geometry.mapping import distance_from_source
from ..roots.angular import angular_intervals_binary_roots
from .angular_profile import integrate_mu_residual
from .quadrature import GkIntegral, integrate_gk15
from .sentinel import hidden_caustic_candidate
from .support import PolarOccupancyAtlas, build_radial_support, unwrap_support_1d
from .uniform import CPU_TIER_EXHAUSTED, CpuMagnificationResult

Array = jnp.ndarray


def _active_angular_union(
    atlas: PolarOccupancyAtlas,
    radius: Array,
) -> tuple[Array, Array]:
    """Return a fixed six-slot linearized union of circular atlas arcs."""

    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=radius.dtype)
    radial_active = (
        atlas.active
        & (radius >= atlas.radial_bounds[:, 0])
        & (radius <= atlas.radial_bounds[:, 1])
    )
    span = atlas.angular_bounds[:, 1] - atlas.angular_bounds[:, 0]
    lower = jnp.mod(atlas.angular_bounds[:, 0], two_pi)
    upper = lower + span
    first = jnp.stack((lower, jnp.minimum(upper, two_pi)), axis=1)
    second = jnp.stack((jnp.zeros_like(lower), upper - two_pi), axis=1)
    first_active = radial_active & (first[:, 1] > first[:, 0])
    second_active = radial_active & (second[:, 1] > second[:, 0])
    intervals = jnp.concatenate((first, second), axis=0)
    active = jnp.concatenate((first_active, second_active), axis=0)
    order = jnp.argsort(jnp.where(active, intervals[:, 0], jnp.inf))
    intervals = intervals[order]
    active = active[order]
    output = jnp.zeros_like(intervals)

    def merge_step(carry, item):
        merged, count, current_lower, current_upper, has_current = carry
        bounds, is_active = item
        overlaps = is_active & has_current & (bounds[0] <= current_upper)
        starts = is_active & ~has_current
        separates = is_active & has_current & ~overlaps
        merged = lax.cond(
            separates,
            lambda values: values.at[count].set(
                jnp.stack((current_lower, current_upper))
            ),
            lambda values: values,
            merged,
        )
        count = count + separates.astype(jnp.int32)
        current_lower = jnp.where(starts | separates, bounds[0], current_lower)
        current_upper = jnp.where(
            overlaps,
            jnp.maximum(current_upper, bounds[1]),
            jnp.where(starts | separates, bounds[1], current_upper),
        )
        has_current = has_current | is_active
        return (merged, count, current_lower, current_upper, has_current), None

    initial = (
        output,
        jnp.int32(0),
        jnp.asarray(0.0, dtype=radius.dtype),
        jnp.asarray(0.0, dtype=radius.dtype),
        jnp.asarray(False),
    )
    (output, count, current_lower, current_upper, has_current), _ = lax.scan(
        merge_step,
        initial,
        (intervals, active),
    )
    output = lax.cond(
        has_current,
        lambda values: values.at[count].set(
            jnp.stack((current_lower, current_upper))
        ),
        lambda values: values,
        output,
    )
    count = count + has_current.astype(jnp.int32)
    # Five circular image arcs can become six linear intervals when the
    # component crossing zero is represented on [0, 2pi].
    return output[:6], jnp.arange(6, dtype=jnp.int32) < jnp.minimum(count, 6)


def _filter_intervals_by_atlas(intervals, atlas, radius):
    """Keep full-circle inside intervals covered by the trace atlas."""

    sectors, sector_active = _active_angular_union(atlas, radius)
    interval_active = (
        jnp.arange(intervals.intervals.shape[0], dtype=jnp.int32)
        < intervals.n_intervals
    )
    midpoint = jnp.mod(
        0.5 * (intervals.intervals[:, 0] + intervals.intervals[:, 1]),
        2.0 * jnp.pi,
    )
    covered = jnp.any(
        sector_active[None, :]
        & (midpoint[:, None] >= sectors[None, :, 0])
        & (midpoint[:, None] <= sectors[None, :, 1]),
        axis=1,
    )
    retained = interval_active & covered
    order = jnp.argsort(~retained)
    return intervals._replace(
        intervals=intervals.intervals[order],
        n_intervals=jnp.sum(retained, dtype=jnp.int32),
    )


def _mag_limb_dark_cpu_fixed_impl(
    w_center,
    rho,
    *,
    s,
    q,
    u1,
    n_limb,
    radial_splits,
    include_ghost_support,
    support=None,
    occupancy_atlas: PolarOccupancyAtlas | None = None,
    radial_integrator=integrate_gk15,
    moment_integrator=None,
):
    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    rho = jnp.asarray(rho, dtype=real_dtype)
    u1 = jnp.asarray(u1, dtype=real_dtype)
    lens = binary_geometry(s, q)
    if support is None:
        support = build_radial_support(
            w_center,
            rho,
            s=lens.s,
            q=lens.q,
            n_limb=n_limb,
            include_all_roots=include_ghost_support,
        )
    support = unwrap_support_1d(support)
    physical_counts = jnp.sum(support.image_mask, axis=0)
    limb_transition = jnp.any(physical_counts != physical_counts[0])
    w_midpoint = w_center - lens.shifted
    eps = jnp.finfo(real_dtype).eps
    intensity_normalization = 3.0 / (jnp.pi * (3.0 - u1))

    def radial_integrand(radius):
        intervals = angular_intervals_binary_roots(
            radius,
            0.0,
            2.0 * jnp.pi,
            w_midpoint,
            rho,
            lens.shifted,
            64.0 * eps,
            a=lens.a,
            e1=lens.e1,
            robust_roots=True,
        )
        if occupancy_atlas is not None:
            intervals = _filter_intervals_by_atlas(
                intervals,
                occupancy_atlas,
                radius,
            )
        interval_active = (
            jnp.arange(intervals.intervals.shape[0]) < intervals.n_intervals
        )
        widths = intervals.intervals[:, 1] - intervals.intervals[:, 0]
        angular_measure = jnp.sum(jnp.where(interval_active, widths, 0.0))

        def mu(theta):
            distance = distance_from_source(
                radius,
                theta,
                w_midpoint,
                lens.shifted,
                nlenses=2,
                a=lens.a,
                e1=lens.e1,
            )
            normalized = distance / rho
            radicand = jnp.maximum(
                0.0,
                (1.0 - normalized) * (1.0 + normalized),
            )
            # Clipped samples carry no brightness or tangent contribution.
            # Keep sqrt away from zero so their JVP cannot form 0 * infinity;
            # testing equality also preserves NaN inputs in the primal value.
            is_zero = radicand == 0.0
            safe_radicand = jnp.where(is_zero, 1.0, radicand)
            return jnp.where(is_zero, 0.0, jnp.sqrt(safe_radicand))
        residual = integrate_mu_residual(mu, intervals)
        if moment_integrator is None:
            angular_value = intensity_normalization * (
                (1.0 - u1) * angular_measure + u1 * residual.value
            )
            angular_error = intensity_normalization * (
                jnp.abs(1.0 - u1) * intervals.error
                + jnp.abs(u1) * residual.error
            )
            status = residual.status
            return radius * angular_value, jnp.abs(radius) * angular_error, status

        return (
            radius * jnp.stack((angular_measure, residual.value)),
            jnp.abs(radius) * jnp.stack((intervals.error, residual.error)),
            residual.status,
        )

    if moment_integrator is None:
        radial = radial_integrator(
            radial_integrand,
            support.intervals,
            support.active,
            subdivisions=radial_splits,
        )
    else:
        moments = moment_integrator(
            vmap(radial_integrand), support.intervals, support.active,
        )
        radial = GkIntegral(
            intensity_normalization * (
                (1.0 - u1) * moments.value[0] + u1 * moments.value[1]
            ),
            intensity_normalization * (
                jnp.abs(1.0 - u1) * moments.error[0] + jnp.abs(u1) * moments.error[1]
            ),
            moments.status,
            moments.n_nodes,
        )
    support_valid = jnp.all(jnp.isfinite(support.image_limb)) & jnp.any(
        support.active
    )
    radial_status = jnp.bitwise_or(
        radial.status,
        jnp.where(support_valid, jnp.int32(0), jnp.int32(CPU_TIER_EXHAUSTED)),
    )
    result = CpuMagnificationResult(
        radial.value / rho**2,
        radial.error / rho**2,
        jnp.int32(0),
        jnp.int32(n_limb),
        radial.n_nodes,
        radial_status,
    )
    return result, limb_transition


def mag_limb_dark_cpu_fixed(
    w_center,
    rho,
    *,
    s,
    q,
    u1=0.0,
    n_limb=32,
    radial_splits=1,
    return_info=False,
):
    """Evaluate one fixed linear limb-darkening CPU tier."""

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    if radial_splits <= 0:
        raise ValueError("radial_splits must be positive")
    result, _ = _mag_limb_dark_cpu_fixed_impl(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_limb=n_limb,
        radial_splits=radial_splits,
        include_ghost_support=False,
    )
    return result if return_info else result.magnification


def mag_limb_dark_cpu(
    w_center,
    rho,
    *,
    s,
    q,
    u1=0.0,
    rtol=1.0e-3,
    return_info=False,
):
    """Evaluate linear limb darkening with the fixed 32/64/128 hierarchy."""

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    rtol = jnp.asarray(rtol, dtype=real_dtype)

    def evaluate(n_limb, radial_splits, tier, include_ghost_support=False):
        result, transition = _mag_limb_dark_cpu_fixed_impl(
            w_center,
            rho,
            s=s,
            q=q,
            u1=u1,
            n_limb=n_limb,
            radial_splits=radial_splits,
            include_ghost_support=include_ghost_support,
        )
        return result._replace(tier=jnp.int32(tier)), transition

    def needs_finer(result, minimum_tier):
        tolerance = rtol * jnp.maximum(jnp.abs(result.magnification), 1.0)
        return (
            (result.tier < minimum_tier)
            | (result.status != 0)
            | ~jnp.isfinite(result.magnification)
            | ~jnp.isfinite(result.estimated_error)
            | (result.estimated_error > tolerance)
        )

    def refine(previous, n_limb, radial_splits, tier, include_ghost_support=False):
        current, _ = evaluate(n_limb, radial_splits, tier, include_ghost_support)
        return current._replace(
            estimated_error=jnp.maximum(
                current.estimated_error,
                jnp.abs(current.magnification - previous.magnification),
            )
        )

    tier0, transition = evaluate(32, 1, 0)
    buried = lax.stop_gradient(
        hidden_caustic_candidate(
            w_center,
            rho,
            s=s,
            q=q,
            limb_transition=transition,
        )
    )
    caustic_topology = lax.stop_gradient(transition | buried)
    minimum_tier = jnp.where(
        caustic_topology | (rtol <= 1.0e-4), jnp.int32(2), jnp.int32(1)
    )
    tier1 = refine(tier0, 64, 2, 1, caustic_topology)
    tier2 = lax.cond(
        needs_finer(tier1, minimum_tier),
        lambda previous: refine(previous, 128, 4, 2, caustic_topology),
        lambda previous: previous,
        tier1,
    )
    tolerance = rtol * jnp.maximum(jnp.abs(tier2.magnification), 1.0)
    tier2 = tier2._replace(
        estimated_error=jnp.maximum(
            tier2.estimated_error,
            0.75 * tolerance,
        )
    )
    exhausted = (
        ~jnp.isfinite(tier2.magnification)
        | ~jnp.isfinite(tier2.estimated_error)
        | (tier2.estimated_error > tolerance)
        | buried
    )
    tier2 = tier2._replace(
        status=jnp.bitwise_or(
            tier2.status,
            jnp.where(exhausted, jnp.int32(CPU_TIER_EXHAUSTED), jnp.int32(0)),
        )
    )
    return tier2 if return_info else tier2.magnification


__all__ = ["mag_limb_dark_cpu", "mag_limb_dark_cpu_fixed"]
