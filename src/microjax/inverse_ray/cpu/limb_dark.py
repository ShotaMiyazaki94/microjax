"""Linear limb-darkening radial-slice kernel for binary CPU ICRS."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from ..geometry.lens import binary_geometry
from ..geometry.mapping import distance_from_source
from ..roots.angular import angular_intervals_binary_roots
from .angular_profile import integrate_mu_residual
from .quadrature import integrate_gk15
from .sentinel import hidden_caustic_candidate
from .support import build_radial_support
from .uniform import CPU_TIER_EXHAUSTED, CpuMagnificationResult

Array = jnp.ndarray


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
):
    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    rho = jnp.asarray(rho, dtype=real_dtype)
    u1 = jnp.asarray(u1, dtype=real_dtype)
    lens = binary_geometry(s, q)
    support = build_radial_support(
        w_center,
        rho,
        s=lens.s,
        q=lens.q,
        n_limb=n_limb,
        include_all_roots=include_ghost_support,
    )
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
        active = jnp.arange(intervals.intervals.shape[0]) < intervals.n_intervals
        widths = intervals.intervals[:, 1] - intervals.intervals[:, 0]
        angular_measure = jnp.sum(jnp.where(active, widths, 0.0))

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
            return jnp.sqrt(radicand)

        residual = integrate_mu_residual(mu, intervals)
        angular_value = intensity_normalization * (
            (1.0 - u1) * angular_measure + u1 * residual.value
        )
        angular_error = intensity_normalization * (
            jnp.abs(1.0 - u1) * intervals.error + jnp.abs(u1) * residual.error
        )
        status = jnp.bitwise_or(intervals.status, residual.status)
        return radius * angular_value, jnp.abs(radius) * angular_error, status

    radial = integrate_gk15(
        radial_integrand,
        support.intervals,
        support.active,
        subdivisions=radial_splits,
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
