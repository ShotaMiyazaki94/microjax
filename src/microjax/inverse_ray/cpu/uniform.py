"""Fixed-tier prototype for uniform binary-source CPU ICRS."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from ..geometry.lens import binary_geometry
from ..roots.angular import angular_measure_binary_roots
from .quadrature import integrate_gk15
from .sentinel import hidden_caustic_candidate
from .support import build_radial_support

Array = jnp.ndarray

CPU_TIER_EXHAUSTED = 1 << 16


class CpuMagnificationResult(NamedTuple):
    """Magnification and diagnostics from one fixed CPU tier.

    For the production one-shot backend, ``status == 0`` means only that no
    finite-value, root, support, topology, or capacity failure was detected.
    It makes no accuracy claim; full solves report ``estimated_error`` as NaN.
    The multipole tier and legacy adaptive backend retain their own numerical
    error diagnostics.
    """

    magnification: Array
    estimated_error: Array
    tier: Array
    n_limb: Array
    n_radial_nodes: Array
    status: Array


def mag_uniform_cpu_fixed(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_limb: int = 32,
    radial_splits: int = 1,
    return_info: bool = False,
) -> Array | CpuMagnificationResult:
    """Evaluate one fixed radial-slice tier for a uniform binary source.

    This is an internal validation kernel.  It intentionally has no topology
    retries, hidden-caustic sentinel, or public backend dispatch yet.
    """

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    if radial_splits <= 0:
        raise ValueError("radial_splits must be positive")

    result, _ = _mag_uniform_cpu_fixed_impl(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        radial_splits=radial_splits,
        include_ghost_support=False,
    )
    return result if return_info else result.magnification


def _mag_uniform_cpu_fixed_impl(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_limb: int,
    radial_splits: int,
    include_ghost_support: bool | Array,
) -> tuple[CpuMagnificationResult, Array]:
    """Evaluate one tier and return its 3-to-5 limb-transition flag."""

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    rho = jnp.asarray(rho, dtype=real_dtype)
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

    def radial_integrand(radius):
        angular = angular_measure_binary_roots(
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
        return (
            radius * angular.measure,
            jnp.abs(radius) * angular.error,
            angular.status,
        )

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
    normalization = jnp.pi * rho**2
    result = CpuMagnificationResult(
        radial.value / normalization,
        radial.error / normalization,
        jnp.int32(0),
        jnp.int32(n_limb),
        radial.n_nodes,
        radial_status,
    )
    return result, limb_transition


def mag_uniform_cpu(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    rtol: float | Array = 1.0e-3,
    _tier0: CpuMagnificationResult | None = None,
    _limb_transition: Array | None = None,
    return_info: bool = False,
) -> Array | CpuMagnificationResult:
    """Evaluate the fixed 32/64/128 CPU accuracy hierarchy.

    Away from caustics, tiers 32 and 64 always run so source-limb
    discretization error is measured by their difference, followed by an
    optional 128/4 tier. A detected 3-to-5 image transition or buried caustic
    instead uses all-root support with staged 64/1 -> 128/2 and
    64/2 -> 128/4 pairs. Requests at or below 1e-4 may fall back to the more
    expensive 64/16 -> 128/32 pair.
    """

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    rtol = jnp.asarray(rtol, dtype=real_dtype)

    def evaluate(n_limb, radial_splits, tier, include_ghost_support=False):
        result, transition = _mag_uniform_cpu_fixed_impl(
            w_center,
            rho,
            s=s,
            q=q,
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

    def refine(
        previous,
        n_limb,
        radial_splits,
        tier,
        include_ghost_support=False,
    ):
        current, _ = evaluate(
            n_limb,
            radial_splits,
            tier,
            include_ghost_support,
        )
        tier_error = jnp.abs(current.magnification - previous.magnification)
        return current._replace(
            estimated_error=jnp.maximum(current.estimated_error, tier_error)
        )

    if _tier0 is None:
        tier0, limb_transition = evaluate(32, 1, 0)
    else:
        tier0 = _tier0
        if _limb_transition is None:
            raise ValueError("_limb_transition is required with _tier0")
        limb_transition = _limb_transition
    buried_caustic = lax.stop_gradient(
        hidden_caustic_candidate(
            w_center,
            rho,
            s=s,
            q=q,
            limb_transition=limb_transition,
        )
    )
    caustic_topology = lax.stop_gradient(limb_transition | buried_caustic)

    def regular_path(previous):
        tier1 = refine(previous, 64, 2, 1)
        minimum_regular_tier = jnp.where(rtol <= 1.0e-4, jnp.int32(2), jnp.int32(1))
        return lax.cond(
            needs_finer(tier1, minimum_regular_tier),
            lambda current: refine(current, 128, 4, 2),
            lambda current: current,
            tier1,
        )

    def topology_path(_previous):
        def topology_pair(coarse_splits, fine_splits):
            coarse, _ = evaluate(
                64,
                coarse_splits,
                1,
                include_ghost_support=True,
            )
            fine, _ = evaluate(
                128,
                fine_splits,
                2,
                include_ghost_support=True,
            )
            scale = jnp.maximum(jnp.abs(fine.magnification), 1.0)
            tier_error = 4.0 * jnp.abs(fine.magnification - coarse.magnification)
            calibrated_floor = 0.75 * rtol * scale
            raw_error = jnp.maximum(fine.estimated_error, tier_error)
            # A 1.5 safety margin covers the worst observed actual/reported
            # ratio (1.338) in the stratified VBBL caustic sweep.
            combined_error = jnp.maximum(
                1.5 * raw_error,
                calibrated_floor,
            )
            return fine._replace(
                estimated_error=combined_error,
                status=jnp.bitwise_or(coarse.status, fine.status),
            )

        fast = topology_pair(1, 2)
        medium = lax.cond(
            needs_finer(fast, jnp.int32(2)),
            lambda _: topology_pair(2, 4),
            lambda _: fast,
            operand=None,
        )
        high_accuracy_requested = rtol <= 1.0e-4
        return lax.cond(
            high_accuracy_requested & needs_finer(medium, jnp.int32(2)),
            lambda _: topology_pair(16, 32),
            lambda _: medium,
            operand=None,
        )

    tier2 = lax.cond(
        caustic_topology,
        topology_path,
        regular_path,
        tier0,
    )
    final_tolerance = rtol * jnp.maximum(jnp.abs(tier2.magnification), 1.0)
    tier2 = tier2._replace(
        estimated_error=jnp.maximum(
            tier2.estimated_error,
            0.75 * final_tolerance,
        )
    )
    exhausted = (
        ~jnp.isfinite(tier2.magnification)
        | ~jnp.isfinite(tier2.estimated_error)
        | (tier2.estimated_error > final_tolerance)
        # A caustic wholly enclosed by a source can give mutually consistent
        # radial tiers while both omit the same narrow radial interval.  The
        # angle-first fallback resolves this geometry; radial agreement alone
        # is therefore not a certificate for buried caustics.
        | buried_caustic
    )
    tier2 = tier2._replace(
        status=jnp.bitwise_or(
            tier2.status,
            jnp.where(
                exhausted,
                jnp.int32(CPU_TIER_EXHAUSTED),
                jnp.int32(0),
            ),
        )
    )
    return tier2 if return_info else tier2.magnification


__all__ = [
    "CpuMagnificationResult",
    "CPU_TIER_EXHAUSTED",
    "mag_uniform_cpu",
    "mag_uniform_cpu_fixed",
]
