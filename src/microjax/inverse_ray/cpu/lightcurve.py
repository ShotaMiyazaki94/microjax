"""Sequential light-curve scheduler for the binary CPU ICRS kernels."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, tree_util

from .cartesian_moment import mag_uniform_cartesian_cpu_adaptive
from .cartesian_limb_dark import mag_limb_dark_cartesian_adaptive
from .one_shot import mag_limb_dark_cpu_one_shot, mag_uniform_cpu_one_shot
from .uniform import CpuMagnificationResult

Array = jnp.ndarray

# The trace locates support extrema; it is not an integration quadrature. This
# is a static capacity: the Roman uniform path uses a 52-sample Cartesian
# scout, rebuilds polar support at 64 only after chart rejection, and reserves
# 96 for the separately compiled high-accuracy continuation and LD refinement.
_CPU_SUPPORT_CAPACITY = 96
_CPU_ONE_SHOT_UNIFORM_SUPPORT = 64
_CPU_ONE_SHOT_LD_SUPPORT = 64
# Production CPU uses one calibrated multipole shortcut policy.  This is a
# scheduler constant, not a user-visible accuracy promise: the one-shot full
# solve has a fixed quadrature rule and does not refine when this value changes.
_CPU_MULTIPOLE_GATE = 1.0e-3


@partial(
    jit,
    static_argnames=("u1", "cartesian_root_mode", "polar_root_mode"),
)
def mag_binary_cpu_one_shot_lightcurve(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
    cartesian_root_mode: str = "production",
    polar_root_mode: str = "companion",
) -> CpuMagnificationResult:
    """Evaluate the state-conditioned ICRS graph without any retry."""

    w_points = jnp.asarray(w_points)
    if u1 == 0.0:

        def evaluate(source):
            solved = mag_uniform_cpu_one_shot(
                source,
                rho,
                s=s,
                q=q,
                n_limb=_CPU_ONE_SHOT_UNIFORM_SUPPORT,
                cartesian_root_mode=cartesian_root_mode,
                polar_root_mode=polar_root_mode,
            )
            return CpuMagnificationResult(
                solved.magnification,
                solved.estimated_error,
                jnp.int32(4) + solved.stage,
                jnp.int32(_CPU_ONE_SHOT_UNIFORM_SUPPORT),
                solved.n_slices,
                solved.status,
            )

    else:

        def evaluate(source):
            solved = mag_limb_dark_cpu_one_shot(
                source,
                rho,
                s=s,
                q=q,
                u1=u1,
                n_limb=_CPU_ONE_SHOT_LD_SUPPORT,
                cartesian_root_mode=cartesian_root_mode,
            )
            return CpuMagnificationResult(
                solved.magnification,
                solved.estimated_error,
                jnp.int32(4) + solved.stage,
                jnp.int32(_CPU_ONE_SHOT_LD_SUPPORT),
                solved.n_slices,
                solved.status,
            )

    return lax.map(evaluate, w_points)


@partial(
    jit,
    static_argnames=("u1", "cartesian_root_mode", "polar_root_mode"),
)
def mag_binary_cpu_one_shot_hybrid_lightcurve(
    w_points: Array,
    multipole: Array,
    multipole_accepted: Array,
    multipole_error: Array,
    multipole_trigger_scale: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
    cartesian_root_mode: str = "production",
    polar_root_mode: str = "companion",
) -> CpuMagnificationResult:
    """Use multipoles first and one state-conditioned full solve otherwise."""

    w_points = jnp.asarray(w_points)
    scale = multipole_trigger_scale
    # ``multipole_error`` is a correction diagnostic rather than a rigorous
    # remainder bound, but it is still a necessary fast-path condition.  Use
    # the same correction gate for uniform and limb-darkened profiles so that
    # brightness law alone cannot weaken the full-solve trigger.
    # Finiteness is part of the shared geometric trigger as well.  Testing
    # the requested-profile value here would let brightness law alone change
    # the fast/full mask, contrary to the scheduler invariant.
    multipole_certified = (
        multipole_accepted
        & jnp.isfinite(multipole_trigger_scale)
        & jnp.isfinite(multipole_error)
    )
    strict_correction = multipole_error <= _CPU_MULTIPOLE_GATE * scale
    multipole_certified = multipole_certified & strict_correction
    result = CpuMagnificationResult(
        multipole,
        jnp.maximum(multipole_error, 0.5 * _CPU_MULTIPOLE_GATE * scale),
        jnp.full(w_points.shape, -1, dtype=jnp.int32),
        jnp.zeros(w_points.shape, dtype=jnp.int32),
        jnp.zeros(w_points.shape, dtype=jnp.int32),
        jnp.zeros(w_points.shape, dtype=jnp.int32),
    )
    if w_points.shape[0] == 0:
        return result

    n_points = w_points.shape[0]
    n_active = jnp.sum(~multipole_certified, dtype=jnp.int32)
    indices = jnp.nonzero(~multipole_certified, size=n_points, fill_value=0)[0]

    if u1 == 0.0:

        def solve(source, external, external_error):
            return mag_uniform_cpu_one_shot(
                source,
                rho,
                s=s,
                q=q,
                n_limb=_CPU_ONE_SHOT_UNIFORM_SUPPORT,
                external_magnification=external,
                external_estimated_error=external_error,
                cartesian_root_mode=cartesian_root_mode,
                polar_root_mode=polar_root_mode,
            )

    else:

        def solve(source, external, external_error):
            return mag_limb_dark_cpu_one_shot(
                source,
                rho,
                s=s,
                q=q,
                u1=u1,
                n_limb=_CPU_ONE_SHOT_LD_SUPPORT,
                external_magnification=external,
                external_estimated_error=external_error,
                cartesian_root_mode=cartesian_root_mode,
            )

    def evaluate_rejected(compact_index, values):
        point_index = indices[compact_index]
        solved = solve(
            w_points[point_index],
            multipole[point_index],
            multipole_error[point_index],
        )
        packed = CpuMagnificationResult(
            solved.magnification,
            solved.estimated_error,
            jnp.int32(4) + solved.stage,
            jnp.int32(_CPU_ONE_SHOT_UNIFORM_SUPPORT if u1 == 0.0 else _CPU_ONE_SHOT_LD_SUPPORT),
            solved.n_slices,
            solved.status,
        )
        return tree_util.tree_map(
            lambda array, value: array.at[point_index].set(value),
            values,
            packed,
        )

    return lax.fori_loop(jnp.int32(0), n_active, evaluate_rejected, result)


@partial(jit, static_argnames=("u1", "rtol"))
def mag_binary_cpu_lightcurve(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
    rtol: float = 1.0e-3,
) -> CpuMagnificationResult:
    """Evaluate the full ICRS charts without the multipole prefilter."""

    w_points = jnp.asarray(w_points)
    internal_rtol = 0.5 * rtol if rtol <= 1.0e-4 else rtol

    if u1 == 0.0:

        def evaluate(source):
            chart = mag_uniform_cartesian_cpu_adaptive(
                source,
                rho,
                s=s,
                q=q,
                rtol=internal_rtol,
                n_limb=_CPU_SUPPORT_CAPACITY,
                continuation=("bernstein_dynamic_full" if internal_rtol <= 1.0e-4 else "bernstein_dynamic"),
            )
            return CpuMagnificationResult(
                chart.magnification,
                chart.estimated_error,
                jnp.int32(4) + chart.stage,
                jnp.int32(_CPU_SUPPORT_CAPACITY),
                chart.n_slices,
                chart.status,
            )

    else:

        def evaluate(source):
            chart = mag_limb_dark_cartesian_adaptive(
                source,
                rho,
                s=s,
                q=q,
                u1=u1,
                rtol=internal_rtol,
                n_limb=_CPU_SUPPORT_CAPACITY,
                return_info=True,
            )
            return CpuMagnificationResult(
                chart.magnification,
                chart.estimated_error,
                jnp.int32(4) + chart.stage,
                jnp.int32(_CPU_SUPPORT_CAPACITY),
                chart.n_slices,
                chart.status,
            )

    return lax.map(evaluate, w_points)


@partial(jit, static_argnames=("u1", "rtol"))
def mag_binary_cpu_hybrid_lightcurve(
    w_points: Array,
    multipole: Array,
    multipole_accepted: Array,
    multipole_error: Array,
    multipole_trigger_scale: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
    rtol: float = 1.0e-3,
) -> CpuMagnificationResult:
    """Use the fast multipole path except where a full CPU solve is required."""

    w_points = jnp.asarray(w_points)
    # The optional 1e-4 mode uses a tighter internal certificate because the
    # empirical tier-difference calibration is dominated by the 1e-3 Roman
    # production regime.
    rtol = 0.5 * rtol if rtol <= 1.0e-4 else rtol

    if u1 == 0.0:

        def full(source, external_multipole, _external_error):
            source_array = jnp.asarray(source)
            rho_array = jnp.asarray(rho, dtype=source_array.real.dtype)
            cartesian = mag_uniform_cartesian_cpu_adaptive(
                source_array,
                rho_array,
                s=s,
                q=q,
                rtol=rtol,
                n_limb=_CPU_SUPPORT_CAPACITY,
                continuation=("bernstein_dynamic_full" if rtol <= 1.0e-4 else "bernstein_dynamic"),
                external_magnification=(None if rtol <= 1.0e-4 else external_multipole),
            )

            return CpuMagnificationResult(
                cartesian.magnification,
                cartesian.estimated_error,
                jnp.int32(4) + cartesian.stage,
                jnp.int32(_CPU_SUPPORT_CAPACITY),
                cartesian.n_slices,
                cartesian.status,
            )

    else:

        def full(source, external_multipole, _external_error):
            cartesian = mag_limb_dark_cartesian_adaptive(
                source,
                rho,
                s=s,
                q=q,
                u1=u1,
                rtol=rtol,
                n_limb=_CPU_SUPPORT_CAPACITY,
                external_magnification=external_multipole,
                return_info=True,
            )

            return CpuMagnificationResult(
                cartesian.magnification,
                cartesian.estimated_error,
                jnp.int32(4) + cartesian.stage,
                jnp.int32(_CPU_SUPPORT_CAPACITY),
                cartesian.n_slices,
                cartesian.status,
            )

    scale = multipole_trigger_scale
    # The geometric prefilter alone is not an accuracy certificate.  In
    # particular, a finite source can straddle the resonant caustic while the
    # point-centred series still passes the relaxed planetary guards.  Require
    # its already-computed remainder estimate at every requested tolerance;
    # bypassing this check at the 1e-3 default produced rare 16--43% false
    # accepts on the standard Jacobian trajectory.
    multipole_certified = (
        multipole_accepted
        & jnp.isfinite(multipole_trigger_scale)
        & jnp.isfinite(multipole_error)
        & (multipole_error <= rtol * scale)
    )
    fast = CpuMagnificationResult(
        multipole,
        jnp.maximum(multipole_error, 0.5 * rtol * scale),
        jnp.full(w_points.shape, -1, dtype=jnp.int32),
        jnp.zeros(w_points.shape, dtype=jnp.int32),
        jnp.zeros(w_points.shape, dtype=jnp.int32),
        jnp.zeros(w_points.shape, dtype=jnp.int32),
    )
    if w_points.shape[0] == 0:
        return fast

    # Compact rejected points and solve exactly that many sources.  A dynamic
    # CPU loop avoids accelerator-style padding and keeps every scalar tier
    # branch lazy.  Forward-mode AD is supported; reverse-mode AD through this
    # data-dependent loop is intentionally outside the CPU API contract.
    n_points = w_points.shape[0]
    n_active = jnp.sum(~multipole_certified, dtype=jnp.int32)
    indices = jnp.nonzero(
        ~multipole_certified,
        size=n_points,
        fill_value=0,
    )[0]

    def evaluate_rejected(compact_index, result):
        point_index = indices[compact_index]
        solved = full(
            w_points[point_index],
            multipole[point_index],
            multipole_error[point_index],
        )
        return tree_util.tree_map(
            lambda values, value: values.at[point_index].set(value),
            result,
            solved,
        )

    return lax.fori_loop(
        jnp.int32(0),
        n_active,
        evaluate_rejected,
        fast,
    )


__all__ = [
    "mag_binary_cpu_hybrid_lightcurve",
    "mag_binary_cpu_lightcurve",
    "mag_binary_cpu_one_shot_hybrid_lightcurve",
    "mag_binary_cpu_one_shot_lightcurve",
]
