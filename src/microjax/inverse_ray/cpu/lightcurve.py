"""CPU trajectory evaluation: multipoles followed by compact scalar solves."""

import jax.numpy as jnp
from jax import jit, lax, tree_util

from .one_shot import mag_cpu_one_shot
from .lens import binary_lens
from .result import CpuMagnificationResult

# A calibrated shortcut policy, not a full-solve error bound.
_CPU_MULTIPOLE_GATE = 1.0e-3


@jit
def mag_binary_cpu_lightcurve(w_points, rho, *, s, q, u1=0.0):
    """Full finite-source calculation, useful for isolating solver regressions."""
    return lax.map(
        lambda source: mag_cpu_one_shot(source, rho, lens=binary_lens(s, q), u1=u1),
        jnp.asarray(w_points),
    )


@jit
def mag_cpu_hybrid_lightcurve(
    w_points,
    multipole,
    accepted,
    correction,
    trigger_scale,
    rho,
    *,
    lens,
    u1=0.0,
):
    """Solve only sources rejected by the shared geometric multipole gate."""
    w_points = jnp.asarray(w_points)
    accepted = (
        accepted
        & jnp.isfinite(trigger_scale)
        & jnp.isfinite(correction)
        & (correction <= _CPU_MULTIPOLE_GATE * trigger_scale)
    )
    result = CpuMagnificationResult(
        multipole,
        jnp.zeros(w_points.shape, dtype=jnp.int32),
    )
    if w_points.shape[0] == 0:
        return result

    count = jnp.sum(~accepted, dtype=jnp.int32)
    indices = jnp.nonzero(~accepted, size=w_points.shape[0], fill_value=0)[0]

    def evaluate(i, values):
        index = indices[i]
        solved = mag_cpu_one_shot(w_points[index], rho, lens=lens, u1=u1)
        return tree_util.tree_map(
            lambda array, value: array.at[index].set(value),
            values,
            solved,
        )

    return lax.fori_loop(jnp.int32(0), count, evaluate, result)
