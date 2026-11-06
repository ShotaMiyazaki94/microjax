"""Shared bounded GL5/GK11 allocation for the two CPU image moments.

Both moments drive the same allocation, independently of limb darkening.
The embedded discrepancy is an internal stopping indicator, not an error bound.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Legendre, Polynomial

_BUDGET = 84
_RTOL = 1e-4
# One common Gauss-Jacobi rule resolves the limb square root on each ray.
# Twelve nodes retain the thin-image LD case that eight nodes undersample.
_PROFILE_ANGLES = np.pi * np.arange(12, 0, -1) / 13
PROFILE_NODES = np.cos(_PROFILE_ANGLES)
PROFILE_WEIGHTS = np.pi * np.sin(_PROFILE_ANGLES) ** 2 / 13
_GL5_X, _GL5_W = np.polynomial.legendre.leggauss(5)
_EMBEDDED_INDICES = np.asarray([0, 2, 4])
_EMBEDDED_X = _GL5_X[_EMBEDDED_INDICES]
_E3_W = np.zeros(5)
_E3_W[_EMBEDDED_INDICES] = np.linalg.solve(
    _EMBEDDED_X[None, :] ** np.arange(3)[:, None], np.asarray([2.0, 0.0, 2.0 / 3.0])
)


def _kronrod11_rule():
    """Build the positive GK11 extension with the exact existing GL5 nodes.

    The even, monic extension E6 satisfies integral(P5 * E6 * x**k) = 0
    for k=1,3,5. Even k vanish by symmetry. The resulting rule is exact
    through degree 17; the two rules share all five GL5 evaluation points.
    Construction uses host NumPy only, outside any JAX evaluation graph.
    """
    legendre = Legendre.basis(5).convert(kind=Polynomial)

    def moment(degree):
        primitive = (legendre * Polynomial([0] * degree + [1])).integ()
        return primitive(1) - primitive(-1)

    system = np.asarray([[moment(k + j) for j in (0, 2, 4)] for k in (1, 3, 5)])
    rhs = -np.asarray([moment(k + 6) for k in (1, 3, 5)])
    c0, c2, c4 = np.linalg.solve(system, rhs)
    extension = Polynomial([c0, 0, c2, 0, c4, 0, 1]).roots()
    if np.iscomplexobj(extension):
        raise ValueError("GK11 extension unexpectedly has complex nodes")
    positive = np.sort(extension[extension > 0])
    extension = np.concatenate((-positive[::-1], positive))
    nodes = np.sort(np.concatenate((_GL5_X, extension)))
    moments = np.asarray([2 / (k + 1) if k % 2 == 0 else 0 for k in range(11)])
    weights = np.linalg.solve(nodes[None, :] ** np.arange(11)[:, None], moments)
    weights = (weights + weights[::-1]) / 2
    if not (
        np.array_equal(nodes[1::2], _GL5_X)
        and np.all(np.diff(nodes) > 0)
        and np.all(np.abs(nodes) < 1)
        and np.all(weights > 0)
    ):
        raise ValueError("GK11 rule must be positive, nested, and interior")
    return nodes, weights


_GK11_X, _GK11_W = _kronrod11_rule()


class MomentIntegral(NamedTuple):
    value: jax.Array
    status: jax.Array


class _CachedMomentState(NamedTuple):
    origin: jax.Array
    left: jax.Array
    right: jax.Array
    values: jax.Array
    errors: jax.Array
    priorities: jax.Array
    statuses: jax.Array
    samples: jax.Array
    upgraded: jax.Array
    n_live: jax.Array
    n_nodes: jax.Array


def _sum_status(values):
    return jnp.sum(values, dtype=jnp.int32)


def integrate_adaptive_moments(profile, cells, active):
    """Integrate two moments, sharing at most 84 rays per active support cell.

    ``profile(x)`` returns moments (N, 2) and invalid-root counts (N,).
    Every cell begins with two GL5 panels in a sine-squared coordinate.
    The largest eligible discrepancy gets a six-ray GK11 extension or a
    ten-ray bisection. Two consecutive passing allocations stop refinement;
    initialization counts as the first pass, so nonempty support is refined
    at least once. Both moments must pass relative indicator 1e-4 separately.
    Forward AD follows moving endpoints and the selected allocation.
    """
    dtype = cells.dtype
    gx = jnp.asarray(_GL5_X, dtype=dtype)
    gw = jnp.asarray(_GL5_W, dtype=dtype)
    ew = jnp.asarray(_E3_W, dtype=dtype)
    kx = jnp.asarray(_GK11_X, dtype=dtype)
    kw = jnp.asarray(_GK11_W, dtype=dtype)
    # Each h split adds one live panel and consumes ten rays plus a prior
    # six-ray p upgrade. That upgrade cannot serve another split: h <= p,
    # so 16*h <= budget - initialization. This bounds storage even when
    # every support cell is active, without changing the ray budget.
    max_h = cells.shape[0] * (_BUDGET - 10) // 16
    capacity = 2 * cells.shape[0] + max_h
    cell_count = jnp.sum(active, dtype=jnp.int32)
    count = 2 * cell_count
    indices = jnp.repeat(jnp.nonzero(active, size=active.size, fill_value=0)[0].astype(jnp.int32), 2)
    origin = jnp.zeros(capacity, dtype=jnp.int32).at[: indices.size].set(indices)
    left = (
        jnp.zeros(capacity, dtype=dtype)
        .at[: indices.size]
        .set(jnp.tile(jnp.asarray([0.0, 0.5], dtype=dtype), cells.shape[0]))
    )
    right = (
        jnp.ones(capacity, dtype=dtype)
        .at[: indices.size]
        .set(jnp.tile(jnp.asarray([0.5, 1.0], dtype=dtype), cells.shape[0]))
    )
    values = jnp.zeros((capacity, 2), dtype=dtype)
    errors = jnp.zeros_like(values)
    priorities = jnp.full(capacity, -jnp.inf, dtype=dtype)
    statuses = jnp.zeros(capacity, dtype=jnp.int32)
    # Only GL5 samples are reused. After a GK11 upgrade the next action is an
    # h split, which replaces the panel with fresh GL5 children. Keep five
    # transformed samples per panel; the eleven-node array is only temporary.
    cache = jnp.zeros((capacity, 5, 2), dtype=dtype)
    upgraded = jnp.zeros(capacity, dtype=jnp.bool_)

    def samples(cell, lo, hi, nodes):
        angle = 0.5 * jnp.pi * (lo + 0.5 * (hi - lo) * (nodes + 1))
        bounds = cells[cell]
        width = bounds[1] - bounds[0]
        x = bounds[0] + width * jnp.sin(angle) ** 2
        jac = 0.25 * jnp.pi * (hi - lo) * width * jnp.sin(2 * angle)
        y, s = profile(x)
        return jac[:, None] * y, s.astype(jnp.int32)

    def paired_gauss_panel(cell, lo, hi):
        # One ten-ray profile call avoids an extra vmap axis in root solves.
        angle = 0.5 * jnp.pi * (lo[:, None] + 0.5 * (hi - lo)[:, None] * (gx[None, :] + 1))
        bounds = cells[cell]
        width = bounds[1] - bounds[0]
        x = bounds[0] + width * jnp.sin(angle) ** 2
        jac = 0.25 * jnp.pi * (hi - lo)[:, None] * width * jnp.sin(2 * angle)
        y, s = profile(x.reshape(-1))
        y = jac[..., None] * y.reshape((2, 5, 2))
        s = s.reshape((2, 5)).astype(jnp.int32)
        value = jnp.sum(gw[None, :, None] * y, axis=1)
        coarse = jnp.sum(ew[None, :, None] * y, axis=1)
        error = jnp.abs(value - coarse)
        return value, error, jax.vmap(_sum_status)(s), y

    def initialize(i, state):
        val, err, priority, status, ys = state
        start = 2 * i
        v, e, s, cy = paired_gauss_panel(
            indices[start],
            jnp.asarray([0.0, 0.5], dtype=dtype),
            jnp.asarray([0.5, 1.0], dtype=dtype),
        )
        slots = start + jnp.arange(2)
        return (
            val.at[slots].set(v),
            err.at[slots].set(e),
            priority.at[slots].set(jnp.sum(e, axis=1)),
            status.at[slots].set(s),
            ys.at[slots].set(cy),
        )

    values, errors, priorities, statuses, cache = jax.lax.fori_loop(
        0,
        cell_count,
        initialize,
        (values, errors, priorities, statuses, cache),
    )
    state = _CachedMomentState(
        origin,
        left,
        right,
        values,
        errors,
        priorities,
        statuses,
        cache,
        upgraded,
        count,
        10 * cell_count,
    )
    limit = _BUDGET * cell_count

    def eligible_panels(state):
        cost = jnp.where(state.upgraded, 10, 6)
        return (jnp.arange(capacity) < state.n_live) & (cost <= limit - state.n_nodes)

    def refine(state):
        priority = jnp.where(eligible_panels(state), state.priorities, -jnp.inf)
        chosen = jnp.argmax(priority)

        def p_upgrade(state):
            y, s = samples(state.origin[chosen], state.left[chosen], state.right[chosen], kx[::2])
            cy = jnp.zeros((11, 2), dtype=dtype).at[1::2].set(state.samples[chosen]).at[::2].set(y)
            # Reused samples retain their status without counting them twice.
            status = _sum_status(jnp.concatenate((state.statuses[chosen, None], s)))
            value = jnp.sum(kw[:, None] * cy, axis=0)
            error = jnp.abs(value - state.values[chosen])
            return state._replace(
                values=state.values.at[chosen].set(value),
                errors=state.errors.at[chosen].set(error),
                priorities=state.priorities.at[chosen].set(jnp.sum(error)),
                statuses=state.statuses.at[chosen].set(status),
                upgraded=state.upgraded.at[chosen].set(True),
                n_nodes=state.n_nodes + 6,
            )

        def h_split(state):
            mid = 0.5 * (state.left[chosen] + state.right[chosen])
            v, e, s, cy = paired_gauss_panel(
                state.origin[chosen],
                jnp.stack([state.left[chosen], mid]),
                jnp.stack([mid, state.right[chosen]]),
            )
            new = state.n_live
            slots = jnp.stack([chosen, new])
            return state._replace(
                origin=state.origin.at[new].set(state.origin[chosen]),
                left=state.left.at[new].set(mid),
                right=state.right.at[new].set(state.right[chosen]).at[chosen].set(mid),
                values=state.values.at[slots].set(v),
                errors=state.errors.at[slots].set(e),
                priorities=state.priorities.at[slots].set(jnp.sum(e, axis=1)),
                statuses=state.statuses.at[slots].set(s),
                samples=state.samples.at[slots].set(cy),
                upgraded=state.upgraded.at[slots].set(False),
                n_live=new + 1,
                n_nodes=state.n_nodes + 10,
            )

        return jax.lax.cond(state.upgraded[chosen], h_split, p_upgrade, state)

    def allocation_passes(state):
        total = jnp.sum(state.values, axis=0)
        error = jnp.sum(state.errors, axis=0)
        return (
            jnp.all(jnp.isfinite(total) & jnp.isfinite(error))
            & jnp.all(error <= _RTOL * jnp.abs(total))
            & (_sum_status(state.statuses) == 0)
        )

    def can_refine(state):
        return jnp.any(eligible_panels(state))

    def keep_refining(carry):
        state, streak = carry
        return can_refine(state) & (streak < 2)

    def refine_and_check(carry):
        state, streak = carry
        state = refine(state)
        streak = jnp.where(allocation_passes(state), streak + 1, 0)
        return state, streak

    state, _ = jax.lax.while_loop(
        keep_refining,
        refine_and_check,
        (state, allocation_passes(state).astype(jnp.int32)),
    )
    return MomentIntegral(jnp.sum(state.values, axis=0), _sum_status(state.statuses))
