"""Bounded inverse-ray quadrature of uniform-area and limb-profile moments.

A profile evaluates both moments with one inverse-ray root solve.
Allocation is independent of the limb-darkening coefficient; callers mix the
integrated moments afterward. Cartesian and angular one-shot integration
reuse GL5 samples in a GK11 extension and stop on an empirical relative
indicator. The legacy fixed-work kernel remains available for radial charts.
Both kernels cap the total ray count at ``budget * n_active_cells``;
allocation can concentrate more than ``budget`` rays in an individual cell.
"""

from math import isfinite
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Legendre, Polynomial

# A five-node Gauss rule integrates polynomials through degree 9. Embed a
# positive three-node interpolatory rule for the allocation error indicator;
# its symmetric nodes integrate through degree 3. The indicator does not
# certify missing support or physical root completeness.
_GL5_X, _GL5_W = np.polynomial.legendre.leggauss(5)
_EMBEDDED_INDICES = np.asarray([0, 2, 4])
_EMBEDDED_X = _GL5_X[_EMBEDDED_INDICES]
_E3_W = np.zeros(5)
_E3_W[_EMBEDDED_INDICES] = np.linalg.solve(
    _EMBEDDED_X[None, :] ** np.arange(3)[:, None],
    np.asarray([2.0, 0.0, 2.0 / 3.0]),
)


class MomentIntegral(NamedTuple):
    value: jax.Array  # [uniform area, integral of mu]
    error: jax.Array  # per-moment embedded discrepancy + propagated error
    status: jax.Array
    n_nodes: jax.Array  # all evaluated rays, including replaced panels


class AdaptiveMomentIntegral(NamedTuple):
    """Moment result with empirical stopping diagnostics, separate from status.

    ``status`` retains the caller's structural root/status reduction. A clear
    status neither certifies accuracy nor implies that the work cap permitted
    convergence. ``converged`` refers only to the embedded moment indicator.
    """

    value: jax.Array
    error: jax.Array
    status: jax.Array
    n_nodes: jax.Array
    converged: jax.Array
    budget_exhausted: jax.Array
    n_actions: jax.Array
    consecutive_passes: jax.Array
    n_p_upgrades: jax.Array
    n_h_splits: jax.Array


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


class _CachedMomentState(NamedTuple):
    origin: jax.Array
    left: jax.Array
    right: jax.Array
    values: jax.Array
    errors: jax.Array
    priorities: jax.Array
    statuses: jax.Array
    samples: jax.Array
    propagated: jax.Array
    upgraded: jax.Array
    n_live: jax.Array
    n_nodes: jax.Array


def _or_status(values):
    return jnp.bitwise_or.reduce(values.reshape(-1))


def _sum_status(values):
    """Use for root counts; the default reducer combines radial status flags."""
    return jnp.sum(values, dtype=jnp.int32)


def integrate_moments(profile, cells, active, *, budget=18, status_reducer=_or_status):
    """Integrate two physical profile moments using a shared fixed ray budget.

    ``profile(x)`` returns ``(moments, errors, statuses)`` of shapes (N,2),
    (N,2), and (N,), respectively. Error inputs are nonnegative absolute
    estimates in the same units as each moment. ``cells`` has shape (M,2).
    Callers supply a static status reducer: sum for invalid-root counts, OR
    for bit fields. Public fixed-order APIs can retain their existing default
    integrator; only one-shot CPU integration needs this moment kernel.

    Every active cell first gets two five-sample panels in its original sin²
    coordinate. The remaining budget bisects the panel with the largest sum
    of moment discrepancies. Each bisection costs ten samples. Total work is
    ``10*n + 10*floor(n*(budget-10)/10) <= budget*n``, where ``n`` is the
    number of active cells. This bound includes discarded parent samples;
    it is not a separate cap on each cell. There is no
    tolerance-controlled retry and no parameter-specific exception route.
    Forward derivatives retain moving support endpoints and follow the
    selected discrete allocation, as with the existing support partition.
    """
    if budget < 10:
        raise ValueError("the moment quadrature budget must be at least ten")
    nodes = jnp.asarray(_GL5_X, dtype=cells.dtype)
    fine_weights = jnp.asarray(_GL5_W, dtype=cells.dtype)
    coarse_weights = jnp.asarray(_E3_W, dtype=cells.dtype)
    cell_count = jnp.sum(active, dtype=jnp.int32)
    count = 2 * cell_count
    indices = jnp.repeat(jnp.nonzero(active, size=active.size, fill_value=0)[0].astype(jnp.int32), 2)
    max_splits = cells.shape[0] * (budget - 10) // 10
    capacity = 2 * cells.shape[0] + max_splits
    origin = jnp.zeros(capacity, dtype=jnp.int32).at[: indices.size].set(indices)
    left = (
        jnp.zeros(capacity, dtype=cells.dtype)
        .at[: indices.size]
        .set(jnp.tile(jnp.asarray([0.0, 0.5], dtype=cells.dtype), cells.shape[0]))
    )
    right = (
        jnp.ones(capacity, dtype=cells.dtype)
        .at[: indices.size]
        .set(jnp.tile(jnp.asarray([0.5, 1.0], dtype=cells.dtype), cells.shape[0]))
    )
    values = jnp.zeros((capacity, 2), dtype=cells.dtype)
    errors = jnp.zeros((capacity, 2), dtype=cells.dtype)
    priorities = jnp.full(capacity, -jnp.inf, dtype=cells.dtype)
    statuses = jnp.zeros(capacity, dtype=jnp.int32)

    def panel(cell, lo, hi):
        angle = 0.5 * jnp.pi * (lo + 0.5 * (hi - lo) * (nodes + 1))
        bounds = cells[cell]
        width = bounds[1] - bounds[0]
        x = bounds[0] + width * jnp.sin(angle) ** 2
        jac = 0.25 * jnp.pi * (hi - lo) * width * jnp.sin(2 * angle)
        y, propagated, status = profile(x)
        weighted = (fine_weights * jac)[:, None]
        fine = jnp.sum(weighted * y, axis=0)
        coarse = jnp.sum((coarse_weights * jac)[:, None] * y, axis=0)
        error = jnp.abs(fine - coarse) + jnp.sum(jnp.abs(weighted) * propagated, axis=0)
        return fine, error, status_reducer(status.astype(jnp.int32))

    def initial(i, state):
        val, err, priority, status = state
        first = 2 * i
        # The two initial half panels share a cell, just as refined children do.
        # Evaluate them together without changing their nodes or allocation.
        v, e, s = jax.vmap(panel, in_axes=(None, 0, 0))(
            indices[first],
            jnp.asarray([0.0, 0.5], dtype=cells.dtype),
            jnp.asarray([0.5, 1.0], dtype=cells.dtype),
        )
        return (
            val.at[first].set(v[0]).at[first + 1].set(v[1]),
            err.at[first].set(e[0]).at[first + 1].set(e[1]),
            priority.at[first].set(jnp.sum(e[0])).at[first + 1].set(jnp.sum(e[1])),
            status.at[first].set(s[0]).at[first + 1].set(s[1]),
        )

    values, errors, priorities, statuses = jax.lax.fori_loop(
        0, cell_count, initial, (values, errors, priorities, statuses)
    )
    splits = cell_count * (budget - 10) // 10

    def refine(i, state):
        orig, lo, hi, val, err, priority, status = state
        chosen = jnp.argmax(priority)
        new = count + i
        mid = 0.5 * (lo[chosen] + hi[chosen])
        child_value, child_error, child_status = jax.vmap(panel, in_axes=(None, 0, 0))(
            orig[chosen], jnp.stack([lo[chosen], mid]), jnp.stack([mid, hi[chosen]])
        )
        orig = orig.at[new].set(orig[chosen])
        lo = lo.at[new].set(mid)
        hi = hi.at[new].set(hi[chosen]).at[chosen].set(mid)
        val = val.at[chosen].set(child_value[0]).at[new].set(child_value[1])
        err = err.at[chosen].set(child_error[0]).at[new].set(child_error[1])
        priority = priority.at[chosen].set(jnp.sum(child_error[0])).at[new].set(jnp.sum(child_error[1]))
        status = status.at[chosen].set(child_status[0]).at[new].set(child_status[1])
        return orig, lo, hi, val, err, priority, status

    _, _, _, values, errors, _, statuses = jax.lax.fori_loop(
        0, splits, refine, (origin, left, right, values, errors, priorities, statuses)
    )
    return MomentIntegral(
        jnp.sum(values, axis=0),
        jnp.sum(errors, axis=0),
        status_reducer(statuses),
        5 * count + 10 * splits,
    )


def integrate_adaptive_moments(profile, cells, active, *, budget=84, rtol=1e-4, status_reducer=_or_status):
    """Integrate two moments with cached GL5/GK11 refinement and a work cap.

    ``profile(x)`` returns moments, nonnegative absolute errors, and statuses
    with shapes (N,2), (N,2), and (N,). Each active support cell starts with
    two GL5 half panels in its sin² coordinate. The largest eligible sum of
    moment indicators selects each refinement. Upgrading a GL5 panel to GK11
    evaluates six new rays and retains all five original values, errors, and
    statuses. Bisecting a GK11 panel evaluates two fresh GL5 panels (ten
    rays); parent samples cannot be reused at their child Gauss nodes.

    Both moment indicators must independently satisfy ``error <= rtol *
    abs(value)`` in two successive allocations, with at least one refinement
    action and a clear reduced status. There is no absolute scale floor.
    The GL5 indicator is its embedded E3 discrepancy; the GK11 indicator is
    abs(GK11-GL5). Both include weighted profile errors. These indicators do
    not certify missing support, missed roots, or parameter derivatives.

    Total evaluations, including discarded parents, cannot exceed ``budget
    * n_active_cells``. The bound is shared across cells. Status describes
    live quadrature rays, with no duplicate count for reused samples. Budget
    exhaustion is separate telemetry and does not invent a structural status
    flag or trigger a chart retry. Empty support returns converged zeros.

    Cached values retain JAX dependencies on parameters and moving support
    endpoints. Forward JVPs and forward-over-forward derivatives follow the
    selected allocation. Reverse differentiation through the dynamic loop
    is unsupported, as for the legacy dynamic allocation kernel.
    """
    if budget < 16:
        raise ValueError("budget must permit at least one six-ray upgrade")
    if not isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and positive")
    dtype = cells.dtype
    if cells.shape[0] == 0:
        zero = jnp.zeros(2, dtype=dtype)
        return AdaptiveMomentIntegral(
            zero,
            zero,
            jnp.int32(0),
            jnp.int32(0),
            jnp.bool_(True),
            jnp.bool_(False),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
            jnp.int32(0),
        )

    gx = jnp.asarray(_GL5_X, dtype=dtype)
    gw = jnp.asarray(_GL5_W, dtype=dtype)
    ew = jnp.asarray(_E3_W, dtype=dtype)
    kx = jnp.asarray(_GK11_X, dtype=dtype)
    kw = jnp.asarray(_GK11_W, dtype=dtype)
    # Every h split consumes ten evaluations and adds only one live panel.
    max_h = cells.shape[0] * (budget - 10) // 10
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
    propagated = jnp.zeros_like(cache)
    upgraded = jnp.zeros(capacity, dtype=jnp.bool_)

    def samples(cell, lo, hi, nodes):
        angle = 0.5 * jnp.pi * (lo + 0.5 * (hi - lo) * (nodes + 1))
        bounds = cells[cell]
        width = bounds[1] - bounds[0]
        x = bounds[0] + width * jnp.sin(angle) ** 2
        jac = 0.25 * jnp.pi * (hi - lo) * width * jnp.sin(2 * angle)
        y, e, s = profile(x)
        return jac[:, None] * y, jnp.abs(jac[:, None]) * e, s.astype(jnp.int32)

    def paired_gauss_panel(cell, lo, hi):
        # One ten-ray profile call avoids an extra vmap axis in root solves.
        angle = 0.5 * jnp.pi * (lo[:, None] + 0.5 * (hi - lo)[:, None] * (gx[None, :] + 1))
        bounds = cells[cell]
        width = bounds[1] - bounds[0]
        x = bounds[0] + width * jnp.sin(angle) ** 2
        jac = 0.25 * jnp.pi * (hi - lo)[:, None] * width * jnp.sin(2 * angle)
        y, e, s = profile(x.reshape(-1))
        y = jac[..., None] * y.reshape((2, 5, 2))
        e = jnp.abs(jac[..., None]) * e.reshape((2, 5, 2))
        s = s.reshape((2, 5)).astype(jnp.int32)
        value = jnp.sum(gw[None, :, None] * y, axis=1)
        coarse = jnp.sum(ew[None, :, None] * y, axis=1)
        error = jnp.abs(value - coarse) + jnp.sum(gw[None, :, None] * e, axis=1)
        return value, error, jax.vmap(status_reducer)(s), y, e

    def initialize(i, state):
        val, err, priority, status, ys, es = state
        start = 2 * i
        v, e, s, cy, ce = paired_gauss_panel(
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
            es.at[slots].set(ce),
        )

    values, errors, priorities, statuses, cache, propagated = jax.lax.fori_loop(
        0,
        cell_count,
        initialize,
        (values, errors, priorities, statuses, cache, propagated),
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
        propagated,
        upgraded,
        count,
        10 * cell_count,
    )
    limit = budget * cell_count

    def eligible_panels(state):
        cost = jnp.where(state.upgraded, 10, 6)
        return (jnp.arange(capacity) < state.n_live) & (cost <= limit - state.n_nodes)

    def refine(state):
        priority = jnp.where(eligible_panels(state), state.priorities, -jnp.inf)
        chosen = jnp.argmax(priority)

        def p_upgrade(state):
            y, e, s = samples(state.origin[chosen], state.left[chosen], state.right[chosen], kx[::2])
            cy = jnp.zeros((11, 2), dtype=dtype).at[1::2].set(state.samples[chosen]).at[::2].set(y)
            ce = jnp.zeros_like(cy).at[1::2].set(state.propagated[chosen]).at[::2].set(e)
            # Both supported reducers (integer sum and bitwise OR) compose
            # from the old panel status and the six newly evaluated rays.
            status = status_reducer(jnp.concatenate((state.statuses[chosen, None], s)))
            value = jnp.sum(kw[:, None] * cy, axis=0)
            error = jnp.abs(value - state.values[chosen]) + jnp.sum(kw[:, None] * ce, axis=0)
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
            v, e, s, cy, ce = paired_gauss_panel(
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
                propagated=state.propagated.at[slots].set(ce),
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
            & jnp.all(error <= rtol * jnp.abs(total))
            & (status_reducer(state.statuses) == 0)
        )

    def can_refine(state):
        return jnp.any(eligible_panels(state))

    def keep_refining(carry):
        state, actions, streak = carry
        return can_refine(state) & ((actions < 1) | (streak < 2))

    def refine_and_check(carry):
        state, actions, streak = carry
        state = refine(state)
        streak = jnp.where(allocation_passes(state), streak + 1, 0)
        return state, actions + 1, streak

    state, actions, streak = jax.lax.while_loop(
        keep_refining,
        refine_and_check,
        (state, jnp.int32(0), allocation_passes(state).astype(jnp.int32)),
    )
    converged = (cell_count == 0) | ((actions >= 1) & (streak >= 2))
    h_splits = state.n_live - count
    p_upgrades = (state.n_nodes - 10 * cell_count - 10 * h_splits) // 6
    return AdaptiveMomentIntegral(
        jnp.sum(state.values, axis=0),
        jnp.sum(state.errors, axis=0),
        status_reducer(state.statuses),
        state.n_nodes,
        converged,
        (~converged) & (~can_refine(state)),
        actions,
        streak,
        p_upgrades,
        h_splits,
    )
