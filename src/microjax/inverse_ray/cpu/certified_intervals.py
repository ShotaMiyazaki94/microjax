"""Small, fixed-shape interval certificates for real sextic level sets.

The CPU ICRS kernels use a degree-six polynomial only to obtain candidate
boundary points.  In a nearly multiple-root regime the companion eigensolve
can return a conjugate pair even though the *factored* lens equation has a
small occupied interval.  The routines in this module provide the bounded
repair used by the Cartesian and polar charts:

* the ordinary companion intervals remain the fast path;
* only rejected candidates whose factored level is negative at their real
  centre are repaired;
* dyadic outside brackets and a fixed bisection schedule recover the two
  boundaries; and
* overlapping repaired intervals are compacted to the degree-six maximum of
  three.

No dynamic arrays or data-dependent Python control flow are used.  A caller
supplies a vectorised factored-level evaluator ``stable_level(points)``.  It
is intentionally a Python callback rather than a JAX argument: when this
function is traced, the callback is inlined and remains compatible with JIT,
JVP, and VJP transformations.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

Array = jnp.ndarray

MAX_REAL_INTERVALS = 3
_EXPANSION_STEPS = 10
_BISECTION_STEPS = 28
# A rejected conjugate pair is promoted only when the cancellation-free
# level is negative by more than round-off.  Values in the 1e-10 range are
# common at a tangency and are not evidence of a finite occupied interval;
# promoting them would recreate the broad artificial cell that the ordinary
# no-root safeguard intentionally suppresses.
_CERTIFICATE_NEGATIVE_TOL = 1.0e-8


def _first_true(values: Array) -> tuple[Array, Array]:
    """Return the first true index and whether one exists along the last axis."""

    count = values.shape[-1]
    indices = jnp.arange(count, dtype=jnp.int32)
    found = jnp.any(values, axis=-1)
    first = jnp.min(jnp.where(values, indices, jnp.int32(count)), axis=-1)
    return first, found


def _bisect_left_inside(
    outside: Array,
    inside: Array,
    stable_level: Callable[[Array], Array],
) -> Array:
    """Refine an outside/inside bracket from the left with fixed bisection."""

    def step(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        value = stable_level(mid)
        # Negative means occupied.  Keep ``hi`` on the occupied side.
        next_hi = jnp.where(value <= 0.0, mid, hi)
        next_lo = jnp.where(value <= 0.0, lo, mid)
        return next_lo, next_hi

    _, hi = jax.lax.fori_loop(
        0,
        _BISECTION_STEPS,
        step,
        (outside, inside),
    )
    return hi


def _bisect_right_inside(
    inside: Array,
    outside: Array,
    stable_level: Callable[[Array], Array],
) -> Array:
    """Refine an inside/outside bracket from the right with fixed bisection."""

    def step(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        value = stable_level(mid)
        # Negative means occupied.  Keep ``lo`` on the occupied side.
        next_lo = jnp.where(value <= 0.0, mid, lo)
        next_hi = jnp.where(value <= 0.0, hi, mid)
        return next_lo, next_hi

    lo, _ = jax.lax.fori_loop(
        0,
        _BISECTION_STEPS,
        step,
        (inside, outside),
    )
    return lo


def _merge_intervals(
    lower: Array,
    upper: Array,
    active: Array,
) -> tuple[Array, Array, Array, Array]:
    """Merge fixed-capacity overlapping intervals in sorted order.

    ``lower``, ``upper``, and ``active`` have shape ``(..., n)``.  The output
    has shape ``(..., 3)``.  ``overflow`` is true when more than three
    disjoint intervals were encountered; this is a diagnostic failure, not a
    reason to silently discard an interval.
    """

    batch_shape = lower.shape[:-1]
    n = lower.shape[-1]
    order = jnp.argsort(jnp.where(active, lower, jnp.inf), axis=-1)
    sorted_lower = jnp.take_along_axis(lower, order, axis=-1)
    sorted_upper = jnp.take_along_axis(upper, order, axis=-1)
    sorted_active = jnp.take_along_axis(active, order, axis=-1)
    dtype = lower.dtype
    merged_lower = jnp.zeros(batch_shape + (MAX_REAL_INTERVALS,), dtype=dtype)
    merged_upper = jnp.zeros(batch_shape + (MAX_REAL_INTERVALS,), dtype=dtype)
    merged_active = jnp.zeros(
        batch_shape + (MAX_REAL_INTERVALS,), dtype=bool
    )
    merged_count = jnp.zeros(batch_shape, dtype=jnp.int32)
    overflow = jnp.zeros(batch_shape, dtype=bool)

    def add_one(index, state):
        out_lower, out_upper, out_active, count, overflow = state
        item_lower = sorted_lower[..., index]
        item_upper = sorted_upper[..., index]
        item_active = sorted_active[..., index]
        # ``count`` is the number of already appended intervals; the current
        # interval is therefore compared with slot ``count - 1``.  Using
        # ``count`` here made every overlapping pair look disjoint and could
        # duplicate area when a repaired interval touched an existing one.
        previous_index = jnp.minimum(
            jnp.maximum(count - 1, jnp.int32(0)),
            jnp.int32(MAX_REAL_INTERVALS - 1),
        )
        previous_upper = jnp.take_along_axis(
            out_upper,
            previous_index[..., None],
            axis=-1,
        )[..., 0]
        previous_active = count > 0
        tolerance = 32.0 * jnp.finfo(dtype).eps * jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(item_lower), jnp.abs(previous_upper)),
        )
        overlaps = (
            item_active
            & previous_active
            & (item_lower <= previous_upper + tolerance)
        )
        append = item_active & ~overlaps & (count < MAX_REAL_INTERVALS)
        too_many = item_active & ~overlaps & (count >= MAX_REAL_INTERVALS)
        append_index = jnp.minimum(count, jnp.int32(MAX_REAL_INTERVALS - 1))
        output_slot = jnp.arange(MAX_REAL_INTERVALS, dtype=jnp.int32)
        append_slot = output_slot == append_index[..., None]
        out_lower = jnp.where(
            append[..., None] & append_slot,
            item_lower[..., None],
            out_lower,
        )
        out_upper = jnp.where(
            append[..., None] & append_slot,
            item_upper[..., None],
            out_upper,
        )
        out_active = out_active | (append[..., None] & append_slot)
        merged_index = jnp.minimum(
            jnp.maximum(count - 1, jnp.int32(0)),
            jnp.int32(MAX_REAL_INTERVALS - 1),
        )
        merged_slot = output_slot == merged_index[..., None]
        out_upper = jnp.where(
            overlaps[..., None] & merged_slot,
            jnp.maximum(out_upper, item_upper[..., None]),
            out_upper,
        )
        count = count + append.astype(jnp.int32)
        return out_lower, out_upper, out_active, count, overflow | too_many

    return jax.lax.fori_loop(
        0,
        n,
        add_one,
        (merged_lower, merged_upper, merged_active, merged_count, overflow),
    )


def repair_rejected_intervals(
    roots: Array,
    valid: Array,
    lower_bound: Array,
    upper_bound: Array,
    stable_level: Callable[[Array], Array],
) -> tuple[Array, Array, Array, Array, Array]:
    """Recover occupied intervals represented by rejected root candidates.

    Parameters
    ----------
    roots, valid:
        Candidate roots and the caller's ordinary acceptance mask, with shape
        ``(batch, 6)``.
    lower_bound, upper_bound:
        Domain bounds with shape ``(batch,)``.  The evaluator receives points
        with shape ``(batch, 6)`` and must return the same shape.
    stable_level:
        Cancellation-free mapped level, negative on the source interior.

    Returns
    -------
    lower, upper, active:
        Up to three sorted, merged repaired intervals.
    needed:
        At least one rejected candidate had a negative factored-level centre.
    invalid:
        Number of candidate centres that could not be covered by a repaired
        interval, plus one for capacity overflow.  The caller propagates this
        to its existing diagnostic/status channel.

    The repair is intentionally conservative.  If a candidate does not have
    a stable negative centre it is not promoted to an interval.  Thus a
    genuinely complex pair cannot create area merely because its squared
    polynomial rounded close to zero.
    """

    real = roots.real
    imag = roots.imag
    finite = jnp.isfinite(real) & jnp.isfinite(imag)
    in_domain = finite & (real >= lower_bound[..., None]) & (
        real <= upper_bound[..., None]
    )
    seed = jnp.clip(real, lower_bound[..., None], upper_bound[..., None])
    real_dtype = real.dtype
    scale = jnp.maximum(1.0, jnp.abs(seed))
    # The factored level is the certificate: if it is strictly negative at a
    # rejected real-axis centre, continuity guarantees an occupied interval
    # nearby even when the companion solver reports the pair as complex.  Do
    # not impose an imaginary-part band here; the observed failure occurs
    # precisely when a conjugate pair crosses that arbitrary band under a
    # one-ULP coefficient perturbation.
    candidate = in_domain & ~valid
    eps_floor = 32.0 * jnp.finfo(real_dtype).eps * scale
    distance = jnp.abs(seed[..., :, None] - seed[..., None, :])
    diagonal = jnp.eye(roots.shape[-1], dtype=bool)
    nearest = jnp.min(jnp.where(diagonal, jnp.inf, distance), axis=-1)
    separation_floor = 0.25 * jnp.where(jnp.isfinite(nearest), nearest, scale)
    step = jnp.maximum(jnp.abs(imag), jnp.maximum(eps_floor, separation_floor))
    domain_width = jnp.maximum(upper_bound - lower_bound, eps_floor[..., 0])
    step = jnp.minimum(step, 0.25 * domain_width[..., None])

    seed_level = stable_level(seed)
    # A strict negative sign is the only promotion criterion.  A true
    # complex pair has no occupied centre and therefore remains discarded.
    inside_seed = (
        candidate
        & jnp.isfinite(seed_level)
        & (seed_level < -_CERTIFICATE_NEGATIVE_TOL)
    )
    needed = jnp.any(inside_seed, axis=-1)

    factors = jnp.asarray(2.0 ** jnp.arange(_EXPANSION_STEPS), dtype=real_dtype)
    offsets = step[..., None] * factors
    left_points = jnp.maximum(
        lower_bound[..., None, None], seed[..., None] - offsets
    )
    right_points = jnp.minimum(
        upper_bound[..., None, None], seed[..., None] + offsets
    )
    left_level = stable_level(left_points)
    right_level = stable_level(right_points)
    left_outside = left_level >= 0.0
    right_outside = right_level >= 0.0
    left_index, left_found = _first_true(left_outside)
    right_index, right_found = _first_true(right_outside)
    left_index = jnp.minimum(left_index, jnp.int32(_EXPANSION_STEPS - 1))
    right_index = jnp.minimum(right_index, jnp.int32(_EXPANSION_STEPS - 1))
    left_out = jnp.take_along_axis(left_points, left_index[..., None], axis=-1)[..., 0]
    right_out = jnp.take_along_axis(right_points, right_index[..., None], axis=-1)[..., 0]
    # If an endpoint itself is outside, it is a valid bracket even when the
    # dyadic list clipped repeatedly to that endpoint.
    # The callback is vectorised over the batch already.  Passing an extra
    # singleton axis here would broadcast an ``(batch,)`` angle array against
    # ``(batch, 1)`` radii and produce a spurious ``(batch, batch)`` matrix.
    left_endpoint_level = stable_level(lower_bound)[..., None]
    right_endpoint_level = stable_level(upper_bound)[..., None]
    left_found = left_found | (left_endpoint_level >= 0.0)
    right_found = right_found | (right_endpoint_level >= 0.0)
    left_out = jnp.where(left_found, left_out, lower_bound[..., None])
    right_out = jnp.where(right_found, right_out, upper_bound[..., None])

    # The seed itself is the occupied side of both brackets.  Bisection is
    # run for every lane for static shape; inactive lanes are masked below.
    left_inside = seed
    right_inside = seed
    left_boundary = _bisect_left_inside(
        left_out,
        left_inside,
        stable_level,
    )
    right_boundary = _bisect_right_inside(
        right_inside,
        right_out,
        stable_level,
    )
    interval_active = (
        inside_seed
        & left_found
        & right_found
        & jnp.isfinite(left_boundary)
        & jnp.isfinite(right_boundary)
        & (right_boundary > left_boundary)
    )
    interval_lower, interval_upper, interval_active, _, overflow = _merge_intervals(
        left_boundary,
        right_boundary,
        interval_active,
    )
    covered = inside_seed & jnp.any(
        (seed[..., :, None] >= interval_lower[..., None, :])
        & (seed[..., :, None] <= interval_upper[..., None, :])
        & interval_active[..., None, :],
        axis=-1,
    )
    unresolved = inside_seed & ~covered
    invalid = jnp.sum(unresolved, axis=-1, dtype=jnp.int32) + overflow.astype(
        jnp.int32
    )
    return interval_lower, interval_upper, interval_active, needed, invalid


__all__ = ["MAX_REAL_INTERVALS", "repair_rejected_intervals"]
