"""Endpoint-aware adaptive radial quadrature for inverse ray shooting."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .quadrature_rules import (
    G15_W_ON_GK31,
    GK31_W,
    GK31_X,
    GL23_W,
    GL23_X,
    GL47_W,
    GL47_X,
)
from .radial import RADIAL_OK, RADIAL_TOLERANCE


Array = jnp.ndarray
# Re-evaluate only cells that still miss their allocated error budget.  Each
# entry is the total number of uniform children spanning the original topology
# cell, not an additional split of the previous children.  Keeping this a short
# static schedule preserves JIT shapes while allowing difficult caustic cells to
# converge instead of failing after the historical single two-way split.
_RADIAL_SUBDIVISION_SCHEDULE = (2, 4, 8, 16)
# Kronrod 31-point rule with embedded Gauss 15-point weights. One 31-node
# radial batch supplies a high-order value and an independent error estimate.


class RadialIntegrand(NamedTuple):
    """Integrand value, propagated absolute error, and status bits."""

    value: Array
    error: Array
    status: Array


class RadialIntegral(NamedTuple):
    """Adaptive integral, its absolute error estimate, and status bits."""

    value: Array
    error: Array
    propagated_error: Array
    status: Array


def _quadrature_rule(
    integrand: Callable[[Array], RadialIntegrand],
    lower: Array,
    upper: Array,
    nodes: Array,
    weights: Array,
) -> RadialIntegral:
    """Apply one Gauss rule after a sine-squared endpoint transform."""

    dtype = jnp.asarray(lower).dtype
    x = jnp.asarray(nodes, dtype=dtype)
    weight = jnp.asarray(weights, dtype=dtype)
    angle = 0.25 * jnp.pi * (x + 1.0)
    width = upper - lower
    radii = lower + width * jnp.sin(angle) ** 2
    jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * angle)
    evaluated = jax.vmap(integrand)(radii)
    combined_weight = weight * jacobian
    value = jnp.sum(combined_weight * evaluated.value)
    error = jnp.sum(jnp.abs(combined_weight) * evaluated.error)
    status = jnp.bitwise_or.reduce(evaluated.status)
    return RadialIntegral(value, error, error, status)


def _integrate_cell(
    integrand: Callable[[Array], RadialIntegrand], lower: Array, upper: Array
) -> RadialIntegral:
    dtype = jnp.asarray(lower).dtype
    x = jnp.asarray(GK31_X, dtype=dtype)
    kronrod_weight = jnp.asarray(GK31_W, dtype=dtype)
    gauss_weight = jnp.asarray(G15_W_ON_GK31, dtype=dtype)
    angle = 0.25 * jnp.pi * (x + 1.0)
    width = upper - lower
    radii = lower + width * jnp.sin(angle) ** 2
    jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * angle)
    evaluated = jax.vmap(integrand)(radii)
    kronrod_combined_weight = kronrod_weight * jacobian
    gauss_combined_weight = gauss_weight * jacobian
    kronrod_value = jnp.sum(kronrod_combined_weight * evaluated.value)
    gauss_value = jnp.sum(gauss_combined_weight * evaluated.value)
    propagated_error = jnp.sum(
        jnp.abs(kronrod_combined_weight) * evaluated.error
    )
    embedded_error = jnp.abs(kronrod_value - gauss_value)
    error = propagated_error + embedded_error
    finite = jnp.isfinite(kronrod_value) & jnp.isfinite(error)
    status = jnp.bitwise_or(
        jnp.bitwise_or.reduce(evaluated.status),
        jnp.where(finite, jnp.int32(RADIAL_OK), jnp.int32(RADIAL_TOLERANCE)),
    )
    return RadialIntegral(kronrod_value, error, propagated_error, status)


def _integrate_cell_gl47(
    integrand: Callable[[Array], RadialIntegrand], lower: Array, upper: Array
) -> RadialIntegral:
    """Integrate one unsplit cell with independent G23/G47 rules."""

    coarse = _quadrature_rule(integrand, lower, upper, GL23_X, GL23_W)
    fine = _quadrature_rule(integrand, lower, upper, GL47_X, GL47_W)
    error = fine.propagated_error + jnp.abs(fine.value - coarse.value)
    finite = jnp.isfinite(fine.value) & jnp.isfinite(error)
    status = jnp.bitwise_or(
        jnp.bitwise_or(coarse.status, fine.status),
        jnp.where(
            finite, jnp.int32(RADIAL_OK), jnp.int32(RADIAL_TOLERANCE)
        ),
    )
    return RadialIntegral(fine.value, error, fine.propagated_error, status)


def _refine_cell_uniform(
    integrand: Callable[[Array], RadialIntegrand],
    lower: Array,
    upper: Array,
    subdivisions: int,
) -> RadialIntegral:
    """Uniformly subdivide and re-integrate one failed cell."""

    edges = jnp.linspace(lower, upper, subdivisions + 1)
    children = jax.vmap(
        lambda lo, hi: _integrate_cell(integrand, lo, hi)
    )(edges[:-1], edges[1:])
    value = jnp.sum(children.value)
    propagated_error = jnp.sum(children.propagated_error)
    error = jnp.sum(children.error)
    status = jnp.bitwise_or.reduce(children.status)
    return RadialIntegral(value, error, propagated_error, status)


def _chunked_refined_cells(
    integrand: Callable[[Array], RadialIntegrand],
    intervals: Array,
    n_active: Array,
    chunk_size: int,
    subdivisions: int,
) -> RadialIntegral:
    """Evaluate a compact prefix of failed cells without scalar GPU maps."""

    capacity = intervals.shape[0]
    pad = (-capacity) % chunk_size
    interval_chunks = jnp.pad(intervals, ((0, pad), (0, 0))).reshape(
        -1, chunk_size, 2
    )
    starts = jnp.arange(interval_chunks.shape[0], dtype=jnp.int32) * chunk_size
    dtype = intervals.dtype
    offsets = jnp.arange(chunk_size, dtype=jnp.int32)

    def evaluate(inputs):
        start, bounds = inputs

        def active(cell_bounds):
            slot_active = start + offsets < n_active
            safe_bounds = jnp.where(
                slot_active[:, None], cell_bounds, cell_bounds[0]
            )
            evaluated = jax.vmap(
                lambda interval: _refine_cell_uniform(
                    integrand,
                    interval[0],
                    interval[1],
                    subdivisions,
                )
            )(safe_bounds)
            return RadialIntegral(
                jnp.where(slot_active, evaluated.value, 0.0),
                jnp.where(slot_active, evaluated.error, 0.0),
                jnp.where(slot_active, evaluated.propagated_error, 0.0),
                jnp.where(slot_active, evaluated.status, jnp.int32(0)),
            )

        def inactive(_):
            return RadialIntegral(
                jnp.zeros(chunk_size, dtype=dtype),
                jnp.zeros(chunk_size, dtype=dtype),
                jnp.zeros(chunk_size, dtype=dtype),
                jnp.zeros(chunk_size, dtype=jnp.int32),
            )

        return jax.lax.cond(
            start < n_active, active, inactive, bounds
        )

    result = jax.lax.map(
        evaluate,
        (starts, interval_chunks),
    )
    return RadialIntegral(
        result.value.reshape(-1)[:capacity],
        result.error.reshape(-1)[:capacity],
        result.propagated_error.reshape(-1)[:capacity],
        result.status.reshape(-1)[:capacity],
    )


def fixed_radial_integral(
    integrand: Callable[[Array], RadialIntegrand],
    intervals: Array,
    n_intervals: Array,
    absolute_tolerance: Array,
    *,
    relative_tolerance: Array = 0.0,
    initial_status: Array = 0,
    chunk_size: int = 16,
    subdivisions: int = 4,
    single_cell_order: int = 31,
) -> RadialIntegral:
    """Integrate every active topology cell once on a fixed fine radial mesh.

    Each original topology interval is split into ``subdivisions`` equal cells;
    every child normally uses the endpoint-transformed embedded G15/K31 rule.
    For one unsplit cell, ``single_cell_order=47`` instead uses independent
    G23/G47 rules. Unlike
    :func:`adaptive_radial_integral`, this path performs no compact/scatter
    refinement and never recomputes an original interval at successively finer
    depths.  It is intended for accelerator workloads where a single regular
    batch is cheaper than divergent bounded retries.

    The embedded-rule difference remains an empirical error estimator, not a
    formal upper bound. Failure of the final public tolerance remains visible
    through ``RADIAL_TOLERANCE``.
    """

    if not 1 <= subdivisions <= 16:
        raise ValueError("subdivisions must be between 1 and 16")
    if single_cell_order not in (31, 47):
        raise ValueError("single_cell_order must be 31 or 47")
    if subdivisions == 1:
        cells = _chunked_initial_cells(
            integrand,
            intervals,
            n_intervals,
            chunk_size,
            high_order=single_cell_order == 47,
        )
    else:
        cells = _chunked_refined_cells(
            integrand,
            intervals,
            n_intervals,
            chunk_size,
            subdivisions,
        )

    active = jnp.arange(intervals.shape[0]) < n_intervals
    value = jnp.sum(jnp.where(active, cells.value, 0.0))
    error = jnp.sum(jnp.where(active, cells.error, 0.0))
    propagated_error = jnp.sum(
        jnp.where(active, cells.propagated_error, 0.0)
    )
    status = jnp.bitwise_or(
        jnp.asarray(initial_status, dtype=jnp.int32),
        jnp.bitwise_or.reduce(jnp.where(active, cells.status, 0)),
    )
    tolerance = absolute_tolerance + relative_tolerance * jnp.abs(value)
    status = jnp.bitwise_or(
        status,
        jnp.where(
            jnp.isfinite(value) & jnp.isfinite(error) & (error <= tolerance),
            jnp.int32(RADIAL_OK),
            jnp.int32(RADIAL_TOLERANCE),
        ),
    )
    return RadialIntegral(value, error, propagated_error, status)


def _chunked_initial_cells(
    integrand: Callable[[Array], RadialIntegrand],
    intervals: Array,
    n_active: Array,
    chunk_size: int,
    high_order: bool = False,
) -> RadialIntegral:
    """Vectorize active topology cells while skipping whole inactive chunks."""

    capacity = intervals.shape[0]
    pad = (-capacity) % chunk_size
    chunks = jnp.pad(intervals, ((0, pad), (0, 0))).reshape(-1, chunk_size, 2)
    starts = jnp.arange(chunks.shape[0], dtype=jnp.int32) * chunk_size
    offsets = jnp.arange(chunk_size, dtype=jnp.int32)
    dtype = intervals.dtype

    def evaluate(inputs):
        start, chunk = inputs

        def active(values):
            slot_active = start + offsets < n_active
            # A partially active chunk used to evaluate padded ``[0, 0]``
            # cells and mask their results afterwards.  Their zero radial node
            # can make the angular polynomial degenerate; reverse mode then
            # encounters ``0 * NaN`` even though the slot has zero cotangent.
            # Reusing the first valid bounds keeps the whole chunk vectorized
            # while ensuring every deliberately over-computed cell is regular.
            safe_values = jnp.where(
                slot_active[:, None], values, values[0]
            )
            integrate_cell = (
                _integrate_cell_gl47 if high_order else _integrate_cell
            )
            evaluated = jax.vmap(
                lambda bounds: integrate_cell(integrand, *bounds)
            )(safe_values)
            return RadialIntegral(
                jnp.where(slot_active, evaluated.value, 0.0),
                jnp.where(slot_active, evaluated.error, 0.0),
                jnp.where(slot_active, evaluated.propagated_error, 0.0),
                jnp.where(slot_active, evaluated.status, jnp.int32(0)),
            )

        def inactive(_):
            return RadialIntegral(
                jnp.zeros(chunk_size, dtype=dtype),
                jnp.zeros(chunk_size, dtype=dtype),
                jnp.zeros(chunk_size, dtype=dtype),
                jnp.zeros(chunk_size, dtype=jnp.int32),
            )

        return jax.lax.cond(start < n_active, active, inactive, chunk)

    result = jax.lax.map(evaluate, (starts, chunks))
    return RadialIntegral(
        result.value.reshape(-1)[:capacity],
        result.error.reshape(-1)[:capacity],
        result.propagated_error.reshape(-1)[:capacity],
        result.status.reshape(-1)[:capacity],
    )


def adaptive_radial_integral(
    integrand: Callable[[Array], RadialIntegrand],
    intervals: Array,
    n_intervals: Array,
    absolute_tolerance: Array,
    *,
    relative_tolerance: Array = 0.0,
    initial_status: Array = 0,
    chunk_size: int = 16,
    max_subdivisions: int = 8,
) -> RadialIntegral:
    """Integrate fixed-shape radial topology with embedded error control.

    The sine-squared map removes the square-root behaviour at radial topology
    changes. Embedded G15/K31 disagreement supplies an error estimate; failing
    cells are compacted and refined as parallel accelerator batches. Cells
    that still miss ``absolute_tolerance + relative_tolerance * abs(value)``
    report ``RADIAL_TOLERANCE`` so the caller can use its bounded retry/reject
    instead of silently returning an under-resolved value.  The default zero
    relative tolerance preserves the historical absolute-only contract.
    ``max_subdivisions`` is a static bounded-adaptivity control. Supported
    values are 1 (no subdivision), 2, 4, 8, and 16.
    """

    if max_subdivisions not in (1, 2, 4, 8, 16):
        raise ValueError(
            "max_subdivisions must be one of 1, 2, 4, 8, or 16"
        )

    initial = _chunked_initial_cells(
        integrand, intervals, n_intervals, chunk_size
    )
    active = jnp.arange(intervals.shape[0]) < n_intervals
    widths = jnp.where(active, intervals[:, 1] - intervals[:, 0], 0.0)
    total_width = jnp.maximum(jnp.sum(widths), jnp.finfo(intervals.dtype).tiny)
    # A failed angular root set cannot be repaired by radial subdivision and is
    # left for the caller's higher-level retry/reject path. Numerically valid
    # cells that miss their allocated budget are compacted and refined together,
    # retaining GPU parallelism instead of entering one scalar recursive map per
    # cell.
    def replace_failed(current, needs_refinement, subdivisions):
        n_refine = jnp.sum(needs_refinement, dtype=jnp.int32)
        sentinel = intervals.shape[0]
        refine_indices = jnp.nonzero(
            needs_refinement,
            size=intervals.shape[0],
            fill_value=sentinel,
        )[0]
        safe_indices = jnp.where(refine_indices < sentinel, refine_indices, 0)
        compact_intervals = intervals[safe_indices]
        compact_refined = _chunked_refined_cells(
            integrand,
            compact_intervals,
            n_refine,
            chunk_size,
            subdivisions,
        )
        extra_float = jnp.zeros(1, intervals.dtype)
        extra_status = jnp.zeros(1, dtype=jnp.int32)
        value = jnp.concatenate((current.value, extra_float))
        error = jnp.concatenate((current.error, extra_float))
        propagated = jnp.concatenate((current.propagated_error, extra_float))
        status = jnp.concatenate((current.status, extra_status))
        value = value.at[refine_indices].set(compact_refined.value)
        error = error.at[refine_indices].set(compact_refined.error)
        propagated = propagated.at[refine_indices].set(
            compact_refined.propagated_error
        )
        status = status.at[refine_indices].set(compact_refined.status)
        return RadialIntegral(
            value[:-1], error[:-1], propagated[:-1], status[:-1]
        )

    def cells_needing_refinement(current):
        """Select local failures only while the global error contract fails."""

        value = jnp.sum(jnp.where(active, current.value, 0.0))
        error = jnp.sum(jnp.where(active, current.error, 0.0))
        status = jnp.bitwise_or(
            jnp.asarray(initial_status, dtype=jnp.int32),
            jnp.bitwise_or.reduce(jnp.where(active, current.status, 0)),
        )
        tolerance = absolute_tolerance + relative_tolerance * jnp.abs(value)
        global_failure = (
            (status == RADIAL_OK)
            & jnp.isfinite(error)
            & (error > tolerance)
        )
        local_tolerance = tolerance * widths / total_width
        return (
            global_failure
            & active
            & (current.status == RADIAL_OK)
            & (current.error > local_tolerance)
        )

    needs_refinement = cells_needing_refinement(initial)
    refined = initial
    for subdivisions in _RADIAL_SUBDIVISION_SCHEDULE:
        if subdivisions > max_subdivisions:
            break
        refined = replace_failed(refined, needs_refinement, subdivisions)
        needs_refinement = cells_needing_refinement(refined)
    value = jnp.sum(jnp.where(active, refined.value, 0.0))
    error = jnp.sum(jnp.where(active, refined.error, 0.0))
    propagated_error = jnp.sum(
        jnp.where(active, refined.propagated_error, 0.0)
    )
    status = jnp.bitwise_or(
        jnp.asarray(initial_status, dtype=jnp.int32),
        jnp.bitwise_or.reduce(jnp.where(active, refined.status, 0)),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            jnp.isfinite(error)
            & (
                error
                <= absolute_tolerance + relative_tolerance * jnp.abs(value)
            ),
            jnp.int32(RADIAL_OK),
            jnp.int32(RADIAL_TOLERANCE),
        ),
    )
    return RadialIntegral(value, error, propagated_error, status)
