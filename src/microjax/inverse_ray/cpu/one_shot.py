"""State-conditioned one-shot CPU ICRS schedulers.

The routines in this module deliberately separate *routing* from *rescue*.
One source-limb trace chooses Cartesian (with an x/y axis choice) or polar
theta-first. Uniform and linear limb darkening use the same two image moments;
only the final brightness-weighted normalization depends on the coefficient.
Both paths reuse GL5 samples in GK11 panels, with empirical relative stopping
inside a bounded ray budget. There is no tolerance-based chart
retry. Non-zero status reports only a detected structural failure; it never
triggers another chart, order, or trace.

This makes the steady-state work auditable and keeps the forward-mode JAX
graph substantially smaller than the historical adaptive retry ladder.
The older uniform-only three-chart helper remains for explicit comparisons;
it is not called by the default CPU scheduler.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .angular_limb_dark import _mag_limb_dark_angular_moment_from_support
from .angular_moment import (
    ANGULAR_MOMENT_SUPPORT,
    ANGULAR_MOMENT_TOPOLOGY,
    _angular_trace_edge_consistent,
    _angular_support_cells_from_trace,
    _split_angular_support_at_angle,
    _split_overlapping_angular_support,
    _uniform_result_from_support,
)
from .angular_projection_support import _angular_support_cells_from_events
from .cartesian_limb_dark import _mag_limb_dark_cartesian_impl
from .cartesian_moment import (
    CartesianAdaptiveResult,
    _cartesian_support_1d,
    _cartesian_result_from_support,
    _cartesian_support_cells_from_trace,
)
from .radial_continuation import _mag_uniform_radial_from_trace
from .projection_support import _cartesian_support_cells_from_events
from .projection_moments import integrate_adaptive_moments
from .support import (
    LimbTrace,
    Support1D,
    trace_binary_source_limb_record,
    unwrap_support_1d,
)

Array = jnp.ndarray

# Both profile charts share one cached two-moment rule. The budget is shared across
# the support, rather than imposing an independent cap on every active cell.
_LD_ADAPTIVE_BUDGET = 84
_LD_ADAPTIVE_RTOL = 1e-4
_integrate_ld_moments = partial(
    integrate_adaptive_moments, budget=_LD_ADAPTIVE_BUDGET, rtol=_LD_ADAPTIVE_RTOL
)

# Public stage labels.  Cartesian x/y are an axis choice inside one integral,
# so expose one Cartesian stage rather than making the diagnostic look like a
# fourth numerical route.
ONE_SHOT_CARTESIAN = 1
ONE_SHOT_CARTESIAN_X_FIRST = ONE_SHOT_CARTESIAN
ONE_SHOT_CARTESIAN_Y_FIRST = ONE_SHOT_CARTESIAN
ONE_SHOT_POLAR_RADIAL_FIRST = 3
ONE_SHOT_POLAR_THETA_FIRST = 4

# Structural status bits specific to the one-shot scheduler. Existing
# support/topology bits are inherited from the fixed-order kernels.
ONE_SHOT_INVALID_ROOTS = 1 << 20
ONE_SHOT_NONFINITE = 1 << 21
ONE_SHOT_UNRESOLVED_GEOMETRY = 1 << 22

# Cartesian x/y are coordinate choices of one kernel, not separate numerical
# routes. The legacy uniform-only helper retains these three route labels.
# The default common-profile solver compiles only Cartesian and theta-first.
_CPU_ROUTE_CARTESIAN = 0
_CPU_ROUTE_CARTESIAN_X_FIRST = _CPU_ROUTE_CARTESIAN
_CPU_ROUTE_CARTESIAN_Y_FIRST = _CPU_ROUTE_CARTESIAN
_CPU_ROUTE_POLAR_RADIAL_FIRST = 1
_CPU_ROUTE_POLAR_THETA_FIRST = 2
_CPU_ROUTE_COUNT = 3

class OneShotState(NamedTuple):
    """Axis-independent state measured from the single source-limb trace."""

    topology_uncertain: Array
    limb_topology: Array
    edge_active: Array
    ghost_residual_ratio: Array
    ghost_image_angle: Array
    transition_image_angle: Array
    transition_image_angle_2: Array
    polar_conditioning: Array
    maximum_angular_step: Array
    active_images_min: Array
    active_images_max: Array


class OneShotGeometryPrepared(NamedTuple):
    """Trace, support, and chart selected from the allowed kernel set."""

    image_limb: Array
    physical_mask: Array
    previous_limb: Array
    following_limb: Array
    previous_mask: Array
    following_mask: Array
    state: OneShotState
    primary_axis: Array
    primary_support: tuple[Array, Array, Array, Array, Array]
    primary_support_1d: Support1D
    partition_theta: Array
    route: Array
    limb_trace: LimbTrace


def _closed_angular_extrema_parity_valid(
    cells: Array,
    limb_topology: Array,
) -> Array:
    """Require paired angular extrema when every image branch is closed.

    Without a source-limb image birth or death, each closed image loop has
    paired angular minima and maxima.  The sorted extrema therefore produce
    an odd number of nonempty angular cells.  An even count exposes a missed
    or duplicated tangency before either quadrature rule can silently agree
    on the same incomplete radial moment.
    """

    # ``AngularSupport.active`` is now the trace-sector union mask, not
    # the number of cells in the extrema partition.  Parity belongs to the
    # partition itself and therefore counts every nonempty atomic cell.
    cell_count = jnp.sum(cells[:, 1] > cells[:, 0], dtype=jnp.int32)
    return jnp.asarray(limb_topology) | ((cell_count & jnp.int32(1)) == 1)


def _trace_transition_support_valid(state: OneShotState) -> Array:
    """Validate a topology transition from the trace's structural fields.

    A 3-to-5 transition is already represented conservatively by the trace:
    same-multiplicity edges are kept, the transition edges are cut, and the
    angular support falls back to its complete partition when topology is
    uncertain.  The previous implementation added several empirical bands in
    ghost residual and conditioning that rejected otherwise finite,
    structurally valid traces.  Those bands made the diagnostic status depend
    on arbitrary numerical thresholds rather than on the trace itself.
    """

    # ``ghost_residual_ratio == inf`` is the normal, well-conditioned state:
    # there is no non-physical root slot to measure.  Require a finite ghost
    # residual only when the trace itself is topology-uncertain and therefore
    # actually relies on that diagnostic.
    ghost_valid = (~state.topology_uncertain) | jnp.isfinite(
        state.ghost_residual_ratio
    )
    metrics_finite = (
        ghost_valid
        & jnp.isfinite(state.polar_conditioning)
        & jnp.isfinite(state.maximum_angular_step)
    )
    return metrics_finite


def _flag_unresolved_trace_transition(
    result: CartesianAdaptiveResult,
    state: OneShotState,
) -> CartesianAdaptiveResult:
    """Preserve a trace-only structural warning across the supported charts."""

    buried_unresolved = (
        (state.active_images_min == 3)
        & (state.active_images_max == 3)
        & (state.ghost_residual_ratio >= 100.0)
        & (state.ghost_residual_ratio < 1000.0)
        & (state.polar_conditioning >= 1000.0)
    )
    trace_valid = _trace_transition_support_valid(state) & ~buried_unresolved
    return result._replace(
        status=jnp.bitwise_or(
            result.status,
            jnp.where(
                trace_valid,
                jnp.int32(0),
                jnp.int32(ONE_SHOT_UNRESOLVED_GEOMETRY),
            ),
        )
    )


def _trace_state(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_limb: int,
):
    """Trace once and extract the state used by both brightness profiles."""

    trace = trace_binary_source_limb_record(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        include_all_roots=False,
    )
    image_limb = trace.image_limb
    physical_mask = trace.physical_mask
    image_counts = trace.image_counts
    limb_topology = jnp.any(image_counts != image_counts[0])
    residual = trace.residual
    ghost_residuals = jnp.where(~physical_mask, residual, jnp.inf)
    ghost = jnp.min(ghost_residuals)
    ghost_index = jnp.argmin(ghost_residuals.reshape(-1))
    ghost_image = image_limb.reshape(-1)[ghost_index]
    # ``argmin(inf)`` is arbitrary when no non-physical root exists.  Never
    # turn that sentinel into a real angular split: a ghost guard is valid
    # only when the trace contains a finite, rejected root.
    ghost_image_angle = jnp.where(
        jnp.isfinite(ghost),
        jnp.mod(jnp.angle(ghost_image), 2.0 * jnp.pi),
        jnp.asarray(0.0, dtype=w_center.real.dtype),
    )
    # A 3->5 transition has two newborn/dead roots at the phase where the
    # image count changes.  Keep the two smallest-residual physical roots at
    # that transition as angular guard seeds.  This is derived entirely from
    # the canonical trace; it does not solve the lens equation again and does
    # not reconnect the cut edge.
    transition_columns = (
        (image_counts != jnp.roll(image_counts, 1))
        | (image_counts != jnp.roll(image_counts, -1))
    )
    transition_residuals = jnp.where(
        physical_mask & transition_columns[None, :],
        residual,
        jnp.inf,
    )
    transition_index = jnp.argmin(transition_residuals.reshape(-1))
    transition_first = transition_residuals.reshape(-1)[transition_index]
    transition_remaining = transition_residuals.reshape(-1).at[transition_index].set(
        jnp.inf
    )
    transition_index_2 = jnp.argmin(transition_remaining)
    transition_second = transition_remaining[transition_index_2]
    transition_flat_limb = image_limb.reshape(-1)
    transition_angle = jnp.where(
        jnp.isfinite(transition_first),
        jnp.mod(jnp.angle(transition_flat_limb[transition_index]), 2.0 * jnp.pi),
        jnp.asarray(0.0, dtype=w_center.real.dtype),
    )
    transition_angle_2 = jnp.where(
        jnp.isfinite(transition_second),
        jnp.mod(jnp.angle(transition_flat_limb[transition_index_2]), 2.0 * jnp.pi),
        jnp.asarray(0.0, dtype=w_center.real.dtype),
    )
    topology = (
        limb_topology
        | (ghost <= 4.0 * rho)
        | ~jnp.all(jnp.isfinite(jnp.where(physical_mask, image_limb, 0.0j)))
    )

    following_limb = trace.following_limb
    outgoing_edge_active = trace.outgoing_edge_active
    angular_step = trace.angular_step
    tangential_motion = jnp.sum(
        jnp.where(
            outgoing_edge_active,
            0.5
            * (jnp.abs(image_limb) + jnp.abs(following_limb))
            * angular_step,
            0.0,
        )
    )
    radial_motion = jnp.sum(
        jnp.where(outgoing_edge_active, trace.radial_step, 0.0)
    )
    motion_floor = 128.0 * jnp.finfo(w_center.real.dtype).eps
    polar_conditioning = tangential_motion / jnp.maximum(radial_motion, motion_floor)
    state = OneShotState(
        topology,
        limb_topology,
        outgoing_edge_active,
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        ghost_image_angle,
        transition_angle,
        transition_angle_2,
        polar_conditioning,
        jnp.max(jnp.where(outgoing_edge_active, angular_step, 0.0)),
        jnp.min(image_counts),
        jnp.max(image_counts),
    )
    return trace, state


def _fixed_cartesian_result(result, *, stage: int) -> CartesianAdaptiveResult:
    """Expose one fixed Cartesian value with structural diagnostics only."""

    status = jnp.bitwise_or(
        result.status,
        jnp.where(
            result.invalid_root_count == 0,
            jnp.int32(0),
            jnp.int32(ONE_SHOT_INVALID_ROOTS),
        ),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            jnp.isfinite(result.magnification),
            jnp.int32(0),
            jnp.int32(ONE_SHOT_NONFINITE),
        ),
    )
    return CartesianAdaptiveResult(
        result.magnification,
        jnp.asarray(jnp.nan, dtype=result.magnification.dtype),
        result.n_slices,
        jnp.int32(stage),
        status,
    )


def _cartesian_stage(axis: Array) -> Array:
    """Return the single public label for the shared Cartesian kernel."""

    del axis
    return jnp.int32(ONE_SHOT_CARTESIAN)


def _fixed_polar_result(
    result,
    *,
    stage: int,
    support_valid: Array | bool,
    replace_support_status: Array | bool = False,
) -> CartesianAdaptiveResult:
    """Expose one fixed polar value with structural diagnostics only."""

    # Fixed polar kernels use bit 0 for a non-finite reduction. Normalize that
    # private bit into the public ONE_SHOT_NONFINITE flag below.
    soft = jnp.bitwise_or(jnp.int32(1), jnp.int32(ANGULAR_MOMENT_TOPOLOGY))
    soft = jnp.bitwise_or(
        soft,
        jnp.where(
            jnp.asarray(replace_support_status),
            jnp.int32(ANGULAR_MOMENT_SUPPORT),
            jnp.int32(0),
        ),
    )
    status = jnp.bitwise_and(result.status, jnp.bitwise_not(soft))
    status = jnp.bitwise_or(
        status,
        jnp.where(
            jnp.asarray(support_valid),
            jnp.int32(0),
            jnp.int32(ANGULAR_MOMENT_SUPPORT),
        ),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            result.invalid_root_count == 0,
            jnp.int32(0),
            jnp.int32(ONE_SHOT_INVALID_ROOTS),
        ),
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            jnp.isfinite(result.magnification),
            jnp.int32(0),
            jnp.int32(ONE_SHOT_NONFINITE),
        ),
    )
    return CartesianAdaptiveResult(
        result.magnification,
        jnp.asarray(jnp.nan, dtype=result.magnification.dtype),
        result.n_theta,
        jnp.int32(stage),
        status,
    )


def _prepare_one_shot_geometry(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_limb: int,
    external_available: bool,
    cartesian_support_builder: Callable = _cartesian_support_cells_from_trace,
    cartesian_support_size: int = 19,
    allow_radial: bool = True,
) -> OneShotGeometryPrepared:
    """Trace and route one source without evaluating a brightness moment.

    The support builder, capacity, and allowed charts are static choices for
    the brightness algorithm. Uniform keeps its native radial kernel; LD
    compiles only Cartesian/theta. No tolerance-based retry is introduced.
    """

    del external_available
    limb_trace, state = _trace_state(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
    )
    image_limb = limb_trace.image_limb
    physical_mask = limb_trace.physical_mask
    neighbors = (
        limb_trace.previous_limb,
        limb_trace.following_limb,
        limb_trace.previous_mask,
        limb_trace.following_mask,
    )
    edge_activity = (
        limb_trace.incoming_edge_active,
        limb_trace.outgoing_edge_active,
    )
    incoming_edge_active, outgoing_edge_active = edge_activity
    previous_limb, following_limb, previous_mask, following_mask = neighbors
    image_counts = limb_trace.image_counts
    incoming_same_count = image_counts == jnp.roll(image_counts, 1)
    outgoing_same_count = image_counts == jnp.roll(image_counts, -1)
    explicit_same_count_cut = (
        (~incoming_edge_active | ~outgoing_edge_active)
        & (incoming_same_count & outgoing_same_count)[None, :]
        & previous_mask
        & following_mask
    )

    def projection_diagnostics(axis):
        projected = jnp.real(image_limb * jnp.conjugate(axis))
        projected_previous = jnp.real(previous_limb * jnp.conjugate(axis))
        projected_following = jnp.real(following_limb * jnp.conjugate(axis))
        # A cut edge must not be used to connect two trace segments, but a
        # sampled turn at the cut is still a conservative support endpoint.
        # Omitting it drops the entire strip between the preceding extremum
        # and a fold transition (the dominant 3->5 Cartesian failure mode).
        # The builder keeps the cut endpoint at its sampled coordinate rather
        # than applying a cross-edge fit, so this does not reconnect topology.
        projected_turn = (
            (projected - projected_previous)
            * (projected_following - projected)
            <= 0.0
        )
        projected_endpoint = physical_mask & (
            ~previous_mask | ~following_mask | explicit_same_count_cut
        )
        extrema = jnp.sum(
            physical_mask & (projected_turn | projected_endpoint),
            dtype=jnp.int32,
        )
        lower = jnp.min(jnp.where(physical_mask, projected, jnp.inf))
        upper = jnp.max(jnp.where(physical_mask, projected, -jnp.inf))
        scale = jnp.maximum(
            upper - lower,
            128.0 * jnp.finfo(w_center.real.dtype).eps,
        )
        maximum_step = jnp.max(
            jnp.where(
                outgoing_edge_active,
                jnp.abs(projected_following - projected),
                0.0,
            )
        ) / scale
        trace = (
            projected,
            projected_previous,
            projected_following,
            incoming_edge_active,
            outgoing_edge_active,
        )
        return extrema, maximum_step, trace

    x_axis = jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype)
    y_axis = jnp.asarray(0.0 + 1.0j, dtype=w_center.dtype)
    x_extrema, x_step, x_trace = projection_diagnostics(x_axis)
    y_extrema, y_step, y_trace = projection_diagnostics(y_axis)
    # Every resolved projection of a closed image loop has paired extrema.
    # Prefer y when x violates that parity and y does not.  If parity ties,
    # compare each count with the generic two extrema per closed image loop;
    # this includes the common four-x versus six-y buried-loop signature.
    # These are structural certificates; raw edge-size preferences caused
    # broad regressions and are deliberately not used for the axis choice.
    closed_trace = (
        ~state.limb_topology
        & (state.active_images_min == 3)
        & (state.active_images_max == 3)
    )
    expected_extrema = jnp.int32(2) * state.active_images_min
    x_parity_bad = (x_extrema & jnp.int32(1)) == 1
    y_parity_bad = (y_extrema & jnp.int32(1)) == 1
    use_y_first = closed_trace & (
        (
            x_parity_bad
            & ~y_parity_bad
        )
        | (
            (x_parity_bad == y_parity_bad)
            & (
                jnp.abs(y_extrema - expected_extrema)
                < jnp.abs(x_extrema - expected_extrema)
            )
        )
    )
    primary_axis = jnp.where(use_y_first, y_axis, x_axis)
    projected_extrema = jnp.where(use_y_first, y_extrema, x_extrema)
    projection_trace = jax.tree.map(
        lambda y_value, x_value: jnp.where(use_y_first, y_value, x_value),
        y_trace,
        x_trace,
    )

    # A closed projected image loop has paired extrema.  An odd number of
    # sampled turns together with a large traced angular jump identifies a
    # branch whose narrow turning sector is poorly represented by Cartesian
    # nodes. Route that trace to the radial-first polar integral.
    source_phase_step = 2.0 * jnp.pi / jnp.asarray(
        n_limb, dtype=w_center.real.dtype
    )
    underresolved_projected_turn = (
        ((projected_extrema & jnp.int32(1)) == 1)
        & (state.maximum_angular_step > 0.3 * source_phase_step)
    )
    # Compare the polar trace quality with the worst Cartesian axis using
    # dimensionless, trace-derived quantities.  ``polar_conditioning`` is
    # the tangential/radial motion ratio; multiplying by the normalized
    # angular edge gives the amount of unresolved image motion represented by
    # one source-limb sample.  The Cartesian counterpart is the largest
    # normalized projected edge.  This is a relative chart comparison, not a
    # parameter-space threshold.
    polar_motion_score = state.polar_conditioning * (
        state.maximum_angular_step / jnp.maximum(source_phase_step, 1.0e-30)
    )
    # For closed traces compare against the worst projected gap: unlike a
    # transition, the polar chart has no cut-edge guard and must beat both
    # Cartesian projections before it is selected.
    cartesian_motion_score = jnp.maximum(x_step, y_step)
    # A transition stays Cartesian when the trace is densely sampled around
    # the fold.  If the ordered image motion contains a resolved angular gap
    # and the polar motion is not effectively radial-free, the polar support
    # is the safer chart.  Both witnesses are measured from the trace; the
    # conditioning floor rejects the numerically singular zero-radial-motion
    # case without looking at (q, s, rho) parameter regions.
    cartesian_selected_score = jnp.where(use_y_first, y_step, x_step)
    transition_chart_better = (
        state.limb_topology
        & (state.active_images_min == 3)
        & (state.active_images_max == 5)
        & (
            state.maximum_angular_step
            > 0.004 * source_phase_step
        )
        & (state.polar_conditioning > 1.0)
        & (polar_motion_score > cartesian_selected_score)
    )
    common = dict(
        w_center=w_center,
        rho=rho,
        s=s,
        q=q,
        # The historical uniform-support builder uses sampled envelopes.
        # The injected LD event builder uses its common fixed Newton schedule
        # and ignores this legacy refinement option.
        refine_extrema=False,
        topology_uncertain=state.topology_uncertain,
        minimum_ghost_residual=state.ghost_residual_ratio * rho,
        limb_topology=state.limb_topology,
        maximum_extrema=20,
        neighbors=neighbors,
        edge_activity=edge_activity,
    )

    # All route predicates below are measurements of the image trace. No
    # region in (q, s, rho, w) selects a support or quadrature chart.
    closed_conditioned = (
        ~state.topology_uncertain
        & ~state.limb_topology
        & (polar_motion_score > cartesian_motion_score)
    )
    conditioned_trace = (
        transition_chart_better
        | closed_conditioned
        | (
            state.topology_uncertain
            & ~state.limb_topology
        )
        | (~state.limb_topology & underresolved_projected_turn)
    ) & (state.maximum_angular_step > 0.004 * source_phase_step)
    radial_first_trace = (
        allow_radial
        & ~state.topology_uncertain
        & ~state.limb_topology
        & (state.active_images_min == 5)
        & (state.polar_conditioning >= 10.0)
    )
    resolved_closed_three_cartesian = (
        ~state.topology_uncertain
        & ~state.limb_topology
        & (state.active_images_min == 3)
        & (state.active_images_max == 3)
        & (cartesian_motion_score <= 4.0e-4)
    )
    theta_first_trace = (
        transition_chart_better
        | (conditioned_trace & ~radial_first_trace)
    ) & ~resolved_closed_three_cartesian
    # x/y is an axis choice inside the shared Cartesian kernel, not a route.
    cartesian_route = jnp.int32(_CPU_ROUTE_CARTESIAN)
    standard_route = jnp.where(
        theta_first_trace,
        jnp.int32(_CPU_ROUTE_POLAR_THETA_FIRST),
        cartesian_route,
    )
    route = jnp.where(
        radial_first_trace,
        jnp.int32(_CPU_ROUTE_POLAR_RADIAL_FIRST),
        standard_route,
    )

    def build_primary_support(_):
        return cartesian_support_builder(
            image_limb,
            physical_mask,
            axis=primary_axis,
            projection_trace=projection_trace,
            **common,
        )

    def empty_primary_support(_):
        return (
            jnp.zeros((cartesian_support_size, 2), dtype=w_center.real.dtype),
            jnp.zeros((cartesian_support_size,), dtype=bool),
            jnp.asarray(False),
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.asarray(False),
        )

    # A Cartesian support is part of the Cartesian integral itself.  Polar
    # routes build only angular support later, so keep a shape-compatible
    # placeholder here instead of constructing unused projection geometry.
    primary_support = jax.lax.cond(
        jax.lax.stop_gradient(route == _CPU_ROUTE_CARTESIAN),
        build_primary_support,
        empty_primary_support,
        operand=None,
    )
    primary_support_1d = _cartesian_support_1d(primary_support)

    return OneShotGeometryPrepared(
        image_limb,
        physical_mask,
        *neighbors,
        state,
        primary_axis,
        primary_support,
        primary_support_1d,
        # Every theta-first route uses the same partition contract.  In
        # particular, an unresolved 3->5 projection can select theta after
        # ``conditioned_trace`` is formed; keeping the older predicate here
        # silently disabled its crossing-edge guard.
        jax.lax.stop_gradient(theta_first_trace),
        jax.lax.stop_gradient(route),
        limb_trace,
    )


# Kept as a private compatibility alias for diagnostic scripts.  Geometry and
# routing are no longer specific to the uniform brightness profile.
_prepare_uniform_one_shot = _prepare_one_shot_geometry


def _solve_uniform_one_shot_prepared(
    prepared: OneShotGeometryPrepared,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    route: int,
    external_magnification: Array | None,
    cartesian_root_mode: str = "production",
    polar_root_mode: str = "companion",
) -> CartesianAdaptiveResult:
    """Evaluate one statically selected uniform route from a shared trace."""

    effective_root_mode = "companion" if cartesian_root_mode == "production" else cartesian_root_mode
    neighbors = (
        prepared.previous_limb,
        prepared.following_limb,
        prepared.previous_mask,
        prepared.following_mask,
    )
    edge_activity = (
        prepared.limb_trace.incoming_edge_active,
        prepared.limb_trace.outgoing_edge_active,
    )
    state = prepared.state

    if route == _CPU_ROUTE_POLAR_RADIAL_FIRST:
        radial = _mag_uniform_radial_from_trace(
            w_center,
            rho,
            s=s,
            q=q,
            rtol=jnp.asarray(1.0e-3, dtype=rho.dtype),
            image_limb=prepared.image_limb,
            physical_mask=prepared.physical_mask,
            independent_roots=False,
            use_simple_support=True,
            check_tolerance=False,
            neighbors=neighbors,
            edge_activity=edge_activity,
        )
        buried_unresolved = (
            (state.active_images_min == 3)
            & (state.active_images_max == 3)
            & (state.ghost_residual_ratio >= 100.0)
            & (state.ghost_residual_ratio < 1000.0)
            & (state.polar_conditioning >= 1000.0)
        )
        unresolved = buried_unresolved | ~_trace_transition_support_valid(state)
        status = jnp.bitwise_or(
            radial.status,
            jnp.where(
                unresolved,
                jnp.int32(ONE_SHOT_UNRESOLVED_GEOMETRY),
                jnp.int32(0),
            ),
        )
        status = jnp.bitwise_or(
            status,
            jnp.where(
                jnp.isfinite(radial.magnification),
                jnp.int32(0),
                jnp.int32(ONE_SHOT_NONFINITE),
            ),
        )
        return CartesianAdaptiveResult(
            radial.magnification,
            jnp.asarray(jnp.nan, dtype=radial.magnification.dtype),
            radial.n_full_root_solves,
            jnp.int32(ONE_SHOT_POLAR_RADIAL_FIRST),
            status,
        )

    if route == _CPU_ROUTE_POLAR_THETA_FIRST:
        support = _angular_support_cells_from_trace(
            w_center,
            rho,
            s=s,
            q=q,
            physical_limb=prepared.image_limb,
            physical_mask=prepared.physical_mask,
            topology_uncertain=state.topology_uncertain,
            minimum_ghost_residual=state.ghost_residual_ratio * rho,
            limb_topology=state.limb_topology,
            neighbors=neighbors,
            edge_activity=edge_activity,
            root_mode=polar_root_mode,
        )
        buried_pair = (
            state.topology_uncertain
            & (state.active_images_min == 3)
            & (state.active_images_max == 3)
            & jnp.isfinite(state.ghost_residual_ratio)
        )
        crossing = (
            state.limb_topology
            & (state.active_images_min == 3)
            & (state.active_images_max == 5)
        )
        critical_support, ghost_guard_applied = _split_angular_support_at_angle(
            support,
            state.ghost_image_angle,
            enabled=prepared.partition_theta & buried_pair,
        )
        # For a 3->5 transition the cut edge is intentionally not traversed,
        # but its newborn/dead pair still marks the occupied angular sector.
        # Retain two trace-derived guard cells on either side of that event.
        critical_support, crossing_guard_applied_1 = _split_angular_support_at_angle(
            critical_support,
            state.transition_image_angle,
            enabled=prepared.partition_theta & crossing,
        )
        critical_support, crossing_guard_applied_2 = _split_angular_support_at_angle(
            critical_support,
            state.transition_image_angle_2,
            enabled=prepared.partition_theta & crossing,
        )
        guard_applied = (
            ghost_guard_applied
            | crossing_guard_applied_1
            | crossing_guard_applied_2
        )
        partitioned_support = _split_overlapping_angular_support(
            critical_support,
            prepared.image_limb,
            prepared.physical_mask,
            parts=4,
        )
        selected_support = jax.tree.map(
            lambda partitioned, regular: jnp.where(
                prepared.partition_theta,
                partitioned,
                regular,
            ),
            partitioned_support,
            support,
        )
        selected_support_1d = selected_support.as_support_1d()
        selected_support = unwrap_support_1d(selected_support_1d)
        cells, active, topology, ghost, limb_topology, tangencies_valid = (
            selected_support
        )
        selected = _uniform_result_from_support(
            w_center,
            rho,
            s=s,
            q=q,
            n_theta=12,
            cells=cells,
            active=active,
            topology_uncertain=topology,
            minimum_ghost_residual=ghost,
            limb_topology=limb_topology,
            tangencies_valid=tangencies_valid,
            root_mode=polar_root_mode,
            certify_intervals=(
                state.topology_uncertain
                & (state.ghost_residual_ratio < 0.1)
                & (
                    ((state.active_images_min == 3) & (state.active_images_max == 3))
                    | ((state.active_images_min == 3) & (state.active_images_max == 5))
                )
            ),
        )
        base_cells = support.cells
        nonempty = base_cells[:, 1] > base_cells[:, 0]
        full_chart = jnp.all(support.active == nonempty)
        edge_consistent = _angular_trace_edge_consistent(
            prepared.image_limb,
            prepared.physical_mask,
            neighbors,
            edge_activity,
        )
        partition_valid = edge_consistent & (
            full_chart
            | (
                (
                    (jnp.sum(~support.tangencies_valid, dtype=jnp.int32) <= 2)
                    | guard_applied
                )
                & _closed_angular_extrema_parity_valid(
                    base_cells,
                    support.limb_topology,
                )
            )
        )
        regular_valid = edge_consistent & jnp.all(support.tangencies_valid) & (
            _closed_angular_extrema_parity_valid(
                base_cells,
                support.limb_topology,
            )
        )
        return _fixed_polar_result(
            selected,
            stage=ONE_SHOT_POLAR_THETA_FIRST,
            support_valid=jnp.where(
                prepared.partition_theta,
                partition_valid,
                regular_valid,
            )
            & _trace_transition_support_valid(state),
            replace_support_status=prepared.partition_theta,
        )

    primary_support = prepared.primary_support_1d
    primary_axis = prepared.primary_axis
    if route != _CPU_ROUTE_CARTESIAN:
        raise ValueError(f"unknown uniform one-shot route: {route}")
    del external_magnification
    primary = _cartesian_result_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        n_slice=8,
        support=primary_support,
        axis=primary_axis,
        continuation=effective_root_mode,
        image_limb=prepared.image_limb,
        physical_mask=prepared.physical_mask,
    )
    return _fixed_cartesian_result(
        primary,
        stage=_cartesian_stage(primary_axis),
    )


def mag_uniform_cpu_one_shot(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_limb: int = 64,
    external_magnification: Array | None = None,
    external_estimated_error: Array | None = None,
    return_state: bool = False,
    cartesian_root_mode: str = "production",
    polar_root_mode: str = "companion",
):
    """Evaluate the legacy uniform-only kernel for explicit comparisons.

    The default CPU backend uses ``mag_binary_cpu_one_shot`` with ``u1=0``.

    ``polar_root_mode`` accepts ``companion`` (the production default) and
    ``ea_fixed20/24/28`` or ``ea_auto20`` for explicit EA experiments. Polar
    support itself is derived only from the tracked image-limb coordinates;
    the selected root mode is used by the integration rays.
    """

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    q = jnp.asarray(q, dtype=w_center.real.dtype)
    del external_estimated_error
    prepared = _prepare_uniform_one_shot(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        external_available=external_magnification is not None,
    )
    branches = tuple(
        lambda _, selected_route=selected_route: _solve_uniform_one_shot_prepared(
            prepared,
            w_center,
            rho,
            s=s,
            q=q,
            route=selected_route,
            external_magnification=external_magnification,
            cartesian_root_mode=cartesian_root_mode,
            polar_root_mode=polar_root_mode,
        )
        for selected_route in range(_CPU_ROUTE_COUNT)
    )
    result = jax.lax.switch(prepared.route, branches, operand=None)
    result = _flag_unresolved_trace_transition(result, prepared.state)
    return (result, prepared.state) if return_state else result


def mag_binary_cpu_one_shot(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array = 0.0,
    n_limb: int = 112,
    external_magnification: Array | None = None,
    external_estimated_error: Array | None = None,
    return_state: bool = False,
    cartesian_root_mode: str = "production",
):
    """Evaluate a uniform or linear-LD source with one common moment solve.

    Support, chart selection and quadrature depend on geometry, not ``u1``.
    The final flux is ``((1-u1)*M0 + u1*M1)/(pi*rho**2*(1-u1/3))``;
    ``u1=0`` is the uniform image-area integral. No retry is performed.
    """

    if n_limb <= 0:
        raise ValueError("n_limb must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    q = jnp.asarray(q, dtype=w_center.real.dtype)
    u1 = jnp.asarray(u1, dtype=w_center.real.dtype)
    del external_estimated_error

    geometry = _prepare_one_shot_geometry(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        external_available=external_magnification is not None,
        cartesian_support_builder=_cartesian_support_cells_from_events,
        cartesian_support_size=21,
        allow_radial=False,
    )
    del external_magnification
    image_limb = geometry.image_limb
    physical_mask = geometry.physical_mask
    neighbors = (
        geometry.previous_limb,
        geometry.following_limb,
        geometry.previous_mask,
        geometry.following_mask,
    )
    edge_activity = (
        geometry.limb_trace.incoming_edge_active,
        geometry.limb_trace.outgoing_edge_active,
    )
    state = geometry.state
    topology = state.topology_uncertain
    ghost = state.ghost_residual_ratio * rho
    limb_topology = state.limb_topology
    route = geometry.route

    def theta_first(_):
        support = _angular_support_cells_from_events(
            w_center,
            rho,
            s=s,
            q=q,
            physical_limb=image_limb,
            physical_mask=physical_mask,
            topology_uncertain=topology,
            minimum_ghost_residual=ghost,
            limb_topology=limb_topology,
            neighbors=neighbors,
            edge_activity=edge_activity,
        )
        selected = _mag_limb_dark_angular_moment_from_support(
            w_center,
            rho,
            s=s,
            q=q,
            u1=u1,
            n_theta=_LD_ADAPTIVE_BUDGET,
            n_radial=8,
            support=support.as_support_1d(),
            estimate_radial_error=False,
            certify_intervals=False,
            moment_integrator=_integrate_ld_moments,
        )
        return _fixed_polar_result(
            selected,
            stage=ONE_SHOT_POLAR_THETA_FIRST,
            support_valid=jnp.all(support.tangencies_valid),
            replace_support_status=True,
        )

    def cartesian(_):
        primary_axis = geometry.primary_axis
        primary_support = geometry.primary_support_1d
        root_mode = (
            "companion"
            if cartesian_root_mode == "production"
            else cartesian_root_mode
        )

        def solve_cartesian(n_slice):
            return _mag_limb_dark_cartesian_impl(
                w_center,
                rho,
                s=s,
                q=q,
                u1=u1,
                n_slice=n_slice,
                n_profile=8,
                n_limb=n_limb,
                axis=primary_axis,
                support=primary_support,
                root_mode=root_mode,
                bernstein_capacity=8,
                return_info=True,
                moment_integrator=_integrate_ld_moments,
            )

        primary = solve_cartesian(_LD_ADAPTIVE_BUDGET)
        return _fixed_cartesian_result(
            primary,
            stage=_cartesian_stage(primary_axis),
        )

    result = jax.lax.cond(
        route == _CPU_ROUTE_CARTESIAN,
        cartesian,
        theta_first,
        operand=None,
    )
    result = _flag_unresolved_trace_transition(result, state)
    return (result, state) if return_state else result


# Compatibility name for callers of the former LD-only entry point.
mag_limb_dark_cpu_one_shot = mag_binary_cpu_one_shot


__all__ = [
    "ONE_SHOT_CARTESIAN_X_FIRST",
    "ONE_SHOT_CARTESIAN_Y_FIRST",
    "ONE_SHOT_POLAR_RADIAL_FIRST",
    "ONE_SHOT_POLAR_THETA_FIRST",
    "ONE_SHOT_INVALID_ROOTS",
    "ONE_SHOT_NONFINITE",
    "ONE_SHOT_UNRESOLVED_GEOMETRY",
    "OneShotState",
    "mag_binary_cpu_one_shot",
    "mag_limb_dark_cpu_one_shot",
    "mag_uniform_cpu_one_shot",
]
