"""One trace, one selected chart, one bounded two-moment integral on CPU."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .angular_limb_dark import _mag_limb_dark_angular_moment_from_support
from .angular_projection_support import _angular_support_cells_from_events
from .cartesian_limb_dark import _mag_limb_dark_cartesian_impl
from .projection_support import _cartesian_support_cells_from_events
from .support import trace_source_limb_record
from .lens import binary_lens

Array = jnp.ndarray
ONE_SHOT_UNRESOLVED_GEOMETRY = 1 << 22


class OneShotState(NamedTuple):
    """Only trace quantities used for route selection or structural status."""

    topology_uncertain: Array
    limb_topology: Array
    ghost_residual_ratio: Array
    polar_conditioning: Array
    maximum_angular_step: Array
    active_images_min: Array
    active_images_max: Array


def _trace_state(
    w_center: Array,
    rho: Array,
    *,
    lens,
    n_limb: int,
):
    """Trace once and extract the state used by both brightness profiles."""

    trace = trace_source_limb_record(
        w_center,
        rho,
        lens=lens,
        n_limb=n_limb,
    )
    image_limb = trace.image_limb
    physical_mask = trace.physical_mask
    image_counts = trace.image_counts
    limb_topology = jnp.any(image_counts != image_counts[0])
    residual = trace.residual
    ghost_residuals = jnp.where(~physical_mask, residual, jnp.inf)
    ghost = jnp.min(ghost_residuals)
    topology = limb_topology | (ghost <= 4.0 * rho) | ~jnp.all(jnp.isfinite(jnp.where(physical_mask, image_limb, 0.0j)))

    following_limb = trace.following_limb
    outgoing_edge_active = trace.outgoing_edge_active
    angular_step = trace.angular_step
    tangential_motion = jnp.sum(
        jnp.where(
            outgoing_edge_active,
            0.5 * (jnp.abs(image_limb) + jnp.abs(following_limb)) * angular_step,
            0.0,
        )
    )
    radial_motion = jnp.sum(jnp.where(outgoing_edge_active, trace.radial_step, 0.0))
    motion_floor = 128.0 * jnp.finfo(w_center.real.dtype).eps
    polar_conditioning = tangential_motion / jnp.maximum(radial_motion, motion_floor)
    state = OneShotState(
        topology,
        limb_topology,
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        polar_conditioning,
        jnp.max(jnp.where(outgoing_edge_active, angular_step, 0.0)),
        jnp.min(image_counts),
        jnp.max(image_counts),
    )
    return trace, state


def _select_chart(limb_trace, state, w_center, *, n_limb):
    """Compare x/y projections and polar motion from the same image trace."""
    image_limb = limb_trace.image_limb
    minimum_images = 3 if image_limb.shape[0] == 5 else 4
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
        projected_turn = (projected - projected_previous) * (projected_following - projected) <= 0.0
        projected_endpoint = physical_mask & (~previous_mask | ~following_mask | explicit_same_count_cut)
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
        maximum_step = (
            jnp.max(
                jnp.where(
                    outgoing_edge_active,
                    jnp.abs(projected_following - projected),
                    0.0,
                )
            )
            / scale
        )
        return extrema, maximum_step

    x_axis = jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype)
    y_axis = jnp.asarray(0.0 + 1.0j, dtype=w_center.dtype)
    x_extrema, x_step = projection_diagnostics(x_axis)
    y_extrema, y_step = projection_diagnostics(y_axis)
    # Closed image loops have paired extrema. Prefer the projection with
    # valid parity, then the count nearest two extrema per image.
    closed_trace = (
        ~state.limb_topology & (state.active_images_min == minimum_images) & (state.active_images_max == minimum_images)
    )
    expected_extrema = jnp.int32(2) * state.active_images_min
    x_parity_bad = (x_extrema & jnp.int32(1)) == 1
    y_parity_bad = (y_extrema & jnp.int32(1)) == 1
    use_y_first = closed_trace & (
        (x_parity_bad & ~y_parity_bad)
        | (
            (x_parity_bad == y_parity_bad)
            & (jnp.abs(y_extrema - expected_extrema) < jnp.abs(x_extrema - expected_extrema))
        )
    )
    primary_axis = jnp.where(use_y_first, y_axis, x_axis)
    projected_extrema = jnp.where(use_y_first, y_extrema, x_extrema)
    # A closed projected image loop has paired extrema.  An odd number of
    # sampled turns together with a large traced angular jump identifies a
    # branch whose narrow turning sector is poorly represented by Cartesian
    # nodes. Route that trace to the polar integral.
    source_phase_step = 2.0 * jnp.pi / jnp.asarray(n_limb, dtype=w_center.real.dtype)
    underresolved_projected_turn = ((projected_extrema & jnp.int32(1)) == 1) & (
        state.maximum_angular_step > 0.3 * source_phase_step
    )
    # Compare unresolved image motion per trace sample in both coordinates.
    polar_motion_score = state.polar_conditioning * (
        state.maximum_angular_step / jnp.maximum(source_phase_step, 1.0e-30)
    )
    # For closed traces compare against the worst projected gap: unlike a
    # transition, the polar chart has no cut-edge guard and must beat both
    # Cartesian projections before it is selected.
    cartesian_motion_score = jnp.maximum(x_step, y_step)
    # At a fold, switch only when the angular gap is resolved and polar
    # motion is not effectively radial-free. No lens-parameter regions route
    # the calculation.
    cartesian_selected_score = jnp.where(use_y_first, y_step, x_step)
    transition_chart_better = (
        state.limb_topology
        & (state.active_images_min == minimum_images)
        & (state.active_images_max == minimum_images + 2)
        & (state.maximum_angular_step > 0.004 * source_phase_step)
        & (state.polar_conditioning > 1.0)
        & (polar_motion_score > cartesian_selected_score)
    )
    closed_conditioned = (
        ~state.topology_uncertain & ~state.limb_topology & (polar_motion_score > cartesian_motion_score)
    )
    conditioned_trace = (
        transition_chart_better
        | closed_conditioned
        | (state.topology_uncertain & ~state.limb_topology)
        | (~state.limb_topology & underresolved_projected_turn)
    ) & (state.maximum_angular_step > 0.004 * source_phase_step)
    resolved_closed_three_cartesian = (
        ~state.topology_uncertain
        & ~state.limb_topology
        & (state.active_images_min == minimum_images)
        & (state.active_images_max == minimum_images)
        & (cartesian_motion_score <= 4.0e-4)
    )
    theta_first = (transition_chart_better | conditioned_trace) & ~resolved_closed_three_cartesian
    return primary_axis, jax.lax.stop_gradient(theta_first)


def mag_cpu_one_shot(w_center, rho, *, lens, u1=0.0):
    """Integrate uniform and linear-LD moments on a shared geometry.

    The 112-point limb trace and bounded quadrature are fixed. Both moments
    control allocation, including at u1=0. A structural warning preserves
    the best-effort value and never triggers a second chart or a retry.
    """
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    u1 = jnp.asarray(u1, dtype=w_center.real.dtype)
    trace, state = _trace_state(w_center, rho, lens=lens, n_limb=112)
    axis, theta_first = _select_chart(trace, state, w_center, n_limb=112)
    neighbors = (
        trace.previous_limb,
        trace.following_limb,
        trace.previous_mask,
        trace.following_mask,
    )
    common = dict(
        topology_uncertain=state.topology_uncertain,
        minimum_ghost_residual=state.ghost_residual_ratio * rho,
        limb_topology=state.limb_topology,
        neighbors=neighbors,
    )

    def cartesian(_):
        support = _cartesian_support_cells_from_events(
            trace.image_limb,
            trace.physical_mask,
            w_center=w_center,
            rho=rho,
            lens=lens,
            axis=axis,
            **common,
        )
        return _mag_limb_dark_cartesian_impl(
            w_center,
            rho,
            lens=lens,
            u1=u1,
            axis=axis,
            support=support,
        )

    def angular(_):
        support = _angular_support_cells_from_events(
            w_center,
            rho,
            lens=lens,
            physical_limb=trace.image_limb,
            physical_mask=trace.physical_mask,
            edge_activity=(trace.incoming_edge_active, trace.outgoing_edge_active),
            **common,
        )
        return _mag_limb_dark_angular_moment_from_support(
            w_center,
            rho,
            lens=lens,
            u1=u1,
            support=support,
        )

    result = jax.lax.cond(theta_first, angular, cartesian, operand=None)
    trace_valid = (
        (~state.topology_uncertain | jnp.isfinite(state.ghost_residual_ratio))
        & jnp.isfinite(state.polar_conditioning)
        & jnp.isfinite(state.maximum_angular_step)
    )
    minimum_images = lens.positions.shape[0] + 1
    buried_unresolved = (
        (state.active_images_min == minimum_images)
        & (state.active_images_max == minimum_images)
        & (state.ghost_residual_ratio >= 100.0)
        & (state.ghost_residual_ratio < 1000.0)
        & (state.polar_conditioning >= 1000.0)
    )
    return result._replace(
        status=result.status
        | jnp.where(
            trace_valid & ~buried_unresolved,
            jnp.int32(0),
            jnp.int32(ONE_SHOT_UNRESOLVED_GEOMETRY),
        )
    )


def mag_binary_cpu_one_shot(w_center, rho, *, s, q, u1=0.0):
    return mag_cpu_one_shot(w_center, rho, lens=binary_lens(s, q), u1=u1)
