"""State-conditioned one-shot CPU ICRS schedulers.

The routines in this module deliberately separate *routing* from *rescue*.
One source-limb trace is used to choose a coordinate chart and one fixed
high-order quadrature before the area integral is evaluated.  The production
path deliberately makes no per-point accuracy estimate.  Non-zero status
reports only a detected structural failure; it never triggers another chart,
order, or trace.

This makes the steady-state work auditable and keeps the forward-mode JAX
graph substantially smaller than the historical adaptive retry ladder.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .angular_limb_dark import (
    _mag_limb_dark_angular_moment_from_support,
    _mag_limb_dark_angular_moment_gk15_pair_from_support,
)
from .angular_moment import (
    ANGULAR_MOMENT_SUPPORT,
    ANGULAR_MOMENT_TOPOLOGY,
    _angular_support_cells_from_trace,
    _split_angular_support_at_angle,
    _split_overlapping_angular_support,
    _uniform_nested_cc_from_support,
    _uniform_result_from_support,
)
from .cartesian_limb_dark import (
    _mag_limb_dark_cartesian_gk15_from_support,
    _mag_limb_dark_cartesian_impl,
)
from .cartesian_moment import (
    CartesianAdaptiveResult,
    _cartesian_gk15_from_support,
    _cartesian_result_from_support,
    _cartesian_support_cells_from_trace,
    _cartesian_trace_diagnostics,
)
from .radial_continuation import _mag_uniform_radial_from_trace
from .sentinel import nearest_caustic_limb_reference
from .support import trace_binary_source_limb, tracked_limb_neighbors

Array = jnp.ndarray

# Public stage labels.  ``lightcurve.py`` adds four so these remain compatible
# with the existing tier convention while making the chosen route observable.
ONE_SHOT_CARTESIAN = 1
ONE_SHOT_CARTESIAN_HIGH = 2
ONE_SHOT_RADIAL = 3
ONE_SHOT_POLAR = 4
ONE_SHOT_POLAR_HIGH = 5

# Structural status bits specific to the one-shot scheduler. Existing
# support/topology bits are inherited from the fixed-order kernels.
ONE_SHOT_INVALID_ROOTS = 1 << 20
ONE_SHOT_NONFINITE = 1 << 21
ONE_SHOT_UNRESOLVED_GEOMETRY = 1 << 22

_UNIFORM_CARTESIAN_EXTERNAL = 0
_UNIFORM_CARTESIAN_INTERNAL = 1
_UNIFORM_CARTESIAN_HIGH = 2
_UNIFORM_POLAR = 3
_UNIFORM_POLAR_HIGH = 4
_UNIFORM_CONDITIONED_POLAR = 5
_UNIFORM_ROUTE_COUNT = 6

# Below this scale the planetary image loops are strongly anisotropic.  A
# source-normal projection makes the line-level sextic nearly tangent to the
# narrow pair, whereas the source-radial projection separates its real roots.
# The same conditioning limit also precedes the rho**2 cancellation of the
# angle-first radial polynomial.  This is a chart choice, not an accuracy
# retry: it changes neither the limb trace nor the fixed quadrature budget.
_SMALL_SOURCE_RADIAL_CHART = 2.0e-4


class OneShotState(NamedTuple):
    """Axis-independent state measured from the single source-limb trace."""

    topology_uncertain: Array
    limb_topology: Array
    ghost_residual_ratio: Array
    polar_conditioning: Array
    active_images_min: Array
    active_images_max: Array


class OneShotGeometryPrepared(NamedTuple):
    """Brightness-independent trace, support, and route selection."""

    image_limb: Array
    physical_mask: Array
    previous_limb: Array
    following_limb: Array
    previous_mask: Array
    following_mask: Array
    state: OneShotState
    primary_axis: Array
    primary_support: tuple[Array, Array, Array, Array, Array]
    route: Array


def _unresolved_buried_caustic_contact(
    w_center: Array,
    rho: Array,
    s: Array,
    q: Array,
    state: OneShotState,
) -> Array:
    """Flag a buried caustic too close to the sampled source boundary.

    A 3 -> 3 image count around the complete limb cannot seed the two image
    loops created wholly inside the disk.  Rejecting every such state is too
    conservative: a caustic deeply inside the source can be covered by the
    radial rings even when it is absent from the physical limb roots.  The
    dimensionless reference-point clearance therefore rejects only a caustic
    close enough to the source boundary that the fixed limb trace can miss its
    narrow contact structure.
    """

    _, clearance = nearest_caustic_limb_reference(
        w_center,
        rho,
        s=s,
        q=q,
    )
    return (
        state.topology_uncertain
        & (state.active_images_min == 3)
        & (state.active_images_max == 3)
        & (clearance <= 0.1)
    )


def _buried_caustic_contact_angle(
    w_center: Array,
    rho: Array,
    s: Array,
    q: Array,
    state: OneShotState,
) -> tuple[Array, Array]:
    """Return a buried-contact flag and its critical-image polar angle."""

    critical, clearance = nearest_caustic_limb_reference(
        w_center,
        rho,
        s=s,
        q=q,
    )
    contact = (
        state.topology_uncertain
        & (state.active_images_min == 3)
        & (state.active_images_max == 3)
        & (clearance <= 0.1)
    )
    return contact, jnp.mod(jnp.angle(critical), 2.0 * jnp.pi)


def _polar_support_unreliable(
    w_center: Array,
    rho: Array,
    s: Array,
    q: Array,
    state: OneShotState,
    *,
    limb_dark: bool,
    buried_contact_resolved: Array | bool = False,
) -> Array:
    """Reject traced polar supports known not to contain every image loop."""

    tiny = jnp.finfo(rho.dtype).tiny
    source_ratio = jnp.abs(w_center) / jnp.maximum(rho, tiny)
    caustic_to_source = jnp.sqrt(q) / jnp.maximum(rho, tiny)
    resonant_zone = jnp.abs(jnp.log(s)) <= 1.5 * jnp.cbrt(q)
    low_q_planetary = (q <= 1.0e-5) & state.topology_uncertain & (source_ratio >= 10.0)
    unresolved_small_source_planetary = (
        (q <= 3.0e-4)
        & state.topology_uncertain
        & state.limb_topology
        & (source_ratio >= 10.0)
        & (caustic_to_source >= 20.0)
    )
    underresolved_resonant = (
        (q <= 1.0e-3)
        & resonant_zone
        & state.topology_uncertain
        & (caustic_to_source >= 3.0)
    )
    return (
        (
            _unresolved_buried_caustic_contact(w_center, rho, s, q, state)
            & ~jnp.asarray(buried_contact_resolved)
        )
        | low_q_planetary
        | unresolved_small_source_planetary
        | (underresolved_resonant & ~jnp.asarray(limb_dark))
    )


def _high_polar_noncontact_override(
    w_center: Array,
    rho: Array,
    tangencies_valid: Array,
) -> Array:
    """Allow a co-converged high-polar pair away from radial contact.

    The high route is also selected for conservative low-q topology warnings.
    Away from the annular contact band, its traced angular cells are complete
    when at most two fitted tangencies are unresolved.  Keeping the override
    below ten source radii excludes the distant planetary-support omissions;
    keeping it outside 0.7--1.3 excludes the origin-contact failures.  This is
    a pre-integration state condition, not a retry or an accuracy escalation.
    """

    source_ratio = jnp.abs(w_center) / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    noncontact = (source_ratio < 0.7) | ((source_ratio > 1.3) & (source_ratio < 10.0))
    return noncontact & (jnp.sum(~tangencies_valid, dtype=jnp.int32) <= 2)


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

    # ``AngularSupport.active`` is now the endpoint-preserving union mask, not
    # the number of cells in the extrema partition.  Parity belongs to the
    # partition itself and therefore counts every nonempty atomic cell.
    cell_count = jnp.sum(cells[:, 1] > cells[:, 0], dtype=jnp.int32)
    return jnp.asarray(limb_topology) | ((cell_count & jnp.int32(1)) == 1)


def _unresolved_distant_topology(
    w_center: Array,
    rho: Array,
    s: Array,
    q: Array,
    state: OneShotState,
    use_polar: Array,
    *,
    limb_dark: bool,
) -> Array:
    """Detect a distant fold not represented reliably by the selected chart."""

    source_ratio = jnp.abs(w_center) / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    exposed_fold = (
        state.limb_topology
        & (state.active_images_min == 3)
        & (state.active_images_max == 5)
    )
    buried_fold = (state.active_images_min == 3) & (state.active_images_max == 3)
    caustic_to_source = jnp.sqrt(q) / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    small_planet = (q <= 1.0e-3) & (caustic_to_source >= 3.0) & ~jnp.asarray(limb_dark)
    tiny_planet = (q <= 1.0e-5) & ~jnp.asarray(limb_dark)
    unresolved_large_resonant = (
        (q <= 1.0e-2) & (caustic_to_source >= 20.0) & ~jnp.asarray(limb_dark)
    )
    nonresonant_planet = (
        (q <= 1.0e-2) & (jnp.abs(jnp.log(s)) > jnp.cbrt(q)) & (caustic_to_source >= 1.0)
    )
    return (
        ~use_polar
        & state.topology_uncertain
        & (exposed_fold | buried_fold)
        & (source_ratio >= 10.0)
        & (tiny_planet | small_planet | unresolved_large_resonant | nonresonant_planet)
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

    image_limb, physical_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        include_all_roots=False,
    )
    neighbors = tracked_limb_neighbors(image_limb, physical_mask)
    topology, ghost, limb_topology = _cartesian_trace_diagnostics(
        w_center,
        rho,
        s=s,
        q=q,
        image_limb=image_limb,
        physical_mask=physical_mask,
    )

    _, following_limb, _, following_mask = neighbors
    connected = physical_mask & following_mask
    radius = jnp.abs(image_limb)
    following_radius = jnp.abs(following_limb)
    angular_step = jnp.abs(jnp.angle(following_limb * jnp.conjugate(image_limb)))
    tangential_motion = jnp.sum(
        jnp.where(
            connected,
            0.5 * (radius + following_radius) * angular_step,
            0.0,
        )
    )
    radial_motion = jnp.sum(
        jnp.where(connected, jnp.abs(following_radius - radius), 0.0)
    )
    motion_floor = 128.0 * jnp.finfo(w_center.real.dtype).eps
    polar_conditioning = tangential_motion / jnp.maximum(radial_motion, motion_floor)
    image_counts = jnp.sum(physical_mask, axis=0, dtype=jnp.int32)
    state = OneShotState(
        topology,
        limb_topology,
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        polar_conditioning,
        jnp.min(image_counts),
        jnp.max(image_counts),
    )
    return image_limb, physical_mask, neighbors, state


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


def _fixed_polar_result(
    result,
    *,
    stage: int,
    support_valid: Array | bool,
) -> CartesianAdaptiveResult:
    """Expose one fixed polar value with structural diagnostics only."""

    # Fixed polar kernels use bit 0 for a non-finite reduction. Normalize that
    # private bit into the public ONE_SHOT_NONFINITE flag below.
    soft = jnp.bitwise_or(jnp.int32(1), jnp.int32(ANGULAR_MOMENT_TOPOLOGY))
    if stage == ONE_SHOT_POLAR_HIGH:
        soft = jnp.bitwise_or(soft, jnp.int32(ANGULAR_MOMENT_SUPPORT))
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
) -> OneShotGeometryPrepared:
    """Trace and route one source without evaluating a brightness moment."""

    image_limb, physical_mask, neighbors, state = _trace_state(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
    )
    # The source-normal axis is the stable default for a single Cartesian
    # support construction.  Very small sources use the orthogonal radial
    # axis because it separates the anisotropic planetary image pair.  Both
    # choices are independent of lens parameters q and s; those parameters
    # must not rotate a chart.  A raw PCA axis can align with the broad image
    # envelope while merging a narrow lobe into one projection cell, so the
    # multi-axis selection below remains reserved for fragmented folds.
    safe_magnitude = jnp.maximum(
        jnp.abs(w_center),
        jnp.finfo(w_center.real.dtype).tiny,
    )
    source_normal_axis = jnp.where(
        jnp.abs(w_center) > 0.0,
        1.0j * w_center / safe_magnitude,
        jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype),
    )
    source_radial_axis = jnp.where(
        jnp.abs(w_center) > 0.0,
        w_center / safe_magnitude,
        jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype),
    )
    small_source = rho <= _SMALL_SOURCE_RADIAL_CHART
    primary_axis = jnp.where(
        small_source,
        source_radial_axis,
        source_normal_axis,
    )
    routing_axis = primary_axis * jnp.exp(
        1.0j * jnp.deg2rad(jnp.asarray(22.5, dtype=w_center.real.dtype))
    )
    source_ratio = jnp.abs(w_center) / jnp.maximum(
        rho,
        jnp.finfo(rho.dtype).tiny,
    )
    # Only these image states can use the routing support's fragmentation
    # score.  Smooth Cartesian points and ordinary polar points need no
    # second endpoint union before their actual support is built.
    annular = state.polar_conditioning >= 25.0
    routing_candidate = annular | (
        state.limb_topology
        & (state.active_images_min == 3)
        & (state.active_images_max == 5)
        & (state.polar_conditioning >= 10.0)
    )
    common = dict(
        topology_uncertain=jnp.asarray(False),
        minimum_ghost_residual=state.ghost_residual_ratio * rho,
        limb_topology=state.limb_topology,
        maximum_extrema=20,
        neighbors=neighbors,
    )
    def build_routing_support(_):
        return _cartesian_support_cells_from_trace(
            image_limb,
            physical_mask,
            axis=routing_axis,
            **common,
        )

    def empty_routing_support(_):
        return (
            jnp.zeros((19, 2), dtype=w_center.real.dtype),
            jnp.zeros((19,), dtype=bool),
            jnp.asarray(False),
            jnp.asarray(jnp.inf, dtype=w_center.real.dtype),
            jnp.asarray(False),
        )

    routing_support = jax.lax.cond(
        jax.lax.stop_gradient(routing_candidate),
        build_routing_support,
        empty_routing_support,
        operand=None,
    )
    routing_cells, routing_active, *_ = routing_support
    routing_widths = jnp.where(
        routing_active,
        routing_cells[:, 1] - routing_cells[:, 0],
        0.0,
    )
    routing_total_width = jnp.sum(routing_widths)
    routing_max_fraction = jnp.max(routing_widths) / jnp.maximum(
        routing_total_width,
        jnp.finfo(w_center.real.dtype).tiny,
    )
    # These are order flags, not chart selectors.  Keep them next to the
    # chart predicates below so the source-ratio and annular tests have one
    # canonical implementation.
    contact = annular & (source_ratio >= 0.70) & (source_ratio <= 1.30)
    high_polar = contact | (state.topology_uncertain & (q <= 1.0e-3))
    # The two former cases (fold and no-fold) partition on limb_topology, so
    # their union is simply the ghost-distance guard itself.
    high_cartesian = state.ghost_residual_ratio <= 2.0
    fragmented_annulus = annular & (routing_max_fraction <= 0.55)
    exposed_distant_fold = (
        state.limb_topology
        & (state.active_images_min == 3)
        & (state.active_images_max == 5)
        & (state.ghost_residual_ratio <= 1.1)
        & (state.polar_conditioning <= 2.0)
        & (source_ratio >= 10.0)
    )
    # Chart choice is based on the traced image state.  The caustic sentinel
    # that produced ``topology_uncertain`` already used the lens parameters to
    # account for hidden structure; repeating a q cut here only made the chart
    # selector parameter-dependent a second time.  The source-ratio guard
    # keeps this state route local to the source rather than to distant image
    # topology.
    central_planetary_topology = state.topology_uncertain & (source_ratio <= 10.0)
    # When the traced limb advances much farther tangentially than radially,
    # Cartesian strips can share a missing image lobe even with independent
    # line roots.  This is a chart-conditioning test, not a lens-parameter or
    # coordinate special case: the same source-limb trace selects the radial
    # chart before either area rule is evaluated.
    long_topology_arc = state.limb_topology & (state.polar_conditioning >= 15.0)
    use_polar = fragmented_annulus | exposed_distant_fold | central_planetary_topology
    needs_internal = (
        ~jnp.asarray(external_available)
        | state.topology_uncertain
        | (state.ghost_residual_ratio <= 5.0)
        | (
            (state.active_images_min == 5)
            & (state.polar_conditioning >= 10.0)
            & (source_ratio <= 6.0)
        )
    )
    cartesian_route = jnp.where(
        needs_internal,
        jnp.int32(_UNIFORM_CARTESIAN_INTERNAL),
        jnp.int32(_UNIFORM_CARTESIAN_EXTERNAL),
    )
    cartesian_route = jnp.where(
        high_cartesian,
        jnp.int32(_UNIFORM_CARTESIAN_HIGH),
        cartesian_route,
    )
    polar_route = jnp.where(
        high_polar,
        jnp.int32(_UNIFORM_POLAR_HIGH),
        jnp.int32(_UNIFORM_POLAR),
    )
    standard_route = jnp.where(use_polar, polar_route, cartesian_route)
    # This is not an axis choice: it selects the radial-first numerical
    # conditioning used inside the polar chart when the level-set coefficient
    # scale is known to be delicate.  Keep it separate from the image-state
    # chart selector above.
    resonant_radial = (q <= 1.0e-3) & (jnp.abs(jnp.log(s)) <= 1.5 * jnp.cbrt(q))
    route = jnp.where(
        state.limb_topology & ~use_polar,
        jnp.int32(_UNIFORM_CARTESIAN_HIGH),
        standard_route,
    )
    route = jnp.where(
        resonant_radial | central_planetary_topology | long_topology_arc,
        jnp.int32(_UNIFORM_CONDITIONED_POLAR),
        route,
    )
    # At very small rho the angle-first level-set coefficients lose the image
    # pair through rho**2 cancellation.  Use the already selected radial
    # Cartesian projection instead of spending the high-polar budget on an
    # incomplete angular support.  Conditioned-polar states above remain
    # untouched because their fixed-r equation is independently stable.
    route = jnp.where(
        small_source & (route == _UNIFORM_POLAR_HIGH),
        jnp.int32(_UNIFORM_CARTESIAN_HIGH),
        route,
    )

    route_is_polar = (
        (route == _UNIFORM_POLAR)
        | (route == _UNIFORM_POLAR_HIGH)
        | (route == _UNIFORM_CONDITIONED_POLAR)
    )

    def build_primary_support(_):
        return _cartesian_support_cells_from_trace(
            image_limb,
            physical_mask,
            axis=primary_axis,
            **common,
        )

    # The routing support above is enough for every polar route.  Construct a
    # source-axis support only for a Cartesian integral; this keeps the
    # brightness-independent scheduler from paying for an unused support.
    primary_support = jax.lax.cond(
        ~jax.lax.stop_gradient(route_is_polar),
        build_primary_support,
        lambda _: routing_support,
        operand=None,
    )
    return OneShotGeometryPrepared(
        image_limb,
        physical_mask,
        *neighbors,
        state,
        primary_axis,
        primary_support,
        jax.lax.stop_gradient(route),
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

    effective_root_mode = (
        "companion" if cartesian_root_mode == "production" else cartesian_root_mode
    )
    neighbors = (
        prepared.previous_limb,
        prepared.following_limb,
        prepared.previous_mask,
        prepared.following_mask,
    )
    state = prepared.state

    if route == _UNIFORM_CONDITIONED_POLAR:

        def radial_first(_):
            radial = _mag_uniform_radial_from_trace(
                w_center,
                rho,
                s=s,
                q=q,
                # The radial result is deliberately structural-only in the
                # one-shot route; this argument is ignored because
                # ``check_tolerance=False`` below.
                rtol=jnp.asarray(1.0e-3, dtype=rho.dtype),
                image_limb=prepared.image_limb,
                physical_mask=prepared.physical_mask,
                independent_roots=False,
                use_simple_support=True,
                check_tolerance=False,
            )
            radial_status = jnp.bitwise_or(
                radial.status,
                jnp.where(
                    _unresolved_buried_caustic_contact(w_center, rho, s, q, state),
                    jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
                    jnp.int32(0),
                ),
            )
            return CartesianAdaptiveResult(
                radial.magnification,
                jnp.asarray(jnp.nan, dtype=radial.magnification.dtype),
                radial.n_full_root_solves,
                jnp.int32(ONE_SHOT_RADIAL),
                jnp.bitwise_or(
                    radial_status,
                    jnp.where(
                        jnp.isfinite(radial.magnification),
                        jnp.int32(0),
                        jnp.int32(ONE_SHOT_NONFINITE),
                    ),
                ),
            )

        def angle_first(_):
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
                root_mode=polar_root_mode,
            )
            buried_contact, critical_angle = _buried_caustic_contact_angle(
                w_center, rho, s, q, state
            )
            critical_support, critical_split = _split_angular_support_at_angle(
                support,
                critical_angle,
                enabled=buried_contact,
            )
            integration_support = _split_overlapping_angular_support(
                critical_support,
                prepared.image_limb,
                prepared.physical_mask,
                parts=3,
            )
            cells, active, topology, ghost, limb_topology, tangencies_valid = (
                integration_support
            )
            selected = _uniform_result_from_support(
                w_center,
                rho,
                s=s,
                q=q,
                n_theta=32,
                cells=cells,
                active=active,
                topology_uncertain=topology,
                minimum_ghost_residual=ghost,
                limb_topology=limb_topology,
                tangencies_valid=tangencies_valid,
                root_mode=polar_root_mode,
            )
            nonempty = support.cells[:, 1] > support.cells[:, 0]
            full_chart = jnp.all(support.active == nonempty)
            buried_contact_resolved = full_chart & buried_contact & critical_split
            unreliable = _polar_support_unreliable(
                w_center,
                rho,
                s,
                q,
                state,
                limb_dark=False,
                buried_contact_resolved=buried_contact_resolved,
            )
            partition_valid = full_chart | (
                (jnp.sum(~tangencies_valid, dtype=jnp.int32) <= 2)
                & _closed_angular_extrema_parity_valid(
                    support.cells, support.limb_topology
                )
            )
            support_valid = (
                partition_valid
                & (
                    (rho > 1.0e-3)
                    | (
                        jnp.abs(w_center)
                        < 0.7 * jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
                    )
                )
                & ~unreliable
            )
            return _fixed_polar_result(
                selected,
                stage=ONE_SHOT_POLAR_HIGH,
                support_valid=support_valid,
            )._replace(
                # Stage 3 denotes the conditioned-polar family.
                stage=jnp.int32(ONE_SHOT_RADIAL),
            )

        # The angle-first level-set coefficients contain a subtraction at
        # scale rho**2.  At rho <= 1e-3 this becomes the dominant x64 limit
        # near planetary caustics.  The fixed-r equation is independently
        # reliable in the resonant zone; nonresonant small-rho points remain
        # fail-closed in ``angle_first`` instead of triggering a rescue.
        resonant = jnp.abs(jnp.log(s)) <= 1.5 * jnp.cbrt(q)
        return jax.lax.cond(
            (rho <= 1.0e-3) & resonant,
            radial_first,
            angle_first,
            operand=None,
        )

    if route in (_UNIFORM_POLAR, _UNIFORM_POLAR_HIGH):
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
            root_mode=polar_root_mode,
        )
        cells, active, topology, ghost, limb_topology, tangencies_valid = support
        if route == _UNIFORM_POLAR_HIGH:
            fine, _ = _uniform_nested_cc_from_support(
                w_center,
                rho,
                s=s,
                q=q,
                fine_intervals=24,
                cells=cells,
                active=active,
                topology_uncertain=topology,
                minimum_ghost_residual=ghost,
                limb_topology=limb_topology,
                root_mode=polar_root_mode,
            )
            unreliable = _polar_support_unreliable(
                w_center, rho, s, q, state, limb_dark=False
            )
            support_valid = (jnp.sum(~tangencies_valid, dtype=jnp.int32) <= 2) & (
                ~unreliable
                | _high_polar_noncontact_override(w_center, rho, tangencies_valid)
            ) & _closed_angular_extrema_parity_valid(cells, limb_topology)
            return _fixed_polar_result(
                fine,
                stage=ONE_SHOT_POLAR_HIGH,
                support_valid=support_valid,
            )
        fine, _ = _uniform_nested_cc_from_support(
            w_center,
            rho,
            s=s,
            q=q,
            fine_intervals=12,
            cells=cells,
            active=active,
            topology_uncertain=topology,
            minimum_ghost_residual=ghost,
            limb_topology=limb_topology,
            root_mode=polar_root_mode,
        )
        return _fixed_polar_result(
            fine,
            stage=ONE_SHOT_POLAR,
            support_valid=(
                jnp.all(tangencies_valid)
                | _high_polar_noncontact_override(w_center, rho, tangencies_valid)
            )
            & _closed_angular_extrema_parity_valid(cells, limb_topology)
            & ~_polar_support_unreliable(w_center, rho, s, q, state, limb_dark=False),
        )

    primary_support = prepared.primary_support
    primary_axis = prepared.primary_axis
    if route not in (
        _UNIFORM_CARTESIAN_EXTERNAL,
        _UNIFORM_CARTESIAN_INTERNAL,
        _UNIFORM_CARTESIAN_HIGH,
    ):
        raise ValueError(f"unknown uniform one-shot route: {route}")
    if route == _UNIFORM_CARTESIAN_EXTERNAL and external_magnification is not None:
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
        return _fixed_cartesian_result(primary, stage=ONE_SHOT_CARTESIAN)
    # GK15/G7 shares every expensive root evaluation. Keep only the GK15 value;
    # the embedded G7 reduction is intentionally not an acceptance test.
    fine, _ = _cartesian_gk15_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        support=primary_support,
        axis=primary_axis,
        continuation=effective_root_mode,
    )
    return _fixed_cartesian_result(
        fine,
        stage=(
            ONE_SHOT_CARTESIAN_HIGH
            if route == _UNIFORM_CARTESIAN_HIGH
            else ONE_SHOT_CARTESIAN
        ),
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
    """Evaluate one uniform source with no retry or post-failure fallback.

    ``polar_root_mode`` accepts ``companion`` (the production default),
    ``ea_fixed20/24/28`` for explicit fixed EA schedules, and ``ea_auto20``
    to use EA only for the large support-certification batches while retaining
    companion roots for the small quadrature-cell batches.
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
        for selected_route in range(_UNIFORM_ROUTE_COUNT)
    )
    result = jax.lax.switch(prepared.route, branches, operand=None)
    selected_polar = (prepared.route == _UNIFORM_POLAR) | (
        prepared.route == _UNIFORM_POLAR_HIGH
    )
    selected_complete_chart = selected_polar | (
        prepared.route == _UNIFORM_CONDITIONED_POLAR
    )
    unresolved = _unresolved_distant_topology(
        w_center,
        rho,
        s,
        q,
        prepared.state,
        selected_complete_chart,
        limb_dark=False,
    )
    result = result._replace(
        status=jnp.where(
            unresolved,
            jnp.bitwise_or(
                result.status, jnp.int32(ONE_SHOT_UNRESOLVED_GEOMETRY)
            ),
            result.status,
        )
    )
    return (result, prepared.state) if return_state else result


def mag_limb_dark_cpu_one_shot(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    n_limb: int = 64,
    external_magnification: Array | None = None,
    external_estimated_error: Array | None = None,
    return_state: bool = False,
    cartesian_root_mode: str = "production",
):
    """Evaluate one linear-LD source with no retry or post-failure fallback."""

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
    )
    image_limb = geometry.image_limb
    physical_mask = geometry.physical_mask
    neighbors = (
        geometry.previous_limb,
        geometry.following_limb,
        geometry.previous_mask,
        geometry.following_mask,
    )
    state = geometry.state
    topology = state.topology_uncertain
    ghost = state.ghost_residual_ratio * rho
    limb_topology = state.limb_topology
    route = geometry.route
    use_polar = (
        (route == _UNIFORM_POLAR)
        | (route == _UNIFORM_POLAR_HIGH)
        | (route == _UNIFORM_CONDITIONED_POLAR)
    )
    high_polar = route == _UNIFORM_POLAR_HIGH
    high_cartesian = route == _UNIFORM_CARTESIAN_HIGH

    def polar(_):
        support = _angular_support_cells_from_trace(
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
        )
        def evaluate(n_theta, n_radial):
            return _mag_limb_dark_angular_moment_from_support(
                w_center,
                rho,
                s=s,
                q=q,
                u1=u1,
                n_theta=n_theta,
                n_radial=n_radial,
                support=support,
                estimate_radial_error=False,
            )

        def high(_):
            return _fixed_polar_result(
                evaluate(24, 12),
                stage=ONE_SHOT_POLAR_HIGH,
                support_valid=(jnp.sum(~support.tangencies_valid, dtype=jnp.int32) <= 2)
                & _closed_angular_extrema_parity_valid(
                    support.cells, support.limb_topology
                )
                & ~_polar_support_unreliable(
                    w_center, rho, s, q, state, limb_dark=True
                ),
            )

        def regular(_):
            return _fixed_polar_result(
                evaluate(12, 8),
                stage=ONE_SHOT_POLAR,
                support_valid=jnp.all(support.tangencies_valid)
                & _closed_angular_extrema_parity_valid(
                    support.cells, support.limb_topology
                )
                & ~_polar_support_unreliable(
                    w_center, rho, s, q, state, limb_dark=True
                ),
            )

        def radial(_):
            buried_contact, critical_angle = _buried_caustic_contact_angle(
                w_center, rho, s, q, state
            )
            critical_support, critical_split = _split_angular_support_at_angle(
                support,
                critical_angle,
                enabled=buried_contact,
            )
            conditioned_support = _split_overlapping_angular_support(
                critical_support,
                image_limb,
                physical_mask,
                parts=3,
            )
            # Most conditioned supports are substantially over-resolved by
            # the historical 15x15 Kronrod rule.  A fixed 12x10 Gauss rule
            # removes 47% of the mapped profile nodes.  Spend the high-order
            # budget only when support construction reports an unverified
            # tangency; this is an axis-independent structural condition,
            # not a lens-parameter special case or a result-based retry.
            high_conditioned_order = jnp.any(
                ~conditioned_support.tangencies_valid
            )

            def high_order(_):
                return _mag_limb_dark_angular_moment_gk15_pair_from_support(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    u1=u1,
                    support=conditioned_support,
                )[1]

            def standard_order(_):
                return _mag_limb_dark_angular_moment_from_support(
                    w_center,
                    rho,
                    s=s,
                    q=q,
                    u1=u1,
                    n_theta=12,
                    n_radial=10,
                    support=conditioned_support,
                    estimate_radial_error=False,
                )

            fine = jax.lax.cond(
                jax.lax.stop_gradient(high_conditioned_order),
                high_order,
                standard_order,
                operand=None,
            )
            nonempty = support.cells[:, 1] > support.cells[:, 0]
            full_chart = jnp.all(support.active == nonempty)
            buried_contact_resolved = full_chart & buried_contact & critical_split
            unreliable = _polar_support_unreliable(
                w_center,
                rho,
                s,
                q,
                state,
                limb_dark=True,
                buried_contact_resolved=buried_contact_resolved,
            )
            partition_valid = full_chart | (
                (jnp.sum(~support.tangencies_valid, dtype=jnp.int32) <= 2)
                & _closed_angular_extrema_parity_valid(
                    support.cells, support.limb_topology
                )
            )
            return _fixed_polar_result(
                fine,
                stage=ONE_SHOT_POLAR_HIGH,
                support_valid=partition_valid & ~unreliable,
            )

        selected = jax.lax.cond(
            route == _UNIFORM_CONDITIONED_POLAR,
            radial,
            lambda _: jax.lax.cond(high_polar, high, regular, operand=None),
            operand=None,
        )
        return selected._replace(
            stage=jnp.where(
                route == _UNIFORM_CONDITIONED_POLAR,
                jnp.int32(ONE_SHOT_RADIAL),
                selected.stage,
            )
        )

    def cartesian(_):
        primary_axis = geometry.primary_axis
        primary_support = geometry.primary_support
        root_mode = (
            "companion" if cartesian_root_mode == "production" else cartesian_root_mode
        )

        def fixed_gk15(_):
            fine, _ = _mag_limb_dark_cartesian_gk15_from_support(
                w_center,
                rho,
                s=s,
                q=q,
                u1=u1,
                axis=primary_axis,
                support=primary_support,
                root_mode=root_mode,
            )
            return _fixed_cartesian_result(
                fine,
                stage=jnp.where(
                    high_cartesian,
                    jnp.int32(ONE_SHOT_CARTESIAN_HIGH),
                    jnp.int32(ONE_SHOT_CARTESIAN),
                ),
            )

        if external_magnification is None:
            return fixed_gk15(None)

        def fixed_external_route(_):
            primary = _mag_limb_dark_cartesian_impl(
                w_center,
                rho,
                s=s,
                q=q,
                u1=u1,
                n_slice=7,
                n_profile=8,
                n_limb=n_limb,
                axis=primary_axis,
                support=primary_support,
                root_mode=root_mode,
                bernstein_capacity=8,
                return_info=True,
            )
            return _fixed_cartesian_result(primary, stage=ONE_SHOT_CARTESIAN)

        return jax.lax.cond(
            jax.lax.stop_gradient(route == _UNIFORM_CARTESIAN_EXTERNAL),
            fixed_external_route,
            fixed_gk15,
            operand=None,
        )

    result = jax.lax.cond(use_polar, polar, cartesian, operand=None)
    unresolved = _unresolved_distant_topology(
        w_center, rho, s, q, state, use_polar, limb_dark=True
    )
    result = result._replace(
        status=jnp.where(
            unresolved,
            jnp.bitwise_or(
                result.status, jnp.int32(ONE_SHOT_UNRESOLVED_GEOMETRY)
            ),
            result.status,
        )
    )
    return (result, state) if return_state else result


__all__ = [
    "ONE_SHOT_CARTESIAN",
    "ONE_SHOT_CARTESIAN_HIGH",
    "ONE_SHOT_RADIAL",
    "ONE_SHOT_POLAR",
    "ONE_SHOT_POLAR_HIGH",
    "ONE_SHOT_INVALID_ROOTS",
    "ONE_SHOT_NONFINITE",
    "ONE_SHOT_UNRESOLVED_GEOMETRY",
    "OneShotState",
    "mag_limb_dark_cpu_one_shot",
    "mag_uniform_cpu_one_shot",
]
