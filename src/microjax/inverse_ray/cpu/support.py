"""Fixed-shape radial support discovery for the binary CPU kernel."""

from __future__ import annotations

from itertools import permutations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from microjax.point_source import lens_eq
from microjax.poly_solver import poly_roots

from .coefficients import binary_quintic_coefficients
from ..geometry.lens import binary_geometry

Array = jnp.ndarray

_BINARY_ROOT_PERMUTATIONS = np.asarray(
    tuple(permutations(range(5))),
    dtype=np.int32,
)

class RadialSupport(NamedTuple):
    """Disjoint radial cells covering all sampled physical image limbs."""

    intervals: Array
    active: Array
    image_limb: Array
    image_mask: Array

    def as_support_1d(self) -> "Support1D":
        """Expose radial cells through the shared support contract."""

        return make_support_1d(
            self.intervals,
            self.active,
            payload=self,
        )


class LimbTrace(NamedTuple):
    """Canonical source-limb trace shared by CPU geometry consumers.

    The quintic roots and their physical mask must be treated as one object:
    rebuilding either one downstream can make support endpoints refer to a
    different set of images than the root certificate.  The incoming and
    outgoing edge masks are deliberately stricter than ``connected`` and only
    permit edges whose two phases have the same number of physical images.  A
    3-to-5 image transition is therefore represented by two disconnected trace
    pieces.  Both directions are stored explicitly because the periodic closing
    edge can permute algebraic root slots.

    This is a data contract, not a new numerical route.  The existing tuple
    return from :func:`trace_binary_source_limb` remains available for legacy
    callers; production one-shot geometry uses
    :func:`trace_binary_source_limb_record` below.
    """

    image_limb: Array
    physical_mask: Array
    phases: Array
    previous_limb: Array
    following_limb: Array
    previous_mask: Array
    following_mask: Array
    image_counts: Array
    residual: Array
    finite_mask: Array
    connected: Array
    incoming_edge_active: Array
    outgoing_edge_active: Array
    angular_step: Array
    radial_step: Array

    @property
    def edge_active(self) -> Array:
        """Backward-compatible alias for the outgoing edge mask."""

        return self.outgoing_edge_active


class Support1D(NamedTuple):
    """Common one-dimensional support contract for the three CPU charts.

    ``active`` identifies cells that participate in the single integral.
    ``core`` marks cells whose endpoints are currently certified; an active
    cell with ``core=False`` is a conservative guard cell.  ``endpoint_valid``
    and ``overflow`` are diagnostics only at this stage.  ``payload`` keeps
    the chart-specific diagnostics lossless while the numerical kernels are
    migrated incrementally; callers must not rebuild support from it.
    """

    cells: Array
    active: Array
    core: Array
    endpoint_valid: Array
    overflow: Array
    payload: object


def unwrap_support_1d(support):
    """Return the legacy chart payload during the compatibility migration."""

    return support.payload if isinstance(support, Support1D) else support


def make_support_1d(
    cells: Array,
    active: Array,
    *,
    endpoint_valid: Array | bool = True,
    overflow: Array | bool = False,
    payload=None,
) -> Support1D:
    """Build a fixed-shape support contract without changing cell values."""

    cells = jnp.asarray(cells)
    active = jnp.asarray(active, dtype=bool)
    endpoint_valid = jnp.asarray(endpoint_valid, dtype=bool)
    if endpoint_valid.ndim == 0:
        endpoint_valid = jnp.full_like(active, endpoint_valid, dtype=bool)
    overflow = jnp.asarray(overflow, dtype=bool)
    core = active & endpoint_valid
    if payload is None:
        payload = (cells, active)
    return Support1D(cells, active, core, endpoint_valid, overflow, payload)


class PolarOccupancyAtlas(NamedTuple):
    """Per-branch radial ranges and minimal unwrapped angular arcs."""

    radial_bounds: Array
    angular_bounds: Array
    active: Array


class PolarProductRegions(NamedTuple):
    """Non-overlapping radial-angular cells occupied by traced limb points."""

    radial_bounds: Array
    angular_bounds: Array
    active: Array
    overflow: Array


def _physical_image_mask(images_com: Array, w_com: Array, s: Array, q: Array) -> Array:
    """Keep the three best roots and any additional well-resolved images."""

    a = 0.5 * s
    secondary_mass = q / (1.0 + q)
    midpoint_to_com = a * (1.0 - q) / (1.0 + q)
    residual = jnp.abs(
        lens_eq(
            images_com - midpoint_to_com,
            nlenses=2,
            a=a,
            e1=secondary_mass,
        )
        - (w_com - midpoint_to_com)
    )
    order = jnp.argsort(residual)
    rank = jnp.zeros_like(order).at[order].set(jnp.arange(order.size))
    residual_scale = jnp.maximum(1.0, jnp.abs(w_com))
    resolved = residual <= 1.0e-7 * residual_scale
    return (rank < 3) | resolved


def _polish_binary_images(
    images_com: Array,
    w_com: Array,
    s: Array,
    q: Array,
) -> Array:
    """Project near-physical quintic roots onto the exact lens equation.

    The q-aware quintic is still an ill-conditioned representation close to a
    very low-mass lens.  A genuine planetary image can therefore have a larger
    lens-equation residual than the three easy images even though its position
    is already close enough for Newton's method.  Two fixed real-Newton steps
    remove that polynomial conditioning error before limb topology is built.

    This is part of every limb trace, not a retry or data-dependent rescue.  A
    root is moved only when it starts close to the exact lens equation, the
    Jacobian is nonsingular, and the proposed update decreases the residual.
    """

    real_dtype = images_com.real.dtype
    a = 0.5 * s
    secondary_mass = q / (1.0 + q)
    midpoint_to_com = a * (1.0 - q) / (1.0 + q)
    images_midpoint = images_com - midpoint_to_com
    source_midpoint = w_com - midpoint_to_com
    residual_scale = jnp.maximum(1.0, jnp.abs(w_com))
    determinant_floor = jnp.sqrt(jnp.finfo(real_dtype).eps)

    initial_residual = (
        lens_eq(images_midpoint, nlenses=2, a=a, e1=secondary_mass)
        - source_midpoint
    )
    eligible = (
        jnp.isfinite(initial_residual.real)
        & jnp.isfinite(initial_residual.imag)
        & (jnp.abs(initial_residual) < 1.0e-4 * residual_scale)
    )

    polished = images_midpoint
    for _ in range(2):
        residual = (
            lens_eq(polished, nlenses=2, a=a, e1=secondary_mass)
            - source_midpoint
        )
        shear = (
            secondary_mass / (jnp.conjugate(polished) - a) ** 2
            + (1.0 - secondary_mass) / (jnp.conjugate(polished) + a) ** 2
        )
        shear_abs = jnp.abs(shear)
        determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
        safe = eligible & (jnp.abs(determinant) > determinant_floor)
        step = (-residual + shear * jnp.conjugate(residual)) / jnp.where(
            safe,
            determinant,
            1.0,
        )
        candidate = polished + jnp.where(safe, step, 0.0 + 0.0j)
        candidate_residual = (
            lens_eq(candidate, nlenses=2, a=a, e1=secondary_mass)
            - source_midpoint
        )
        improves = (
            safe
            & jnp.isfinite(candidate.real)
            & jnp.isfinite(candidate.imag)
            & (jnp.abs(candidate_residual) < jnp.abs(residual))
        )
        polished = jnp.where(improves, candidate, polished)

    return polished + midpoint_to_com


def trace_binary_source_limb(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_limb: int,
    include_all_roots: bool | Array = False,
    phases: Array | None = None,
) -> tuple[Array, Array]:
    """Solve and polish a source limb from q-aware quintic coefficients.

    The returned image coordinates are the guarded Newton-polished roots used
    to construct the returned physical mask.  Keeping one coordinate set for
    both purposes is essential: all later support extrema and branch
    continuation must see the same roots that passed the lens-equation
    residual test.

    ``phases`` is optional so the production uniform trace remains unchanged.
    Supplying it enables fixed-shape, non-uniform scout/refinement experiments;
    the values must be sorted on ``[0, 2 pi)`` and have length ``n_limb``.
    """

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    if phases is None:
        phases = 2.0 * jnp.pi * jnp.arange(n_limb, dtype=real_dtype) / n_limb
    else:
        phases = jnp.asarray(phases, dtype=real_dtype)
        if phases.ndim != 1 or phases.shape[0] != n_limb:
            raise ValueError("phases must be one-dimensional with length n_limb")
    source_limb = w_center + jnp.asarray(rho, dtype=real_dtype) * jnp.exp(1.0j * phases)
    quintics = binary_quintic_coefficients(source_limb, s=s, q=q)

    first_roots = poly_roots(quintics.coefficients[:1])[0]

    def continue_roots(previous, coefficients):
        roots = poly_roots(
            coefficients[None, :],
            custom_init=True,
            roots_init=previous[None, :],
        )[0]
        return roots, roots

    _, remaining_roots = jax.lax.scan(
        continue_roots,
        first_roots,
        quintics.coefficients[1:],
    )
    roots_planet = jnp.concatenate((first_roots[None, :], remaining_roots), axis=0)
    images_com = roots_planet + quintics.image_shift
    polished_images = jax.vmap(
        lambda images, source: _polish_binary_images(images, source, s, q)
    )(images_com, source_limb)
    image_mask = jax.vmap(
        lambda images, source: _physical_image_mask(images, source, s, q)
    )(polished_images, source_limb)
    image_mask = jnp.where(
        jnp.asarray(include_all_roots),
        jnp.ones_like(image_mask),
        image_mask,
    )
    # The mask is evaluated on the polished roots.  Return those same roots
    # to every downstream consumer (support extrema, branch neighbours, and
    # strip/radial seeds); returning ``images_com`` here would make the trace
    # geometry and its physical-root certificate refer to different points.
    return jnp.moveaxis(polished_images, 0, 1), jnp.moveaxis(image_mask, 0, 1)


def trace_binary_source_limb_record(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_limb: int,
    include_all_roots: bool | Array = False,
    phases: Array | None = None,
) -> LimbTrace:
    """Return one canonical trace and its connectivity certificate.

    This helper intentionally wraps the established root trace instead of
    changing its solver or physical-mask policy.  It only derives data that
    used to be recomputed by individual route selectors: residuals, image
    multiplicities, periodic neighbours, and same-multiplicity active edges.
    Consequently adopting the record is behavior-preserving while making it
    impossible for a later support builder to silently use a different trace.
    """

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    if phases is None:
        phases = 2.0 * jnp.pi * jnp.arange(n_limb, dtype=real_dtype) / n_limb
    else:
        phases = jnp.asarray(phases, dtype=real_dtype)
    image_limb, physical_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        include_all_roots=include_all_roots,
        phases=phases,
    )
    previous_limb, following_limb, previous_mask, following_mask = (
        tracked_limb_neighbors(image_limb, physical_mask)
    )

    lens = binary_geometry(s, q)
    source_limb = w_center + jnp.asarray(rho, dtype=real_dtype) * jnp.exp(
        1.0j * phases
    )
    residual = jax.vmap(
        lambda images, source: jnp.abs(
            lens_eq(
                images - lens.shifted,
                nlenses=2,
                a=lens.a,
                e1=lens.e1,
            )
            - (source - lens.shifted)
        ),
        in_axes=(1, 0),
    )(image_limb, source_limb).T
    finite_mask = jnp.isfinite(image_limb.real) & jnp.isfinite(image_limb.imag)
    image_counts = jnp.sum(physical_mask, axis=0, dtype=jnp.int32)
    connected = physical_mask & following_mask
    incoming_connected = physical_mask & previous_mask
    same_previous_multiplicity = image_counts == jnp.roll(image_counts, 1)
    same_following_multiplicity = image_counts == jnp.roll(image_counts, -1)
    incoming_edge_active = (
        incoming_connected & same_previous_multiplicity[None, :]
    )
    outgoing_edge_active = connected & same_following_multiplicity[None, :]
    radius = jnp.abs(image_limb)
    following_radius = jnp.abs(following_limb)
    angular_step = jnp.abs(
        jnp.angle(following_limb * jnp.conjugate(image_limb))
    )
    radial_step = jnp.abs(following_radius - radius)
    return LimbTrace(
        image_limb,
        physical_mask,
        phases,
        previous_limb,
        following_limb,
        previous_mask,
        following_mask,
        image_counts,
        residual,
        finite_mask,
        connected,
        incoming_edge_active,
        outgoing_edge_active,
        angular_step,
        radial_step,
    )


def tracked_limb_neighbors(
    image_limb: Array,
    image_mask: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return periodic neighbors while preserving tracked root branches.

    Sequential quintic solves keep each root in its previous slot locally,
    but analytic continuation around a source loop can permute those slots.
    The last-to-first edge therefore needs an explicit five-root assignment;
    a plain ``roll`` can connect unrelated images exactly where a caustic is
    enclosed. The 120 possible binary-root permutations are small enough to
    score exhaustively once per trace.
    """

    if image_limb.shape[0] != 5:
        raise ValueError("binary limb tracking requires five algebraic roots")
    root_permutations = jnp.asarray(_BINARY_ROOT_PERMUTATIONS)
    first = image_limb[:, 0]
    last = image_limb[:, -1]
    reordered_last = last[root_permutations]
    assignment_cost = jnp.sum(
        jnp.abs(reordered_last - first[None, :]) ** 2,
        axis=1,
    )
    selected = jnp.argmin(jax.lax.stop_gradient(assignment_cost))
    closing_permutation = root_permutations[selected]
    inverse_permutation = jnp.zeros_like(closing_permutation).at[
        closing_permutation
    ].set(jnp.arange(5, dtype=jnp.int32))

    previous = jnp.roll(image_limb, 1, axis=1).at[:, 0].set(
        last[closing_permutation]
    )
    following = jnp.roll(image_limb, -1, axis=1).at[:, -1].set(
        first[inverse_permutation]
    )
    previous_mask = jnp.roll(image_mask, 1, axis=1).at[:, 0].set(
        image_mask[:, -1][closing_permutation]
    )
    following_mask = jnp.roll(image_mask, -1, axis=1).at[:, -1].set(
        image_mask[:, 0][inverse_permutation]
    )
    return previous, following, previous_mask, following_mask


def limb_edge_activity_from_masks(
    image_mask: Array,
    previous_mask: Array,
    following_mask: Array,
) -> tuple[Array, Array]:
    """Return the legacy per-root edge graph when no canonical trace exists.

    Canonical one-shot callers must pass ``LimbTrace.incoming_edge_active`` and
    ``LimbTrace.outgoing_edge_active`` instead.  Keeping this fallback in one
    place lets older fixed/adaptive entry points preserve their historical
    connectivity without allowing individual support builders to infer it in
    subtly different ways.
    """

    return image_mask & previous_mask, image_mask & following_mask


def radial_support_from_limb(
    image_limb: Array,
    image_mask: Array,
    *,
    neighbors: tuple[Array, Array, Array, Array] | None = None,
    edge_activity: tuple[Array, Array] | None = None,
    split_turning_points: bool = False,
    maximum_extrema: int = 20,
) -> RadialSupport:
    """Project five tracked image limbs into fixed-shape radial cells.

    The production radial-first route splits at traced radial turning points;
    legacy fixed tiers retain the cheaper per-branch envelope partition.
    """

    radii = jnp.abs(image_limb)
    branch_active = jnp.any(image_mask, axis=1)
    lower = jnp.min(jnp.where(image_mask, radii, jnp.inf), axis=1)
    upper = jnp.max(jnp.where(image_mask, radii, -jnp.inf), axis=1)

    # Keep sampled extrema as cell endpoints.  Expanding them by a generic
    # margin would move the square-root onset into the cell interior and
    # defeat the sine-squared endpoint transform.  The fixed tier comparison
    # below certifies the remaining source-limb discretization error.
    lower = jnp.maximum(0.0, lower)
    lower = jnp.where(branch_active, lower, 0.0)
    upper = jnp.where(branch_active, upper, 0.0)

    if split_turning_points:
        if neighbors is None:
            neighbors = tracked_limb_neighbors(image_limb, image_mask)
        previous, following, previous_mask, following_mask = neighbors
        if edge_activity is None:
            edge_activity = limb_edge_activity_from_masks(
                image_mask,
                previous_mask,
                following_mask,
            )
        incoming_edge_active, outgoing_edge_active = edge_activity
        previous_radius = jnp.abs(previous)
        following_radius = jnp.abs(following)
        incoming = radii - previous_radius
        outgoing = following_radius - radii
        # A canonical edge cut is a segment boundary, but a 3-to-5
        # multiplicity transition must not turn every persistent root in that
        # edge into a radial endpoint.  The latter would add five artificial
        # panels at every caustic crossing and was measurably less accurate.
        # Keep explicit same-multiplicity cuts (used by legacy callers and
        # diagnostics) as point endpoints; for a multiplicity transition the
        # existing physical-mask endpoint candidates are sufficient.
        image_counts = jnp.sum(image_mask, axis=0, dtype=jnp.int32)
        incoming_same_count = image_counts == jnp.roll(image_counts, 1)
        outgoing_same_count = image_counts == jnp.roll(image_counts, -1)
        same_count_at_point = incoming_same_count & outgoing_same_count
        explicit_same_count_cut = (
            (~incoming_edge_active | ~outgoing_edge_active)
            & same_count_at_point[None, :]
            & previous_mask
            & following_mask
        )
        candidate = image_mask & (
            (
                incoming_edge_active
                & outgoing_edge_active
                & (incoming * outgoing <= 0.0)
            )
            | ~previous_mask
            | ~following_mask
            | explicit_same_count_cut
        )
        curvature = previous_radius - 2.0 * radii + following_radius
        safe_curvature = jnp.where(
            jnp.abs(curvature) > 64.0 * jnp.finfo(radii.dtype).eps,
            curvature,
            1.0,
        )
        offset = jnp.clip(
            0.5 * (previous_radius - following_radius) / safe_curvature,
            -1.0,
            1.0,
        )
        fitted = (
            radii
            + 0.5 * (following_radius - previous_radius) * offset
            + 0.5 * curvature * offset**2
        )
        fitted = jnp.where(
            incoming_edge_active & outgoing_edge_active,
            fitted,
            radii,
        )
        indices = jnp.nonzero(
            candidate.reshape(-1),
            size=maximum_extrema,
            fill_value=-1,
        )[0]
        safe_indices = jnp.maximum(indices, 0)
        endpoints = fitted.reshape(-1)[safe_indices]
        endpoints = jnp.where(indices >= 0, endpoints, jnp.inf)
        endpoints = jnp.sort(endpoints)
    else:
        endpoints = jnp.sort(jnp.concatenate((lower, upper)))
    cells = jnp.stack((endpoints[:-1], endpoints[1:]), axis=-1)
    midpoint = 0.5 * (cells[:, 0] + cells[:, 1])
    covered = jnp.any(
        branch_active[:, None]
        & (midpoint[None, :] >= lower[:, None])
        & (midpoint[None, :] <= upper[:, None]),
        axis=0,
    )
    active = covered & (cells[:, 1] > cells[:, 0])
    return RadialSupport(cells, active, image_limb, image_mask)


def polar_occupancy_atlas_from_limb(
    image_limb: Array,
    image_mask: Array,
    *,
    neighbors: tuple[Array, Array, Array, Array] | None = None,
) -> PolarOccupancyAtlas:
    """Build one conservative polar box for each tracked image branch.

    The angular box is the complement of the largest empty circular gap in
    the branch's limb samples. It is represented as one unwrapped interval,
    so a branch crossing zero does not become an almost-full-circle box.
    Radial extrema use the same local quadratic fit as fixed-shape support.
    """

    if neighbors is None:
        neighbors = tracked_limb_neighbors(image_limb, image_mask)
    previous, following, previous_mask, following_mask = neighbors
    finite = jnp.isfinite(image_limb.real) & jnp.isfinite(image_limb.imag)
    valid = image_mask & finite
    active = jnp.any(valid, axis=1)

    radius = jnp.abs(image_limb)
    previous_radius = jnp.abs(previous)
    following_radius = jnp.abs(following)
    curvature = previous_radius - 2.0 * radius + following_radius
    safe_curvature = jnp.where(
        jnp.abs(curvature) > 64.0 * jnp.finfo(radius.dtype).eps,
        curvature,
        1.0,
    )
    offset = jnp.clip(
        0.5 * (previous_radius - following_radius) / safe_curvature,
        -1.0,
        1.0,
    )
    fitted_radius = (
        radius
        + 0.5 * (following_radius - previous_radius) * offset
        + 0.5 * curvature * offset**2
    )
    fitted_radius = jnp.where(
        previous_mask & following_mask,
        fitted_radius,
        radius,
    )
    radial_lower = jnp.min(jnp.where(valid, fitted_radius, jnp.inf), axis=1)
    radial_upper = jnp.max(jnp.where(valid, fitted_radius, -jnp.inf), axis=1)
    radial_lower = jnp.where(active, jnp.maximum(radial_lower, 0.0), 0.0)
    radial_upper = jnp.where(active, radial_upper, 0.0)

    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=radius.dtype)
    angles = jnp.mod(jnp.angle(image_limb), two_pi)
    sorted_angles = jnp.sort(jnp.where(valid, angles, jnp.inf), axis=1)
    counts = jnp.sum(valid, axis=1, dtype=jnp.int32)
    safe_last = jnp.maximum(counts - 1, 0)
    first = sorted_angles[:, 0]
    last = jnp.take_along_axis(
        sorted_angles,
        safe_last[:, None],
        axis=1,
    )[:, 0]
    adjacent_gap = sorted_angles[:, 1:] - sorted_angles[:, :-1]
    gap_index = jnp.arange(adjacent_gap.shape[1], dtype=jnp.int32)
    adjacent_gap = jnp.where(
        gap_index[None, :] < (counts - 1)[:, None],
        adjacent_gap,
        -jnp.inf,
    )
    wrap_gap = first + two_pi - last
    all_gaps = jnp.concatenate((adjacent_gap, wrap_gap[:, None]), axis=1)
    largest_gap_index = jnp.argmax(all_gaps, axis=1)
    internal_gap = largest_gap_index < adjacent_gap.shape[1]
    safe_gap_index = jnp.minimum(largest_gap_index, adjacent_gap.shape[1] - 1)
    gap_lower = jnp.take_along_axis(
        sorted_angles,
        safe_gap_index[:, None],
        axis=1,
    )[:, 0]
    gap_upper = jnp.take_along_axis(
        sorted_angles,
        (safe_gap_index + 1)[:, None],
        axis=1,
    )[:, 0]
    angular_lower = jnp.where(internal_gap, gap_upper, first)
    angular_upper = jnp.where(internal_gap, gap_lower + two_pi, last)

    following_angle_step = jnp.abs(
        jnp.angle(following * jnp.conjugate(image_limb))
    )
    image_count = jnp.sum(image_mask, axis=0, dtype=jnp.int32)
    same_multiplicity = image_count == jnp.roll(image_count, -1)
    connected = valid & following_mask & same_multiplicity[None, :]
    margin = jnp.max(
        jnp.where(connected, following_angle_step, 0.0),
        axis=1,
    )
    angular_lower = angular_lower - margin
    angular_upper = angular_upper + margin
    span = angular_upper - angular_lower
    angular_lower = jnp.where(span >= two_pi, 0.0, angular_lower)
    angular_upper = jnp.where(span >= two_pi, two_pi, angular_upper)
    angular_lower = jnp.where(active, angular_lower, 0.0)
    angular_upper = jnp.where(active, angular_upper, 0.0)
    return PolarOccupancyAtlas(
        jnp.stack((radial_lower, radial_upper), axis=1),
        jnp.stack((angular_lower, angular_upper), axis=1),
        active,
    )


def _merge_support_intervals(
    intervals: Array,
    active: Array,
) -> tuple[Array, Array]:
    """Merge a fixed-capacity set of overlapping one-dimensional intervals."""

    order = jnp.argsort(jnp.where(active, intervals[:, 0], jnp.inf))
    intervals = intervals[order]
    active = active[order]
    output = jnp.zeros_like(intervals)

    def step(carry, item):
        merged, count, lower, upper, has_current = carry
        bounds, is_active = item
        overlaps = is_active & has_current & (bounds[0] <= upper)
        starts = is_active & ~has_current
        separates = is_active & has_current & ~overlaps
        merged = jax.lax.cond(
            separates,
            lambda values: values.at[count].set(jnp.stack((lower, upper))),
            lambda values: values,
            merged,
        )
        count = count + separates.astype(jnp.int32)
        lower = jnp.where(starts | separates, bounds[0], lower)
        upper = jnp.where(
            overlaps,
            jnp.maximum(upper, bounds[1]),
            jnp.where(starts | separates, bounds[1], upper),
        )
        return (merged, count, lower, upper, has_current | is_active), None

    initial = (
        output,
        jnp.int32(0),
        jnp.asarray(0.0, dtype=intervals.dtype),
        jnp.asarray(0.0, dtype=intervals.dtype),
        jnp.asarray(False),
    )
    (output, count, lower, upper, has_current), _ = jax.lax.scan(
        step,
        initial,
        (intervals, active),
    )
    output = jax.lax.cond(
        has_current,
        lambda values: values.at[count].set(jnp.stack((lower, upper))),
        lambda values: values,
        output,
    )
    return output, count + has_current.astype(jnp.int32)


def _partition_support_intervals(
    intervals: Array,
    active: Array,
) -> tuple[Array, Array]:
    """Partition a conservative 1-D union at every support endpoint."""

    endpoints = jnp.where(active[:, None], intervals, jnp.inf).reshape(-1)
    endpoints = jnp.sort(endpoints)
    lower = endpoints[:-1]
    upper = endpoints[1:]
    midpoint = 0.5 * (lower + upper)
    finite_width = jnp.isfinite(upper) & (upper > lower)
    covered = jnp.any(
        active[:, None]
        & (midpoint[None, :] >= intervals[:, 0, None])
        & (midpoint[None, :] <= intervals[:, 1, None]),
        axis=0,
    )
    partition_active = finite_width & covered
    partition = jnp.stack((lower, upper), axis=1)
    partition = jnp.where(partition_active[:, None], partition, 0.0)
    return partition, partition_active


def polar_product_regions_from_limb(
    image_limb: Array,
    image_mask: Array,
    *,
    capacity: int = 20,
    include_enclosed: bool = False,
    include_segments: bool = False,
    partition_supports: bool = False,
    prune_products: bool = True,
) -> PolarProductRegions:
    """Pair one-dimensional polar supports without overlapping area cells.

    Radial branch envelopes and minimal circular angular arcs are independently
    merged into disjoint one-dimensional supports. Their Cartesian products
    are retained when they contain a physical limb sample. If
    ``include_enclosed`` is true, cells whose centre has non-zero winding with
    respect to a closed tracked image limb are retained as well. With
    ``include_segments``, consecutive ordered limb samples retain every cell
    intersecting their swept radial-angular range. With
    ``partition_supports``, conservative one-dimensional supports are first
    split at every endpoint and only then pruned in two dimensions. This is the
    tracked-root counterpart of the Paper 1 polar region builder.
    """

    atlas = polar_occupancy_atlas_from_limb(image_limb, image_mask)
    if partition_supports:
        radial_regions, radial_active = _partition_support_intervals(
            atlas.radial_bounds,
            atlas.active,
        )
    else:
        radial_regions, radial_count = _merge_support_intervals(
            atlas.radial_bounds,
            atlas.active,
        )
        radial_active = jnp.arange(radial_regions.shape[0]) < radial_count
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=image_limb.real.dtype)
    span = atlas.angular_bounds[:, 1] - atlas.angular_bounds[:, 0]
    lower = jnp.mod(atlas.angular_bounds[:, 0], two_pi)
    upper = lower + span
    first = jnp.stack((lower, jnp.minimum(upper, two_pi)), axis=1)
    second = jnp.stack((jnp.zeros_like(lower), upper - two_pi), axis=1)
    angular_intervals = jnp.concatenate((first, second), axis=0)
    angular_active = jnp.concatenate(
        (
            atlas.active & (first[:, 1] > first[:, 0]),
            atlas.active & (second[:, 1] > second[:, 0]),
        )
    )
    if partition_supports:
        angular_regions, angular_active = _partition_support_intervals(
            angular_intervals,
            angular_active,
        )
    else:
        angular_regions, angular_count = _merge_support_intervals(
            angular_intervals,
            angular_active,
        )
        angular_active = jnp.arange(angular_regions.shape[0]) < angular_count
    finite = jnp.isfinite(image_limb.real) & jnp.isfinite(image_limb.imag)
    valid = (image_mask & finite).reshape(-1)
    radius = jnp.abs(image_limb).reshape(-1)
    angle = jnp.mod(jnp.angle(image_limb), two_pi).reshape(-1)
    in_radial = (
        radial_active[:, None]
        & (radius[None, :] >= radial_regions[:, 0, None])
        & (radius[None, :] <= radial_regions[:, 1, None])
    )
    in_angular = (
        angular_active[:, None]
        & (angle[None, :] >= angular_regions[:, 0, None])
        & (angle[None, :] <= angular_regions[:, 1, None])
    )
    occupied = jnp.any(
        in_radial[:, None, :]
        & in_angular[None, :, :]
        & valid[None, None, :],
        axis=2,
    )
    if include_segments:
        branch_radius = jnp.abs(image_limb)
        following_radius = jnp.roll(branch_radius, -1, axis=1)
        segment_radial_lower = jnp.minimum(branch_radius, following_radius)
        segment_radial_upper = jnp.maximum(branch_radius, following_radius)
        branch_angle = jnp.mod(jnp.angle(image_limb), two_pi)
        following_limb = jnp.roll(image_limb, -1, axis=1)
        angle_step = jnp.angle(following_limb * jnp.conjugate(image_limb))
        following_angle = branch_angle + angle_step
        segment_angle_lower = jnp.minimum(branch_angle, following_angle)
        segment_angle_upper = jnp.maximum(branch_angle, following_angle)
        image_count = jnp.sum(image_mask, axis=0, dtype=jnp.int32)
        same_multiplicity = image_count == jnp.roll(image_count, -1)
        segment_active = (
            image_mask
            & jnp.roll(image_mask, -1, axis=1)
            & same_multiplicity[None, :]
        )

        radial_intersection = (
            radial_active[:, None, None]
            & (radial_regions[:, 1, None, None] >= segment_radial_lower)
            & (radial_regions[:, 0, None, None] <= segment_radial_upper)
        )
        angular_intersection = jnp.zeros(
            (
                angular_regions.shape[0],
                image_limb.shape[0],
                image_limb.shape[1],
            ),
            dtype=bool,
        )
        for shift in (-2.0, 0.0, 2.0):
            shifted_lower = angular_regions[:, 0] + shift * jnp.pi
            shifted_upper = angular_regions[:, 1] + shift * jnp.pi
            angular_intersection = angular_intersection | (
                angular_active[:, None, None]
                & (shifted_upper[:, None, None] >= segment_angle_lower)
                & (shifted_lower[:, None, None] <= segment_angle_upper)
            )
        segment_occupied = jnp.any(
            radial_intersection[:, None, :, :]
            & angular_intersection[None, :, :, :]
            & segment_active[None, None, :, :],
            axis=(2, 3),
        )
        occupied = occupied | segment_occupied
    if include_enclosed:
        radial_centres = jnp.mean(radial_regions, axis=1)
        angular_centres = jnp.mean(angular_regions, axis=1)
        cell_centres = (
            radial_centres[:, None]
            * jnp.exp(1.0j * angular_centres[None, :])
        ).reshape(-1)
        branch_points = image_limb[:, :, None]
        following_points = jnp.roll(image_limb, -1, axis=1)[:, :, None]
        image_count = jnp.sum(image_mask, axis=0, dtype=jnp.int32)
        same_multiplicity = image_count == jnp.roll(image_count, -1)
        edge_active = (
            image_mask
            & jnp.roll(image_mask, -1, axis=1)
            & same_multiplicity[None, :]
        )[:, :, None]
        turn = jnp.angle(
            (following_points - cell_centres[None, None, :])
            * jnp.conjugate(branch_points - cell_centres[None, None, :])
        )
        winding = jnp.sum(jnp.where(edge_active, turn, 0.0), axis=1)
        branch_enclosed = jnp.any(jnp.abs(winding) > jnp.pi, axis=0)
        stitched_enclosed = jnp.abs(jnp.sum(winding, axis=0)) > jnp.pi
        enclosed = (branch_enclosed | stitched_enclosed).reshape(occupied.shape)
        candidate_active = radial_active[:, None] & angular_active[None, :]
        occupied = occupied | (candidate_active & enclosed)
    if not prune_products:
        occupied = radial_active[:, None] & angular_active[None, :]
    flat_occupied = occupied.reshape(-1)
    indices = jnp.nonzero(flat_occupied, size=capacity, fill_value=-1)[0]
    safe_indices = jnp.maximum(indices, 0)
    radial_index = safe_indices // angular_regions.shape[0]
    angular_index = safe_indices % angular_regions.shape[0]
    active = indices >= 0
    selected_radial = jnp.where(
        active[:, None],
        radial_regions[radial_index],
        jnp.zeros((capacity, 2), dtype=image_limb.real.dtype),
    )
    selected_angular = jnp.where(
        active[:, None],
        angular_regions[angular_index],
        jnp.zeros((capacity, 2), dtype=image_limb.real.dtype),
    )
    return PolarProductRegions(
        selected_radial,
        selected_angular,
        active,
        jnp.sum(flat_occupied, dtype=jnp.int32) > capacity,
    )


def build_radial_support(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_limb: int,
    include_all_roots: bool | Array = False,
) -> RadialSupport:
    """Trace the source limb and construct its radial-envelope union."""

    image_limb, image_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
        include_all_roots=include_all_roots,
    )
    return radial_support_from_limb(image_limb, image_mask)


__all__ = [
    "LimbTrace",
    "RadialSupport",
    "Support1D",
    "build_radial_support",
    "limb_edge_activity_from_masks",
    "make_support_1d",
    "radial_support_from_limb",
    "tracked_limb_neighbors",
    "trace_binary_source_limb",
    "trace_binary_source_limb_record",
    "unwrap_support_1d",
]
