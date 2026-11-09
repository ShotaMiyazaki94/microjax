"""Build topology-preserving radial support from tracked image limbs.

The legacy dense angular grid needed small rectangles in ``(r, theta)`` to
avoid evaluating mostly empty pixels.  Once the angular source boundary is
solved exactly on every ring, angular rectangles are unnecessary and can even
drop a disconnected image component.  This module instead projects every
sampled source-limb root slot onto radius and returns the union of those radial
intervals.  Local-chart grouping belongs here because it partitions image
geometry before any quadrature rule is selected.
"""

from __future__ import annotations

from itertools import permutations
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from microjax.utils import match_points

Array = jnp.ndarray

RADIAL_OK = 0
RADIAL_CAPACITY = 8
RADIAL_TOLERANCE = 16
RADIAL_TOPOLOGY = 32

# Root continuation reduced the stress maximum from 199 false extrema to 29
# physical/conservative candidates. Capacity 64 retains >2x headroom. Overflow
# is coarsened to branch extrema and reported as RADIAL_CAPACITY until a larger
# nested kernel supplies a clean certificate.
_BREAKPOINT_CAPACITY = 64
RADIAL_INTERVAL_CAPACITY = 64
RADIAL_RETRY_BREAKPOINT_CAPACITY = 128
RADIAL_RETRY_INTERVAL_CAPACITY = 144
_TOPOLOGY_MOTION_MARGIN_SAFETY = 4.0
_BINARY_ASSIGNMENT_PERMUTATIONS = jnp.asarray(tuple(permutations(range(5))), dtype=jnp.int32)


class RadialTopology(NamedTuple):
    """Fixed-shape topology intervals and construction diagnostics."""

    intervals: Array
    n_intervals: Array
    status: Array
    n_candidates_raw: Array
    n_intervals_raw: Array


def _match_limb_root_indices(previous_roots, previous_mask, roots, mask):
    """Return a stable continuation assignment for one limb sample."""

    if previous_roots.shape[0] != 5:
        return match_points(previous_roots, roots)

    # Five binary roots admit an exhaustive 5! assignment. Mask changes get a
    # larger cost than any normalized geometric displacement, so a physical
    # branch remains physical unless an actual 3<->5 caustic transition occurs.
    candidates = roots[_BINARY_ASSIGNMENT_PERMUTATIONS]
    candidate_masks = mask[_BINARY_ASSIGNMENT_PERMUTATIONS]
    distance = jnp.abs(candidates - previous_roots[None, :])
    distance_scale = jnp.maximum(
        jnp.max(jnp.where(jnp.isfinite(distance), distance, 0.0)),
        jnp.finfo(distance.dtype).tiny,
    )
    normalized_distance = jnp.sum(jnp.where(jnp.isfinite(distance), distance / distance_scale, 1.0), axis=1)
    mask_changes = jnp.sum(candidate_masks != previous_mask[None, :], axis=1)
    cost = (previous_roots.shape[0] + 1) * mask_changes + normalized_distance
    return _BINARY_ASSIGNMENT_PERMUTATIONS[jnp.argmin(cost)]


def track_limb_images(image_limb: Array, mask_limb: Array) -> Tuple[Array, Array]:
    """Match polynomial roots between adjacent source-limb samples.

    Point-source roots have no public ordering guarantee.  Nearest-neighbour
    continuation prevents a change in root ordering from turning two compact
    image branches into one artificially broad radial interval.  All roots,
    including temporarily non-physical ones, participate in the matching;
    ``mask_limb`` is permuted by the same indices and remains the authority for
    which samples bound the finite-source preimage.
    """

    if image_limb.ndim != 2 or mask_limb.shape != image_limb.shape:
        raise ValueError("image_limb and mask_limb must have the same 2D shape")

    roots_by_sample = jnp.moveaxis(image_limb, 0, 1)
    masks_by_sample = jnp.moveaxis(mask_limb, 0, 1)
    first_roots = roots_by_sample[0]
    first_mask = masks_by_sample[0]

    def match_sample(previous, sample):
        previous_roots, previous_mask = previous
        roots, mask = sample
        indices = _match_limb_root_indices(previous_roots, previous_mask, roots, mask)
        matched_roots = roots[indices]
        matched_mask = mask[indices]
        return (matched_roots, matched_mask), (matched_roots, matched_mask)

    _, (remaining_roots, remaining_masks) = jax.lax.scan(
        match_sample,
        (first_roots, first_mask),
        (roots_by_sample[1:], masks_by_sample[1:]),
    )
    tracked_roots = jnp.concatenate((first_roots[None, :], remaining_roots), axis=0)
    tracked_masks = jnp.concatenate((first_mask[None, :], remaining_masks), axis=0)
    return jnp.moveaxis(tracked_roots, 0, 1), jnp.moveaxis(tracked_masks, 0, 1)


def _lens_branch_margin(
    image_limb: Array,
    mask_limb: Array,
    rho: float,
    shifted: Array,
    lens_positions: Array,
    lens_masses: Array,
    radial_origin: Array = 0.0 + 0.0j,
) -> Array:
    """Estimate image-plane motion between adjacent source-limb samples.

    A fixed multiple of ``rho`` is not an image-plane distance near a caustic.
    Implicit differentiation of the point-lens equation supplies ``dz/dphi``
    along the circular source limb for any fixed lens-position/mass arrays.
    One full sample-step of the largest finite radial speed on each tracked
    image branch is retained as a support margin; the historical scalar margin
    remains a lower bound.
    """

    real_dtype = image_limb.real.dtype
    angles = jnp.linspace(
        0.0,
        2.0 * jnp.pi,
        image_limb.shape[1],
        dtype=real_dtype,
    )
    z_midpoint = image_limb - shifted
    conjugate_z = jnp.conjugate(z_midpoint)
    shear = jnp.sum(
        lens_masses[:, None, None] / (conjugate_z[None, :, :] - jnp.conjugate(lens_positions)[:, None, None]) ** 2,
        axis=0,
    )
    source_derivative = 1j * jnp.asarray(rho, dtype=real_dtype) * jnp.exp(1j * angles)[None, :]
    shear_abs = jnp.abs(shear)
    determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
    nonsingular = determinant != 0.0
    safe_determinant = jnp.where(nonsingular, determinant, 1.0)
    image_derivative = (source_derivative - shear * jnp.conjugate(source_derivative)) / safe_determinant
    radial_origin = jnp.asarray(radial_origin, dtype=image_limb.dtype)
    if radial_origin.ndim == image_limb.ndim - 1:
        radial_origin = radial_origin[..., None]
    local_image = image_limb - radial_origin
    radial_derivative = jnp.real(jnp.conjugate(local_image) * image_derivative) / jnp.maximum(
        jnp.abs(local_image), jnp.finfo(real_dtype).tiny
    )
    sample_step = 2.0 * jnp.pi / jnp.maximum(image_limb.shape[1] - 1, 1)
    local_margin = sample_step * jnp.abs(radial_derivative)
    derivative_valid = mask_limb & jnp.isfinite(local_margin) & nonsingular
    return jnp.max(jnp.where(derivative_valid, local_margin, 0.0), axis=1)


def _merge_radial_intervals(lower: Array, upper: Array, active: Array) -> Array:
    """Return a fixed-shape sorted union of radial intervals."""

    capacity = lower.shape[0]
    sort_key = jnp.where(active, lower, jnp.inf)
    order = jnp.argsort(sort_key)
    lower = lower[order]
    upper = upper[order]
    active = active[order]
    output = jnp.zeros((capacity, 2), dtype=lower.dtype)

    def merge_one(state, interval):
        regions, count = state
        lo, hi, is_active = interval

        def insert_or_merge(current):
            current_regions, current_count = current
            last_index = jnp.maximum(current_count - 1, 0)
            last = current_regions[last_index]
            overlaps = (current_count > 0) & (lo <= last[1])

            def merge(values):
                regions_, count_ = values
                updated = jnp.asarray([regions_[last_index, 0], jnp.maximum(last[1], hi)])
                return regions_.at[last_index].set(updated), count_

            def append(values):
                regions_, count_ = values
                regions_ = regions_.at[count_].set(jnp.asarray([lo, hi]))
                return regions_, count_ + jnp.int32(1)

            return jax.lax.cond(overlaps, merge, append, current)

        state = jax.lax.cond(is_active, insert_or_merge, lambda value: value, state)
        return state, None

    (output, _), _ = jax.lax.scan(
        merge_one,
        (output, jnp.int32(0)),
        (lower, upper, active),
    )
    return output


def _polish_sampled_turning_radii(radii: Array, turning: Array, scale: Array) -> Array:
    """Place sampled radial extrema at the vertex of a local parabola.

    The source-limb samples are uniformly spaced in angle. A three-point
    vertex estimate therefore removes the leading sampling offset without a
    new lens solve. Only certified interior turning points are changed; flat
    or extrapolated fits retain the sampled radius.
    """

    previous = jnp.roll(radii, 1, axis=1)
    following = jnp.roll(radii, -1, axis=1)
    curvature = previous - 2.0 * radii + following
    floor = 256.0 * jnp.finfo(radii.dtype).eps * scale
    safe = turning & (jnp.abs(curvature) > floor)
    offset = jnp.where(safe, 0.5 * (previous - following) / curvature, 0.0)
    safe = safe & (jnp.abs(offset) <= 1.0)
    vertex = radii + 0.25 * (following - previous) * offset
    return jnp.where(safe & jnp.isfinite(vertex), vertex, radii)


def _regions_from_samples(
    image_samples: Array,
    mask_samples: Array,
    rho: float,
    margin_r: float,
    origin_inside: Array,
    branch_margin: Array | None = None,
) -> Array:
    """Project sampled root slots and merge their radial ranges."""

    radii = jnp.abs(image_samples)
    finite = jnp.isfinite(radii)
    valid = mask_samples & finite
    active = jnp.any(valid, axis=1)

    lower = jnp.min(jnp.where(valid, radii, jnp.inf), axis=1)
    upper = jnp.max(jnp.where(valid, radii, -jnp.inf), axis=1)
    # A fold-caustic crossing creates or destroys two images together.  Their
    # two source-limb tracks jointly bound one connected image component, so
    # the radial projection must span both tracks even when their individual
    # projections do not overlap.  For a binary lens there is at most one such
    # transient pair at a source-limb position; identical nontrivial validity
    # patterns therefore identify the pair without a geometric threshold.
    transient = active & ~jnp.all(valid, axis=1)
    same_validity = jnp.all(valid[:, None, :] == valid[None, :, :], axis=2)
    connected = same_validity & transient[:, None] & transient[None, :]
    lower = jnp.where(
        transient,
        jnp.min(jnp.where(connected, lower[None, :], jnp.inf), axis=1),
        lower,
    )
    upper = jnp.where(
        transient,
        jnp.max(jnp.where(connected, upper[None, :], -jnp.inf), axis=1),
        upper,
    )
    margin = jnp.maximum(
        jnp.asarray(margin_r, dtype=radii.dtype) * jnp.asarray(rho, dtype=radii.dtype),
        0.0,
    )
    if branch_margin is not None:
        margin = jnp.maximum(margin, branch_margin)
    lower = jnp.where(active, jnp.maximum(lower - margin, 0.0), 0.0)
    upper = jnp.where(active, upper + margin, 0.0)

    innermost = jnp.argmin(jnp.where(active, lower, jnp.inf))
    lower = lower.at[innermost].set(jnp.where(origin_inside & jnp.any(active), 0.0, lower[innermost]))
    return _merge_radial_intervals(lower, upper, active)


def define_radial_topology(
    image_limb: Array,
    mask_limb: Array,
    rho: float,
    *,
    margin_r: float = 0.5,
    origin_inside: Array = False,
    track_roots: bool = True,
    binary_margin_parameters: tuple[Array, Array, Array] | None = None,
    lens_margin_parameters: tuple[Array, Array, Array] | None = None,
    radial_origin: Array = 0.0 + 0.0j,
    sampled_turning_points: bool = True,
    filter_roundoff_turning_points: bool = False,
    turning_radii_override: Array | None = None,
    breakpoint_capacity: int = _BREAKPOINT_CAPACITY,
    interval_capacity: int = RADIAL_INTERVAL_CAPACITY,
) -> RadialTopology:
    """Split radial support at every sampled topology-changing extremum.

    Besides the outer support ranges, local radial extrema of each sampled root
    slot and endpoints of valid-mask segments are retained.  At
    these radii angular image intervals are born, merge, or disappear, and the
    radial area integrand has square-root endpoint behaviour.  Supplying all of
    them to the quadrature is what makes a low-order endpoint transform reliable
    for caustic-crossing finite sources.

    If raw sampled breakpoints exceed capacity, the topology is conservatively
    coarsened to one global min/max pair per branch so that a bounded comparison
    value can still be evaluated, and ``RADIAL_CAPACITY`` is set.  The bit must
    not be silently discarded: the binary production path clears it only after
    a larger fixed-shape kernel at nested limb sampling is status-clean and
    agrees within the public error budget.  Structural interval overflow is
    likewise explicit.  The limits are algorithmic fixed buffers, not accuracy
    resolutions exposed to users.

    ``lens_margin_parameters`` provides the centre-of-mass displacement,
    midpoint-frame lens positions, and mass fractions for the generic implicit
    image-motion guard. ``binary_margin_parameters`` remains as a compatibility
    shorthand and produces the same two-lens arrays.

    ``turning_radii_override`` may replace sampled parabolic vertices with
    exact radial tangencies in the original ``image_limb`` layout. Non-finite
    entries leave the ordinary sampled estimate unchanged.
    """

    if track_roots:
        image_limb, mask_limb = track_limb_images(image_limb, mask_limb)

    branch_margin = None
    if lens_margin_parameters is not None:
        shifted, lens_positions, lens_masses = lens_margin_parameters
        branch_margin = _lens_branch_margin(
            image_limb,
            mask_limb,
            rho,
            shifted,
            jnp.asarray(lens_positions),
            jnp.asarray(lens_masses),
            radial_origin,
        )
    elif binary_margin_parameters is not None:
        shifted, a, e1 = binary_margin_parameters
        branch_margin = _lens_branch_margin(
            image_limb,
            mask_limb,
            rho,
            shifted,
            jnp.asarray([a, -a]),
            jnp.asarray([e1, 1.0 - e1]),
            radial_origin,
        )
    if branch_margin is not None:
        # The implicit derivative is sampled only on the source-limb grid.  A
        # sharp between-sample minimum near a fold can have a larger speed than
        # either endpoint; one Euler step was observed to miss support at
        # Nlimb=1000 even though Nlimb=500 and 2000 enclosed it.  Four sampled
        # steps are an empirical fixed guard, not a derivative bound; the
        # independent nested-grid area comparison remains the acceptance
        # certificate.
        branch_margin = _TOPOLOGY_MOTION_MARGIN_SAFETY * branch_margin

    radial_samples = image_limb - jnp.asarray(radial_origin, dtype=image_limb.dtype)
    regions = _regions_from_samples(
        radial_samples,
        mask_limb,
        rho,
        margin_r,
        origin_inside,
        branch_margin=branch_margin,
    )
    # EA uses a fixed ordered initialization, which originally made root slots
    # look stable enough to skip this continuation scan.  Broad GPU validation
    # showed that the remaining swaps create many false extrema and consume the
    # radial error budget.  Tracking is therefore the default; disabling it is
    # retained only for diagnostics and historical comparisons.
    radii = jnp.abs(radial_samples)
    valid = mask_limb & jnp.isfinite(radii)
    previous_r = jnp.roll(radii, 1, axis=1)
    next_r = jnp.roll(radii, -1, axis=1)
    previous_valid = jnp.roll(valid, 1, axis=1)
    next_valid = jnp.roll(valid, -1, axis=1)
    left_slope = radii - previous_r
    right_slope = next_r - radii
    scale = jnp.maximum(jnp.max(jnp.where(valid, radii, 0.0)), 1.0)
    slope_floor = 64.0 * jnp.finfo(radii.dtype).eps * scale
    if filter_roundoff_turning_points:
        slope_floor = jnp.maximum(slope_floor, jnp.sqrt(jnp.finfo(radii.dtype).eps) * scale)
    turning = (
        valid
        & previous_valid
        & next_valid
        & (left_slope * right_slope <= 0.0)
        & ((jnp.abs(left_slope) + jnp.abs(right_slope)) > slope_floor)
    )
    polished_radii = _polish_sampled_turning_radii(radii, turning, scale)
    if turning_radii_override is not None:
        turning_radii_override = jnp.asarray(turning_radii_override, dtype=radii.dtype)
        if turning_radii_override.shape != radii.shape:
            raise ValueError("turning_radii_override must match image_limb shape")
        polished_radii = jnp.where(
            turning & jnp.isfinite(turning_radii_override),
            turning_radii_override,
            polished_radii,
        )
    segment_endpoint = valid & ~(previous_valid & next_valid)
    sample_slots = jnp.arange(radii.shape[1])[None, :]
    branch_min_index = jnp.argmin(jnp.where(valid, radii, jnp.inf), axis=1)
    branch_max_index = jnp.argmax(jnp.where(valid, radii, -jnp.inf), axis=1)
    # Select one representative for each global extremum.  Near a very
    # low-mass companion, polynomial roots can be constant to machine
    # precision over many limb samples; equality against branch_min/max would
    # otherwise spend the entire static buffer on duplicate breakpoints.
    global_extremum = valid & (
        (sample_slots == branch_min_index[:, None]) | (sample_slots == branch_max_index[:, None])
    )
    base_candidate_mask = segment_endpoint | global_extremum
    candidate_mask = base_candidate_mask
    if sampled_turning_points:
        candidate_mask = candidate_mask | turning
    n_candidates_raw = jnp.sum(candidate_mask, dtype=jnp.int32)
    # A caustic-enclosing annulus can make root continuation alternate between
    # its inner and outer boundary, while an ill-conditioned near-lens root can
    # flicker across the physical-root residual threshold.  Treating all those
    # zig-zags and mask endpoints as physical breakpoints overflows the static
    # buffer.  In that case retain one min/max pair per branch and let adaptive
    # quadrature enforce the error tolerance on the complete support interval.
    raw_candidate_overflow = n_candidates_raw > breakpoint_capacity
    candidate_mask = jnp.where(
        raw_candidate_overflow,
        global_extremum,
        candidate_mask,
    )
    n_candidates = jnp.sum(candidate_mask, dtype=jnp.int32)
    candidate_indices = jnp.nonzero(
        candidate_mask.ravel(),
        size=breakpoint_capacity,
        fill_value=candidate_mask.size - 1,
    )[0]
    candidate_values = polished_radii.ravel()[candidate_indices]
    candidate_valid = jnp.arange(breakpoint_capacity) < jnp.minimum(n_candidates, breakpoint_capacity)

    region_active = regions[:, 1] > regions[:, 0]
    bound_values = regions.ravel()
    bound_valid = jnp.repeat(region_active, 2)
    values = jnp.concatenate((candidate_values, bound_values))
    values_valid = jnp.concatenate((candidate_valid, bound_valid))
    values = jnp.sort(jnp.where(values_valid, values, jnp.inf))
    finite_values = jnp.isfinite(values)
    previous = jnp.concatenate((jnp.asarray([-jnp.inf]), values[:-1]))
    unique_tolerance = 256.0 * jnp.finfo(radii.dtype).eps * jnp.maximum(1.0, jnp.abs(values))
    unique = finite_values & ((values - previous) > unique_tolerance)
    unique_indices = jnp.nonzero(unique, size=values.size, fill_value=values.size - 1)[0]
    n_unique = jnp.sum(unique, dtype=jnp.int32)
    sorted_unique = values[unique_indices]
    unique_slots = jnp.arange(values.size) < n_unique

    interval_lo = sorted_unique[:-1]
    interval_hi = sorted_unique[1:]
    interval_mid = 0.5 * (interval_lo + interval_hi)
    covered = jnp.any(
        (interval_mid[:, None] >= regions[None, :, 0])
        & (interval_mid[:, None] <= regions[None, :, 1])
        & region_active[None, :],
        axis=1,
    )
    interval_valid = unique_slots[:-1] & unique_slots[1:] & (interval_hi > interval_lo) & covered
    n_intervals_raw = jnp.sum(interval_valid, dtype=jnp.int32)
    interval_indices = jnp.nonzero(
        interval_valid,
        size=interval_capacity,
        fill_value=interval_valid.size - 1,
    )[0]
    intervals = jnp.stack((interval_lo[interval_indices], interval_hi[interval_indices]), axis=1)
    n_intervals = jnp.minimum(n_intervals_raw, interval_capacity)
    intervals = jnp.where(
        (jnp.arange(interval_capacity) < n_intervals)[:, None],
        intervals,
        0.0,
    )
    overflow = raw_candidate_overflow | (n_candidates > breakpoint_capacity) | (n_intervals_raw > interval_capacity)
    status = jnp.where(overflow, jnp.int32(RADIAL_CAPACITY), jnp.int32(RADIAL_OK))
    # A topology with no finite algebraic trace (or no finite physical branch)
    # is not an empty lens image.  Mark it explicitly so callers cannot turn a
    # failed root trace into a falsely certified zero-area integral.
    trace_valid = jnp.all(jnp.isfinite(image_limb)) & jnp.any(valid)
    status = jnp.bitwise_or(
        status,
        jnp.where(trace_valid, jnp.int32(RADIAL_OK), jnp.int32(RADIAL_TOPOLOGY)),
    )
    return RadialTopology(
        intervals,
        n_intervals,
        status,
        n_candidates_raw,
        n_intervals_raw,
    )
