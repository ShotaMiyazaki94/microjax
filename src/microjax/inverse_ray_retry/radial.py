"""Topology-preserving radial support for polar inverse ray shooting.

The legacy dense angular grid needed small rectangles in ``(r, theta)`` to
avoid evaluating mostly empty pixels.  Once the angular source boundary is
solved exactly on every ring, angular rectangles are unnecessary and can even
drop a disconnected image component.  This module instead projects every
sampled source-limb root slot onto radius and returns the union of those radial
intervals.
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

# Root continuation reduced the current broad binary-lens stress maximum from
# 199 false extrema to 29 physical/conservative candidates.  A capacity of 64
# keeps more than a factor-of-two headroom across the stress and caustic-map
# suites without carrying the former false-extremum buffers through every GPU
# batch. Noisy raw-candidate overflow is coarsened to branch-global extrema so
# that a bounded comparison value remains available, but it is now reported as
# ``RADIAL_CAPACITY``.  A caller may clear that bit only after a larger static
# kernel supplies a clean nested-sampling certificate.
_BREAKPOINT_CAPACITY = 64
RADIAL_INTERVAL_CAPACITY = 64
RADIAL_RETRY_BREAKPOINT_CAPACITY = 128
RADIAL_RETRY_INTERVAL_CAPACITY = 144
# Image-local charts can expose hundreds of numerical validity transitions
# around a caustic-enclosing annulus even after branch grouping.  This kernel is
# reached only after both global stages fail and therefore trades more static
# workspace for a fail-closed final rescue.
RADIAL_LOCAL_RETRY_BREAKPOINT_CAPACITY = 512
RADIAL_LOCAL_RETRY_INTERVAL_CAPACITY = 320
_TOPOLOGY_MOTION_MARGIN_SAFETY = 4.0
_BINARY_ASSIGNMENT_PERMUTATIONS = jnp.asarray(
    tuple(permutations(range(5))), dtype=jnp.int32
)


class RadialTopology(NamedTuple):
    """Fixed-shape topology intervals and construction diagnostics."""

    intervals: Array
    n_intervals: Array
    status: Array
    n_candidates_raw: Array
    n_intervals_raw: Array


class LocalImageCharts(NamedTuple):
    """Fixed-capacity disjoint local frames built from tracked limb images."""

    image_limb: Array
    mask_limb: Array
    centers: Array
    radii: Array
    branch_labels: Array
    active: Array
    status: Array


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

    def assignment(previous_roots, previous_mask, roots, mask):
        if image_limb.shape[0] == 5:
            # Greedy matching is order-dependent and can switch two close
            # binary roots differently when midpoint limb samples are inserted.
            # Five roots admit all 5! assignments as one small static array;
            # selecting the minimum total displacement makes N/2N continuation
            # consistent without a host-side Hungarian solve.
            candidates = roots[_BINARY_ASSIGNMENT_PERMUTATIONS]
            candidate_masks = mask[_BINARY_ASSIGNMENT_PERMUTATIONS]
            distance = jnp.abs(candidates - previous_roots[None, :])
            distance_scale = jnp.maximum(
                jnp.max(jnp.where(jnp.isfinite(distance), distance, 0.0)),
                jnp.finfo(distance.dtype).tiny,
            )
            normalized_distance = jnp.sum(
                jnp.where(jnp.isfinite(distance), distance / distance_scale, 1.0),
                axis=1,
            )
            mask_changes = jnp.sum(
                candidate_masks != previous_mask[None, :], axis=1
            )
            # One mask mismatch costs more than the largest possible sum of
            # normalized geometric distances.  Thus physical roots remain
            # physical branches whenever the image count is unchanged; at an
            # actual caustic crossing the unavoidable 3<->5 transition is then
            # resolved by minimum total motion.
            cost = (
                (image_limb.shape[0] + 1) * mask_changes
                + normalized_distance
            )
            return _BINARY_ASSIGNMENT_PERMUTATIONS[jnp.argmin(cost)]
        return match_points(previous_roots, roots)

    def match_sample(previous, sample):
        previous_roots, previous_mask = previous
        roots, mask = sample
        indices = assignment(previous_roots, previous_mask, roots, mask)
        matched_roots = roots[indices]
        matched_mask = mask[indices]
        return (matched_roots, matched_mask), (matched_roots, matched_mask)

    _, (remaining_roots, remaining_masks) = jax.lax.scan(
        match_sample,
        (first_roots, first_mask),
        (roots_by_sample[1:], masks_by_sample[1:]),
    )
    tracked_roots = jnp.concatenate(
        (first_roots[None, :], remaining_roots), axis=0
    )
    tracked_masks = jnp.concatenate(
        (first_mask[None, :], remaining_masks), axis=0
    )
    return jnp.moveaxis(tracked_roots, 0, 1), jnp.moveaxis(
        tracked_masks, 0, 1
    )


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
        lens_masses[:, None, None]
        / (
            conjugate_z[None, :, :]
            - jnp.conjugate(lens_positions)[:, None, None]
        )
        ** 2,
        axis=0,
    )
    source_derivative = (
        1j
        * jnp.asarray(rho, dtype=real_dtype)
        * jnp.exp(1j * angles)[None, :]
    )
    determinant = 1.0 - jnp.abs(shear) ** 2
    nonsingular = determinant != 0.0
    safe_determinant = jnp.where(nonsingular, determinant, 1.0)
    image_derivative = (
        source_derivative - shear * jnp.conjugate(source_derivative)
    ) / safe_determinant
    radial_origin = jnp.asarray(radial_origin, dtype=image_limb.dtype)
    if radial_origin.ndim == image_limb.ndim - 1:
        radial_origin = radial_origin[..., None]
    local_image = image_limb - radial_origin
    radial_derivative = jnp.real(
        jnp.conjugate(local_image) * image_derivative
    ) / jnp.maximum(jnp.abs(local_image), jnp.finfo(real_dtype).tiny)
    sample_step = 2.0 * jnp.pi / jnp.maximum(
        image_limb.shape[1] - 1, 1
    )
    local_margin = sample_step * jnp.abs(radial_derivative)
    derivative_valid = (
        mask_limb & jnp.isfinite(local_margin) & nonsingular
    )
    return jnp.max(
        jnp.where(derivative_valid, local_margin, 0.0), axis=1
    )


def build_local_image_charts(
    image_limb: Array,
    mask_limb: Array,
    rho: float,
    *,
    margin_r: float,
    shifted: Array,
    a: Array,
    e1: Array,
    interior_images: Array | None = None,
    interior_mask: Array | None = None,
    prefer_interior_anchor: bool = False,
) -> LocalImageCharts:
    """Cluster binary source-limb images into disjoint local polar charts.

    One initial disk is built for each of the five tracked binary root slots.
    Overlapping disks, including transient fold pairs with the same validity
    pattern, are merged with five static passes.  When point-source images of
    the source centre are supplied, each chart is anchored on one of those
    guaranteed-interior points instead of the mean of a possibly non-convex
    boundary. ``prefer_interior_anchor`` is a fixed-shape retry mode for a
    non-annular transient group containing multiple source-centre images: it
    selects one of those guaranteed-interior images instead of their mean.
    Annuli retain the near-lens mean in both modes. The resulting enclosing
    disks are pairwise disjoint;
    integrating one chart per active representative can therefore cover all
    sampled image components without double counting.
    """

    if (interior_images is None) != (interior_mask is None):
        raise ValueError(
            "interior_images and interior_mask must be supplied together"
        )

    image_limb, mask_limb = track_limb_images(image_limb, mask_limb)
    finite = jnp.isfinite(image_limb.real) & jnp.isfinite(image_limb.imag)
    valid = mask_limb & finite
    branch_active = jnp.any(valid, axis=1)
    branch_count = jnp.sum(valid, axis=1)
    safe_count = jnp.maximum(branch_count, 1)
    branch_centers = jnp.sum(
        jnp.where(valid, image_limb, 0.0 + 0.0j), axis=1
    ) / safe_count
    branch_margin = _lens_branch_margin(
        image_limb,
        valid,
        rho,
        shifted,
        jnp.asarray([a, -a]),
        jnp.asarray([e1, 1.0 - e1]),
        branch_centers,
    )
    branch_margin = _TOPOLOGY_MOTION_MARGIN_SAFETY * branch_margin
    base_margin = jnp.asarray(margin_r, dtype=image_limb.real.dtype) * jnp.asarray(
        rho, dtype=image_limb.real.dtype
    )
    branch_radii = jnp.max(
        jnp.where(
            valid,
            jnp.abs(image_limb - branch_centers[:, None]),
            0.0,
        ),
        axis=1,
    ) + jnp.maximum(base_margin, branch_margin)

    capacity = image_limb.shape[0]
    slots = jnp.arange(capacity, dtype=jnp.int32)
    sentinel = jnp.int32(capacity)
    labels = jnp.where(branch_active, slots, sentinel)
    transient = branch_active & ~jnp.all(valid, axis=1)
    same_validity = jnp.all(
        valid[:, None, :] == valid[None, :, :], axis=2
    )
    forced_pair = same_validity & transient[:, None] & transient[None, :]
    initial_overlap = (
        jnp.abs(branch_centers[:, None] - branch_centers[None, :])
        <= branch_radii[:, None] + branch_radii[None, :]
    ) & branch_active[:, None] & branch_active[None, :]
    initial_adjacency = initial_overlap | forced_pair

    def propagate_branch_labels(current, adjacency):
        neighbour_labels = jnp.where(
            adjacency, current[None, :], sentinel
        )
        return jnp.where(
            branch_active,
            jnp.min(neighbour_labels, axis=1),
            sentinel,
        )

    for _ in range(capacity):
        labels = propagate_branch_labels(labels, initial_adjacency)

    if interior_images is not None:
        interior_images = jnp.asarray(interior_images, dtype=image_limb.dtype)
        interior_mask = jnp.asarray(interior_mask, dtype=bool)
        if interior_images.ndim != 1 or interior_mask.shape != interior_images.shape:
            raise ValueError(
                "interior_images and interior_mask must have the same 1D shape"
            )
        interior_valid = (
            interior_mask
            & jnp.isfinite(interior_images.real)
            & jnp.isfinite(interior_images.imag)
        )
        distance_to_branch = jnp.min(
            jnp.where(
                valid[None, :, :],
                jnp.abs(
                    interior_images[:, None, None]
                    - image_limb[None, :, :]
                ),
                jnp.inf,
            ),
            axis=2,
        )
        nearest_branch = jnp.argmin(distance_to_branch, axis=1)
        interior_valid = interior_valid & jnp.isfinite(
            jnp.min(distance_to_branch, axis=1)
        )

    def chart_geometry(current_labels):
        membership = (
            current_labels[:, None] == slots[None, :]
        ) & branch_active[:, None]
        point_membership = membership.T[:, :, None] & valid[None, :, :]
        point_count = jnp.sum(point_membership, axis=(1, 2))
        safe_point_count = jnp.maximum(point_count, 1)
        boundary_centers = jnp.sum(
            jnp.where(
                point_membership,
                image_limb[None, :, :],
                0.0 + 0.0j,
            ),
            axis=(1, 2),
        ) / safe_point_count
        boundary_radii = jnp.max(
            jnp.where(
                point_membership,
                jnp.abs(
                    image_limb[None, :, :]
                    - boundary_centers[:, None, None]
                ),
                0.0,
            ),
            axis=(1, 2),
        )
        centers = boundary_centers
        if interior_images is not None:
            interior_labels = current_labels[nearest_branch]
            interior_membership = (
                interior_labels[None, :] == slots[:, None]
            ) & interior_valid[None, :]
            interior_distance = jnp.where(
                interior_membership,
                jnp.abs(
                    interior_images[None, :] - boundary_centers[:, None]
                ),
                jnp.inf,
            )
            nearest_interior = jnp.argmin(interior_distance, axis=1)
            interior_count = jnp.sum(interior_membership, axis=1)
            # One source-centre image gives a guaranteed interior anchor for a
            # simply connected image component.  Multiple centre images in
            # one chart identify a caustic-enclosing component (commonly an
            # annulus around a lens).  Its sampled boundary mean moves with
            # Nlimb and is a poorly conditioned polar origin for a thin
            # Einstein ring.  The mean of the enclosed source-centre images is
            # sampling-independent and stays close to, but not exactly on, the
            # lens singularity; anchoring on either image alone would create a
            # huge off-centre disk.
            has_unique_interior = interior_count == 1
            group_has_transient = jnp.any(
                membership.T & transient[None, :], axis=1
            )
            lens_centers = jnp.asarray(
                [shifted + a, shifted - a], dtype=image_limb.dtype
            )
            nearest_lens = jnp.argmin(
                jnp.abs(
                    lens_centers[None, :] - boundary_centers[:, None]
                ),
                axis=1,
            )
            unique_interior_centers = interior_images[nearest_interior]
            transient_centers = 0.5 * (
                unique_interior_centers + lens_centers[nearest_lens]
            )
            unique_centers = jnp.where(
                group_has_transient,
                transient_centers,
                unique_interior_centers,
            )
            interior_centers = jnp.sum(
                jnp.where(
                    interior_membership,
                    interior_images[None, :],
                    0.0 + 0.0j,
                ),
                axis=1,
            ) / jnp.maximum(interior_count, 1)
            # Multiple source-centre images in one branch group have two
            # geometrically different meanings.  An annulus encloses a lens;
            # its interior-image mean is a stable near-lens polar origin.  A
            # transient fold pair does not enclose a lens.  Its two interior
            # images lie in the physical image while their mean can lie in the
            # gap and map outside the source, which made the local area
            # oscillate with Nlimb.  Detect the annular case from the boundary
            # disk itself.  Otherwise retain the nearest guaranteed-interior
            # point-source image as the chart origin.
            lens_enclosed = jnp.any(
                jnp.abs(
                    lens_centers[None, :] - boundary_centers[:, None]
                )
                <= boundary_radii[:, None],
                axis=1,
            )
            if prefer_interior_anchor:
                multiple_centers = jnp.where(
                    lens_enclosed,
                    interior_centers,
                    unique_interior_centers,
                )
            else:
                multiple_centers = interior_centers
            centers = jnp.where(
                has_unique_interior,
                unique_centers,
                jnp.where(
                    interior_count > 1, multiple_centers, boundary_centers
                ),
            )
        radii = jnp.max(
            jnp.where(
                point_membership,
                jnp.abs(image_limb[None, :, :] - centers[:, None, None]),
                0.0,
            ),
            axis=(1, 2),
        )
        group_margin = jnp.max(
            jnp.where(membership.T, jnp.maximum(base_margin, branch_margin), 0.0),
            axis=1,
        )
        active = point_count > 0
        return centers, radii + group_margin, active

    for _ in range(capacity):
        centers, radii, chart_active = chart_geometry(labels)
        chart_overlap = (
            jnp.abs(centers[:, None] - centers[None, :])
            <= radii[:, None] + radii[None, :]
        ) & chart_active[:, None] & chart_active[None, :]
        safe_labels = jnp.minimum(labels, capacity - 1)
        reachable = chart_overlap[safe_labels]
        labels = jnp.where(
            branch_active,
            jnp.min(jnp.where(reachable, slots[None, :], sentinel), axis=1),
            sentinel,
        )

    centers, radii, chart_active = chart_geometry(labels)
    final_overlap = (
        jnp.abs(centers[:, None] - centers[None, :])
        <= radii[:, None] + radii[None, :]
    ) & chart_active[:, None] & chart_active[None, :]
    distinct_overlap = final_overlap & ~jnp.eye(capacity, dtype=bool)
    status = jnp.where(
        ~jnp.any(branch_active) | jnp.any(distinct_overlap),
        jnp.int32(RADIAL_CAPACITY),
        jnp.int32(RADIAL_OK),
    )
    return LocalImageCharts(
        image_limb,
        valid,
        centers,
        radii,
        labels,
        chart_active,
        status,
    )


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
                updated = jnp.asarray(
                    [regions_[last_index, 0], jnp.maximum(last[1], hi)]
                )
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
    same_validity = jnp.all(
        valid[:, None, :] == valid[None, :, :], axis=2
    )
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
        jnp.asarray(margin_r, dtype=radii.dtype)
        * jnp.asarray(rho, dtype=radii.dtype),
        0.0,
    )
    if branch_margin is not None:
        margin = jnp.maximum(margin, branch_margin)
    lower = jnp.where(active, jnp.maximum(lower - margin, 0.0), 0.0)
    upper = jnp.where(active, upper + margin, 0.0)

    innermost = jnp.argmin(jnp.where(active, lower, jnp.inf))
    lower = lower.at[innermost].set(
        jnp.where(origin_inside & jnp.any(active), 0.0, lower[innermost])
    )
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

    radial_samples = image_limb - jnp.asarray(
        radial_origin, dtype=image_limb.dtype
    )
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
        slope_floor = jnp.maximum(
            slope_floor, jnp.sqrt(jnp.finfo(radii.dtype).eps) * scale
        )
    turning = (
        valid
        & previous_valid
        & next_valid
        & (left_slope * right_slope <= 0.0)
        & ((jnp.abs(left_slope) + jnp.abs(right_slope)) > slope_floor)
    )
    segment_endpoint = valid & ~(previous_valid & next_valid)
    sample_slots = jnp.arange(radii.shape[1])[None, :]
    branch_min_index = jnp.argmin(
        jnp.where(valid, radii, jnp.inf), axis=1
    )
    branch_max_index = jnp.argmax(
        jnp.where(valid, radii, -jnp.inf), axis=1
    )
    # Select one representative for each global extremum.  Near a very
    # low-mass companion, polynomial roots can be constant to machine
    # precision over many limb samples; equality against branch_min/max would
    # otherwise spend the entire static buffer on duplicate breakpoints.
    global_extremum = valid & (
        (sample_slots == branch_min_index[:, None])
        | (sample_slots == branch_max_index[:, None])
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
    candidate_values = radii.ravel()[candidate_indices]
    candidate_valid = jnp.arange(breakpoint_capacity) < jnp.minimum(
        n_candidates, breakpoint_capacity
    )

    region_active = regions[:, 1] > regions[:, 0]
    bound_values = regions.ravel()
    bound_valid = jnp.repeat(region_active, 2)
    values = jnp.concatenate((candidate_values, bound_values))
    values_valid = jnp.concatenate((candidate_valid, bound_valid))
    values = jnp.sort(jnp.where(values_valid, values, jnp.inf))
    finite_values = jnp.isfinite(values)
    previous = jnp.concatenate((jnp.asarray([-jnp.inf]), values[:-1]))
    unique_tolerance = (
        256.0
        * jnp.finfo(radii.dtype).eps
        * jnp.maximum(1.0, jnp.abs(values))
    )
    unique = finite_values & ((values - previous) > unique_tolerance)
    unique_indices = jnp.nonzero(
        unique, size=values.size, fill_value=values.size - 1
    )[0]
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
    interval_valid = (
        unique_slots[:-1]
        & unique_slots[1:]
        & (interval_hi > interval_lo)
        & covered
    )
    n_intervals_raw = jnp.sum(interval_valid, dtype=jnp.int32)
    interval_indices = jnp.nonzero(
        interval_valid,
        size=interval_capacity,
        fill_value=interval_valid.size - 1,
    )[0]
    intervals = jnp.stack(
        (interval_lo[interval_indices], interval_hi[interval_indices]), axis=1
    )
    n_intervals = jnp.minimum(n_intervals_raw, interval_capacity)
    intervals = jnp.where(
        (jnp.arange(interval_capacity) < n_intervals)[:, None],
        intervals,
        0.0,
    )
    overflow = raw_candidate_overflow | (n_candidates > breakpoint_capacity) | (
        n_intervals_raw > interval_capacity
    )
    status = jnp.where(
        overflow, jnp.int32(RADIAL_CAPACITY), jnp.int32(RADIAL_OK)
    )
    return RadialTopology(
        intervals,
        n_intervals,
        status,
        n_candidates_raw,
        n_intervals_raw,
    )
