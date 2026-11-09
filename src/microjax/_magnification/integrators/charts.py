"""Selection, ownership, and construction of image-local polar charts."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..geometry.topology import (
    RADIAL_CAPACITY,
    RADIAL_INTERVAL_CAPACITY,
    RADIAL_OK,
    RadialTopology,
    define_radial_topology,
    track_limb_images,
)
from ..geometry.lens import BinaryGeometry
from ..roots.level_set import binary_level_set, triple_level_set
from .common import Array

PLANET_CHART_MAX_GLOBAL_ANGLE = 2.0e-2
PLANET_CHART_ZONE_RADII = 8.0
COMPACT_CHART_MAX_GLOBAL_ANGLE = 2.0e-2
COMPACT_CHART_MOTION_SAFETY = 4.0


class ChartedTopology(NamedTuple):
    """Packed global/local topology and fixed-shape ownership metadata."""

    topology: RadialTopology
    interval_parameters: Array
    chart_centers: Array
    chart_radii: Array
    chart_active: Array


def _tracked_branch_geometry(image_limb, mask_limb, rho, margin_r):
    """Track limb roots and return conservative branch disks."""

    image_limb, mask_limb = track_limb_images(image_limb, mask_limb)
    finite = jnp.isfinite(image_limb.real) & jnp.isfinite(image_limb.imag)
    valid = mask_limb & finite
    branch_active = jnp.any(valid, axis=1)
    branch_count = jnp.maximum(jnp.sum(valid, axis=1), 1)
    branch_centers = (
        jnp.sum(jnp.where(valid, image_limb, 0.0 + 0.0j), axis=1) / branch_count
    )
    base_margin = jnp.asarray(margin_r, dtype=image_limb.real.dtype) * jnp.asarray(
        rho, dtype=image_limb.real.dtype
    )
    step_valid = valid & jnp.roll(valid, -1, axis=1)
    step_motion = jnp.abs(jnp.roll(image_limb, -1, axis=1) - image_limb)
    motion_margin = COMPACT_CHART_MOTION_SAFETY * jnp.max(
        jnp.where(step_valid & jnp.isfinite(step_motion), step_motion, 0.0), axis=1
    )
    branch_radii = (
        jnp.max(
            jnp.where(valid, jnp.abs(image_limb - branch_centers[:, None]), 0.0), axis=1
        )
        + base_margin
        + motion_margin
    )
    return image_limb, valid, branch_active, branch_centers, branch_radii


def _triple_compact_mixed_topology(
    image_limb: Array,
    mask_limb: Array,
    rho: float,
    *,
    margin_r: float,
    w_center_shifted: Array,
    origin_inside: Array,
    shifted: Array,
    a: Array,
    e1: Array,
    e2: Array,
    r3_complex: Array,
    lens_margin_parameters,
) -> ChartedTopology:
    """Move spatially isolated small-angle triple images to local charts.

    The source-limb solve is reused. Compact branch disks are grouped with a
    fixed number of label-propagation steps, certified to contain the chart
    centre, and required to be spatially disjoint from every retained global
    branch. Angular interval ownership then prevents double counting when a
    local image and an Einstein-ring image share the same global radius.
    """

    image_limb, valid, branch_active, branch_centers, branch_radii = (
        _tracked_branch_geometry(image_limb, mask_limb, rho, margin_r)
    )
    real_dtype = image_limb.real.dtype
    branch_capacity = image_limb.shape[0]
    slots = jnp.arange(branch_capacity, dtype=jnp.int32)
    sentinel = jnp.int32(branch_capacity)
    angle_scale = branch_radii / jnp.maximum(
        jnp.abs(branch_centers), jnp.finfo(real_dtype).tiny
    )
    compact_branch = branch_active & (angle_scale <= COMPACT_CHART_MAX_GLOBAL_ANGLE)
    labels = jnp.where(compact_branch, slots, sentinel)
    transient = compact_branch & ~jnp.all(valid, axis=1)
    same_validity = jnp.all(valid[:, None, :] == valid[None, :, :], axis=2)
    adjacency = (
        (
            jnp.abs(branch_centers[:, None] - branch_centers[None, :])
            <= branch_radii[:, None] + branch_radii[None, :]
        )
        & compact_branch[:, None]
        & compact_branch[None, :]
    )
    adjacency = adjacency | (same_validity & transient[:, None] & transient[None, :])

    def propagate(current):
        neighbours = jnp.where(adjacency, current[None, :], sentinel)
        return jnp.where(compact_branch, jnp.min(neighbours, axis=1), sentinel)

    for _ in range(branch_capacity):
        labels = propagate(labels)

    def chart_geometry(current_labels):
        membership = (current_labels[:, None] == slots[None, :]) & compact_branch[
            :, None
        ]
        point_membership = membership.T[:, :, None] & valid[None, :, :]
        point_count = jnp.sum(point_membership, axis=(1, 2))
        centers = jnp.sum(
            jnp.where(point_membership, image_limb[None, :, :], 0.0 + 0.0j), axis=(1, 2)
        )
        centers = centers / jnp.maximum(point_count, 1)
        radii = jnp.max(
            jnp.where(
                membership.T,
                jnp.abs(branch_centers[None, :] - centers[:, None])
                + branch_radii[None, :],
                0.0,
            ),
            axis=1,
        )
        return centers, radii, point_count > 0

    for _ in range(branch_capacity):
        chart_centers, chart_radii, chart_active = chart_geometry(labels)
        chart_overlap = (
            (
                jnp.abs(chart_centers[:, None] - chart_centers[None, :])
                <= chart_radii[:, None] + chart_radii[None, :]
            )
            & chart_active[:, None]
            & chart_active[None, :]
        )
        safe_labels = jnp.minimum(labels, branch_capacity - 1)
        reachable = chart_overlap[safe_labels]
        labels = jnp.where(
            compact_branch,
            jnp.min(jnp.where(reachable, slots[None, :], sentinel), axis=1),
            sentinel,
        )

    chart_centers, chart_radii, chart_active = chart_geometry(labels)
    global_branch = branch_active & ~compact_branch
    host_overlap = (
        jnp.abs(chart_centers[:, None] - branch_centers[None, :])
        <= chart_radii[:, None] + branch_radii[None, :]
    ) & global_branch[None, :]
    chart_level = jax.vmap(
        lambda center: triple_level_set(
            center,
            w_center_shifted,
            rho,
            shifted,
            a=a,
            e1=e1,
            e2=e2,
            r3_complex=r3_complex,
        )
    )(chart_centers)
    grouped_angle_scale = chart_radii / jnp.maximum(
        jnp.abs(chart_centers), jnp.finfo(real_dtype).tiny
    )
    chart_active = (
        chart_active
        & (chart_level <= 0.0)
        & (grouped_angle_scale <= COMPACT_CHART_MAX_GLOBAL_ANGLE)
        & ~jnp.any(host_overlap, axis=1)
        & jnp.any(global_branch)
    )
    chart_active = jax.lax.stop_gradient(chart_active)
    chart_centers = jnp.where(chart_active, chart_centers, 0.0 + 0.0j)
    chart_radii = jax.lax.stop_gradient(jnp.where(chart_active, chart_radii, 0.0))
    safe_labels = jnp.minimum(labels, branch_capacity - 1)
    local_branch = compact_branch & (labels < sentinel) & chart_active[safe_labels]
    host_mask = valid & ~local_branch[:, None]
    chart_masks = (
        (labels[:, None, None] == slots[None, :, None])
        & local_branch[:, None, None]
        & valid[:, None, :]
    ).transpose(1, 0, 2)
    host_topology = define_radial_topology(
        image_limb,
        host_mask,
        rho,
        margin_r=margin_r,
        origin_inside=origin_inside,
        track_roots=False,
        lens_margin_parameters=lens_margin_parameters,
    )

    def local_topology(mask, center, center_inside):
        return define_radial_topology(
            image_limb,
            mask,
            rho,
            margin_r=margin_r,
            origin_inside=center_inside,
            track_roots=False,
            lens_margin_parameters=lens_margin_parameters,
            radial_origin=center,
        )

    local_topologies = jax.vmap(local_topology)(
        chart_masks, chart_centers, chart_active
    )
    capacity = host_topology.intervals.shape[0]
    local_interval_active = (
        jnp.arange(capacity)[None, :] < local_topologies.n_intervals[:, None]
    )
    local_outer_radius = jnp.max(
        jnp.where(local_interval_active, local_topologies.intervals[:, :, 1], 0.0),
        axis=1,
    )
    ownership_supported = ~chart_active | (local_outer_radius <= chart_radii)
    final_host_branch = branch_active & ~local_branch
    ownership_overlaps_host = (
        jnp.abs(chart_centers[:, None] - branch_centers[None, :])
        <= local_outer_radius[:, None] + branch_radii[None, :]
    ) & final_host_branch[None, :]
    use_charts = jnp.any(chart_active) & jnp.all(
        ~chart_active
        | (ownership_supported & ~jnp.any(ownership_overlaps_host, axis=1))
    )
    use_charts = jax.lax.stop_gradient(use_charts)
    chart_active = chart_active & use_charts
    chart_centers = jnp.where(chart_active, chart_centers, 0.0 + 0.0j)
    chart_radii = jax.lax.stop_gradient(
        jnp.where(chart_active, local_outer_radius, 0.0)
    )
    host_active = jnp.arange(capacity) < host_topology.n_intervals
    local_active = local_interval_active & chart_active[:, None]
    combined_intervals = jnp.concatenate(
        (host_topology.intervals[None, :, :], local_topologies.intervals), axis=0
    )
    combined_intervals = combined_intervals.reshape(-1, 2)
    combined_centers = jnp.concatenate(
        (jnp.zeros(1, dtype=image_limb.dtype), chart_centers)
    )[:, None]
    combined_centers = jnp.broadcast_to(
        combined_centers, (branch_capacity + 1, capacity)
    ).reshape(-1)
    combined_owners = jnp.concatenate(
        (
            jnp.full((1, capacity), -1, dtype=jnp.int32),
            jnp.broadcast_to(slots[:, None], (branch_capacity, capacity)),
        ),
        axis=0,
    ).reshape(-1)
    combined_active = jnp.concatenate(
        (host_active[None, :], local_active), axis=0
    ).reshape(-1)
    n_intervals_raw = jnp.sum(combined_active, dtype=jnp.int32)
    indices = jnp.nonzero(
        combined_active, size=capacity, fill_value=combined_active.size - 1
    )[0]
    n_intervals = jnp.minimum(n_intervals_raw, capacity)
    packed_active = jnp.arange(capacity) < n_intervals
    intervals = jnp.where(packed_active[:, None], combined_intervals[indices], 0.0)
    centers = jnp.where(packed_active, combined_centers[indices], 0.0 + 0.0j)
    owners = jnp.where(packed_active, combined_owners[indices], -1)
    interval_parameters = jnp.stack((centers, owners.astype(image_limb.dtype)), axis=1)
    local_status = jnp.bitwise_or.reduce(
        jnp.where(chart_active, local_topologies.status, jnp.int32(0))
    )
    status = jnp.bitwise_or(host_topology.status, local_status)
    status = jnp.bitwise_or(
        status,
        jnp.where(
            n_intervals_raw <= capacity,
            jnp.int32(RADIAL_OK),
            jnp.int32(RADIAL_CAPACITY),
        ),
    )
    local_topology = RadialTopology(
        intervals,
        n_intervals,
        status,
        host_topology.n_candidates_raw
        + jnp.sum(jnp.where(chart_active, local_topologies.n_candidates_raw, 0)),
        n_intervals_raw,
    )
    global_topology = define_radial_topology(
        image_limb,
        valid,
        rho,
        margin_r=margin_r,
        origin_inside=origin_inside,
        track_roots=False,
        lens_margin_parameters=lens_margin_parameters,
    )
    topology = jax.tree_util.tree_map(
        lambda local, global_: jnp.where(use_charts, local, global_),
        local_topology,
        global_topology,
    )
    global_parameters = jnp.stack(
        (
            jnp.zeros(capacity, dtype=image_limb.dtype),
            jnp.full(capacity, -1, dtype=image_limb.real.dtype).astype(
                image_limb.dtype
            ),
        ),
        axis=1,
    )
    interval_parameters = jnp.where(use_charts, interval_parameters, global_parameters)
    return ChartedTopology(
        topology, interval_parameters, chart_centers, chart_radii, chart_active
    )


def _owned_angular_intervals(
    intervals, n_intervals, radius, interval_parameter, charts: ChartedTopology
):
    """Keep angular arcs owned by the global chart or one local chart."""

    center = interval_parameter[0]
    owner = jnp.asarray(jnp.real(interval_parameter[1]), dtype=jnp.int32)
    active = jnp.arange(intervals.shape[0]) < n_intervals
    midpoint = 0.5 * (intervals[:, 0] + intervals[:, 1])
    points = center + radius * jnp.exp(1j * midpoint)
    in_chart = (
        jnp.abs(points[:, None] - charts.chart_centers[None, :])
        <= charts.chart_radii[None, :]
    ) & charts.chart_active[None, :]
    safe_owner = jnp.maximum(owner, 0)
    local_owned = in_chart[:, safe_owner]
    global_owned = ~jnp.any(in_chart, axis=1)
    owned = active & jnp.where(owner >= 0, local_owned, global_owned)
    indices = jnp.nonzero(
        owned, size=intervals.shape[0], fill_value=intervals.shape[0] - 1
    )[0]
    count = jnp.sum(owned, dtype=jnp.int32)
    packed_active = jnp.arange(intervals.shape[0]) < count
    packed = jnp.where(packed_active[:, None], intervals[indices], 0.0)
    return packed, count


def _planetary_mixed_topology(
    image_limb: Array,
    mask_limb: Array,
    rho: float,
    *,
    margin_r: float,
    lens: BinaryGeometry,
    w_center_shifted: Array,
    origin_inside: Array,
    jacobian_radial_margin: bool,
    interval_capacity: int = RADIAL_INTERVAL_CAPACITY,
) -> tuple[RadialTopology, Array]:
    """Move a certified, radially separated planetary group to local charts.

    The host Einstein-ring branches retain the centre-of-mass polar origin.
    A compact group close to the low-mass lens is translated only when its
    global radial projection is disjoint from every retained host branch.  The
    returned fixed-capacity topology packs host and local intervals together;
    ``interval_centers`` lets one radial scheduler evaluate both coordinate
    charts without a global/local device conditional.
    """

    image_limb, mask_limb = track_limb_images(image_limb, mask_limb)
    finite = jnp.isfinite(image_limb.real) & jnp.isfinite(image_limb.imag)
    valid = mask_limb & finite
    branch_active = jnp.any(valid, axis=1)
    branch_count = jnp.maximum(jnp.sum(valid, axis=1), 1)
    branch_centers = (
        jnp.sum(jnp.where(valid, image_limb, 0.0 + 0.0j), axis=1) / branch_count
    )
    base_margin = jnp.asarray(margin_r, dtype=image_limb.real.dtype) * jnp.asarray(
        rho, dtype=image_limb.real.dtype
    )
    branch_radii = (
        jnp.max(
            jnp.where(
                valid,
                jnp.abs(image_limb - branch_centers[:, None]),
                0.0,
            ),
            axis=1,
        )
        + base_margin
    )

    planet_position = jnp.asarray(lens.shifted + lens.a, dtype=image_limb.dtype)
    planet_scale = jnp.sqrt(jnp.maximum(jnp.asarray(lens.e1), 0.0))
    planet_zone = PLANET_CHART_ZONE_RADII * (
        planet_scale + jnp.asarray(rho, dtype=image_limb.real.dtype)
    )
    planet_branch = branch_active & (
        jnp.abs(branch_centers - planet_position) <= planet_zone
    )

    # Use one polar chart for the complete planetary image group.  Splitting
    # root slots into several nearby charts is not a partition of the image
    # preimage: around a fold, different source-limb branches can bound the
    # same connected image region, and each chart can then integrate that area
    # independently.  One translated origin preserves the small angular scale
    # while making double counting structurally impossible.
    branch_capacity = image_limb.shape[0]
    slots = jnp.arange(branch_capacity, dtype=jnp.int32)
    sentinel = jnp.int32(branch_capacity)
    labels = jnp.where(planet_branch, jnp.int32(0), sentinel)
    transient = planet_branch & ~jnp.all(valid, axis=1)
    same_validity = jnp.all(valid[:, None, :] == valid[None, :, :], axis=2)
    adjacency = (
        (
            jnp.abs(branch_centers[:, None] - branch_centers[None, :])
            <= branch_radii[:, None] + branch_radii[None, :]
        )
        & planet_branch[:, None]
        & planet_branch[None, :]
    )
    adjacency = adjacency | (same_validity & transient[:, None] & transient[None, :])

    def propagate(current, connected):
        neighbours = jnp.where(connected, current[None, :], sentinel)
        return jnp.where(planet_branch, jnp.min(neighbours, axis=1), sentinel)

    for _ in range(branch_capacity):
        labels = propagate(labels, adjacency)

    def chart_geometry(current_labels):
        membership = (current_labels[:, None] == slots[None, :]) & planet_branch[
            :, None
        ]
        point_membership = membership.T[:, :, None] & valid[None, :, :]
        point_count = jnp.sum(point_membership, axis=(1, 2))
        centers = jnp.sum(
            jnp.where(
                point_membership,
                image_limb[None, :, :],
                0.0 + 0.0j,
            ),
            axis=(1, 2),
        ) / jnp.maximum(point_count, 1)
        radii = (
            jnp.max(
                jnp.where(
                    point_membership,
                    jnp.abs(image_limb[None, :, :] - centers[:, None, None]),
                    0.0,
                ),
                axis=(1, 2),
            )
            + base_margin
        )
        return centers, radii, point_count > 0

    for _ in range(branch_capacity):
        chart_centers, chart_radii, chart_active = chart_geometry(labels)
        chart_overlap = (
            (
                jnp.abs(chart_centers[:, None] - chart_centers[None, :])
                <= chart_radii[:, None] + chart_radii[None, :]
            )
            & chart_active[:, None]
            & chart_active[None, :]
        )
        safe_labels = jnp.minimum(labels, branch_capacity - 1)
        reachable = chart_overlap[safe_labels]
        labels = jnp.where(
            planet_branch,
            jnp.min(jnp.where(reachable, slots[None, :], sentinel), axis=1),
            sentinel,
        )

    chart_centers, chart_radii, chart_active = chart_geometry(labels)
    global_radii = jnp.abs(image_limb)
    branch_lower = jnp.maximum(
        jnp.min(jnp.where(valid, global_radii, jnp.inf), axis=1) - base_margin,
        0.0,
    )
    branch_upper = (
        jnp.max(jnp.where(valid, global_radii, -jnp.inf), axis=1) + base_margin
    )
    host_branch = branch_active & ~planet_branch
    planet_lower = jnp.min(jnp.where(planet_branch, branch_lower, jnp.inf))
    planet_upper = jnp.max(jnp.where(planet_branch, branch_upper, -jnp.inf))
    host_lower = jnp.min(jnp.where(host_branch, branch_lower, jnp.inf))
    host_upper = jnp.max(jnp.where(host_branch, branch_upper, -jnp.inf))
    radial_support_disjoint = (planet_upper < host_lower) | (planet_lower > host_upper)
    # Radial separation about the global origin is not sufficient to prove
    # that a translated planetary chart contains only planetary images.  A
    # local disk can overlap a host-image disk even when their *global radial*
    # projections are disjoint; integrating the source level set in both
    # charts then counts that host area twice.  The limb-derived disks are
    # already available, so make spatial ownership an explicit, fixed-shape
    # prerequisite.  This is a one-pass geometry test, not a rescue solve.
    overlaps_host = (
        (
            jnp.abs(chart_centers[:, None] - branch_centers[None, :])
            <= chart_radii[:, None] + branch_radii[None, :]
        )
        & chart_active[:, None]
        & host_branch[None, :]
    )
    spatial_support_disjoint = ~jnp.any(overlaps_host)
    chart_level = jax.vmap(
        lambda center: binary_level_set(
            center,
            jnp.asarray(w_center_shifted, dtype=image_limb.dtype),
            jnp.asarray(rho, dtype=global_radii.dtype),
            jnp.asarray(lens.shifted, dtype=global_radii.dtype),
            a=jnp.asarray(lens.a, dtype=global_radii.dtype),
            e1=jnp.asarray(lens.e1, dtype=global_radii.dtype),
        )
    )(chart_centers)
    global_angle_scale = chart_radii / jnp.maximum(
        jnp.abs(chart_centers), jnp.finfo(global_radii.dtype).tiny
    )
    # A polar chart origin need not lie inside an image component.  The radial
    # topology already represents an exterior component as an annulus and is
    # given ``origin_inside=False`` below.  Requiring an interior centroid
    # unnecessarily rejects the strongly anisotropic non-resonant planetary
    # groups for which translation is most useful.
    chart_geometry_valid = jnp.all(
        ~chart_active | (global_angle_scale <= PLANET_CHART_MAX_GLOBAL_ANGLE)
    )
    use_local = (
        (jnp.asarray(lens.e1) <= 0.5)
        & jnp.any(planet_branch)
        & jnp.any(host_branch)
        & radial_support_disjoint
        & spatial_support_disjoint
        & chart_geometry_valid
    )
    use_local = jax.lax.stop_gradient(use_local)
    chart_centers = jax.lax.stop_gradient(
        jnp.where(use_local & chart_active, chart_centers, 0.0 + 0.0j)
    )
    local_branch = planet_branch & use_local
    host_mask = valid & ~local_branch[:, None]
    chart_masks = (
        (labels[:, None, None] == slots[None, :, None])
        & local_branch[:, None, None]
        & valid[:, None, :]
    ).transpose(1, 0, 2)
    chart_active = chart_active & use_local
    margin_parameters = (
        (lens.shifted, lens.a, lens.e1) if jacobian_radial_margin else None
    )
    host_topology = define_radial_topology(
        image_limb,
        host_mask,
        rho,
        margin_r=margin_r,
        origin_inside=origin_inside,
        track_roots=False,
        binary_margin_parameters=margin_parameters,
        interval_capacity=interval_capacity,
    )
    local_origin_inside = chart_active & (chart_level <= 0.0)

    def local_topology(mask, center, center_inside):
        return define_radial_topology(
            image_limb,
            mask,
            rho,
            margin_r=margin_r,
            origin_inside=center_inside,
            track_roots=False,
            binary_margin_parameters=margin_parameters,
            radial_origin=center,
            # A compact low-q image can be nearly circular about its local
            # chart centre. Root-solve roundoff then produces many tiny radial
            # zig-zags that are not physical topology changes. Global extrema
            # and mask endpoints remain present after this existing x64 filter.
            filter_roundoff_turning_points=True,
            interval_capacity=interval_capacity,
        )

    local_topologies = jax.vmap(local_topology)(
        chart_masks, chart_centers, local_origin_inside
    )
    capacity = host_topology.intervals.shape[0]
    host_active = jnp.arange(capacity) < host_topology.n_intervals
    local_active = (
        jnp.arange(capacity)[None, :] < local_topologies.n_intervals[:, None]
    ) & chart_active[:, None]
    combined_intervals = jnp.concatenate(
        (
            host_topology.intervals[None, :, :],
            local_topologies.intervals,
        ),
        axis=0,
    ).reshape(-1, 2)
    combined_centers = jnp.concatenate(
        (
            jnp.zeros(1, dtype=image_limb.dtype),
            chart_centers,
        )
    )[:, None]
    combined_centers = jnp.broadcast_to(
        combined_centers, (branch_capacity + 1, capacity)
    ).reshape(-1)
    combined_active = jnp.concatenate(
        (host_active[None, :], local_active), axis=0
    ).reshape(-1)
    n_intervals_raw = jnp.sum(combined_active, dtype=jnp.int32)
    indices = jnp.nonzero(
        combined_active,
        size=capacity,
        fill_value=combined_active.size - 1,
    )[0]
    n_intervals = jnp.minimum(n_intervals_raw, capacity)
    packed_active = jnp.arange(capacity) < n_intervals
    intervals = jnp.where(packed_active[:, None], combined_intervals[indices], 0.0)
    interval_centers = jnp.where(packed_active, combined_centers[indices], 0.0 + 0.0j)
    local_status = jnp.bitwise_or.reduce(
        jnp.where(chart_active, local_topologies.status, jnp.int32(0))
    )
    status = jnp.bitwise_or(host_topology.status, local_status)
    status = jnp.bitwise_or(
        status,
        jnp.where(
            n_intervals_raw <= capacity,
            jnp.int32(RADIAL_OK),
            jnp.int32(RADIAL_CAPACITY),
        ),
    )
    topology = RadialTopology(
        intervals,
        n_intervals,
        status,
        host_topology.n_candidates_raw
        + jnp.sum(jnp.where(chart_active, local_topologies.n_candidates_raw, 0)),
        n_intervals_raw,
    )
    return topology, interval_centers
