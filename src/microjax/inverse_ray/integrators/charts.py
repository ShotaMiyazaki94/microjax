"""Selection and construction of image-local polar charts."""

import jax
import jax.numpy as jnp

from ..geometry.topology import (
    RADIAL_CAPACITY,
    RADIAL_OK,
    RadialTopology,
    define_radial_topology,
    track_limb_images,
)
from ..geometry.lens import BinaryGeometry
from ..roots.level_set import binary_level_set
from .common import Array

PLANET_CHART_MAX_GLOBAL_ANGLE = 2.0e-2
PLANET_CHART_ZONE_RADII = 8.0


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
    branch_centers = jnp.sum(jnp.where(valid, image_limb, 0.0 + 0.0j), axis=1) / branch_count
    base_margin = jnp.asarray(margin_r, dtype=image_limb.real.dtype) * jnp.asarray(rho, dtype=image_limb.real.dtype)
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
    planet_zone = PLANET_CHART_ZONE_RADII * (planet_scale + jnp.asarray(rho, dtype=image_limb.real.dtype))
    planet_branch = branch_active & (jnp.abs(branch_centers - planet_position) <= planet_zone)

    # Build up to five disjoint charts only within the planetary group.  Static
    # label propagation merges overlapping branch disks and transient fold
    # pairs without a data-dependent Python branch.
    branch_capacity = image_limb.shape[0]
    slots = jnp.arange(branch_capacity, dtype=jnp.int32)
    sentinel = jnp.int32(branch_capacity)
    labels = jnp.where(planet_branch, slots, sentinel)
    transient = planet_branch & ~jnp.all(valid, axis=1)
    same_validity = jnp.all(valid[:, None, :] == valid[None, :, :], axis=2)
    adjacency = (
        (jnp.abs(branch_centers[:, None] - branch_centers[None, :]) <= branch_radii[:, None] + branch_radii[None, :])
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
        membership = (current_labels[:, None] == slots[None, :]) & planet_branch[:, None]
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
            (jnp.abs(chart_centers[:, None] - chart_centers[None, :]) <= chart_radii[:, None] + chart_radii[None, :])
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
    branch_upper = jnp.max(jnp.where(valid, global_radii, -jnp.inf), axis=1) + base_margin
    host_branch = branch_active & ~planet_branch
    planet_lower = jnp.min(jnp.where(planet_branch, branch_lower, jnp.inf))
    planet_upper = jnp.max(jnp.where(planet_branch, branch_upper, -jnp.inf))
    host_lower = jnp.min(jnp.where(host_branch, branch_lower, jnp.inf))
    host_upper = jnp.max(jnp.where(host_branch, branch_upper, -jnp.inf))
    radial_support_disjoint = (planet_upper < host_lower) | (planet_lower > host_upper)
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
    global_angle_scale = chart_radii / jnp.maximum(jnp.abs(chart_centers), jnp.finfo(global_radii.dtype).tiny)
    chart_geometry_valid = jnp.all(
        ~chart_active | ((chart_level <= 0.0) & (global_angle_scale <= PLANET_CHART_MAX_GLOBAL_ANGLE))
    )
    use_local = (
        (jnp.asarray(lens.e1) <= 0.5)
        & jnp.any(planet_branch)
        & jnp.any(host_branch)
        & radial_support_disjoint
        & chart_geometry_valid
    )
    use_local = jax.lax.stop_gradient(use_local)
    chart_centers = jax.lax.stop_gradient(jnp.where(use_local & chart_active, chart_centers, 0.0 + 0.0j))
    local_branch = planet_branch & use_local
    host_mask = valid & ~local_branch[:, None]
    chart_masks = (
        (labels[:, None, None] == slots[None, :, None]) & local_branch[:, None, None] & valid[:, None, :]
    ).transpose(1, 0, 2)
    chart_active = chart_active & use_local
    margin_parameters = (lens.shifted, lens.a, lens.e1) if jacobian_radial_margin else None
    host_topology = define_radial_topology(
        image_limb,
        host_mask,
        rho,
        margin_r=margin_r,
        origin_inside=origin_inside,
        track_roots=False,
        binary_margin_parameters=margin_parameters,
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
        )

    local_topologies = jax.vmap(local_topology)(chart_masks, chart_centers, local_origin_inside)
    capacity = host_topology.intervals.shape[0]
    host_active = jnp.arange(capacity) < host_topology.n_intervals
    local_active = (jnp.arange(capacity)[None, :] < local_topologies.n_intervals[:, None]) & chart_active[:, None]
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
    combined_centers = jnp.broadcast_to(combined_centers, (branch_capacity + 1, capacity)).reshape(-1)
    combined_active = jnp.concatenate((host_active[None, :], local_active), axis=0).reshape(-1)
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
    local_status = jnp.bitwise_or.reduce(jnp.where(chart_active, local_topologies.status, jnp.int32(0)))
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
        host_topology.n_candidates_raw + jnp.sum(jnp.where(chart_active, local_topologies.n_candidates_raw, 0)),
        n_intervals_raw,
    )
    return topology, interval_centers
