"""Continuous projection cells for Cartesian inverse-ray integration.

Projection tangencies bound the image support. Critical/source-limb crossings
also partition its interior: those crossings can produce narrow features in
the strip profile without being projection extrema. Both are solved before
quadrature, using a fixed Newton schedule and the existing image trace.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .line_roots import binary_line_level_set_roots
from .support import tracked_limb_neighbors
from .support_events import _critical_event_image_seeds, _refine_support_events


def _observed_critical_events(
    image_limb, physical_mask, w_center, rho, s, q, *, neighbors
):
    """Refine twelve physical branch ends into critical/source-limb crossings."""

    _, _, previous_mask, following_mask = neighbors
    counts = jnp.sum(physical_mask, axis=0)
    previous_change = counts != jnp.roll(counts, 1)
    following_change = counts != jnp.roll(counts, -1)
    endpoint = physical_mask & (
        (~previous_mask & previous_change[None, :])
        | (~following_mask & following_change[None, :])
    )
    index = jnp.nonzero(endpoint.reshape(-1), size=12, fill_value=-1)[0]
    seeds = image_limb.reshape(-1)[jnp.maximum(index, 0)]
    roots, valid, residual = _refine_support_events(
        seeds, w_center, rho, s, q, critical=True
    )
    return roots, valid & (index >= 0), residual


def _cartesian_projection_events(
    image_limb, physical_mask, w_center, rho, s, q, axis, *, neighbors
):
    """Resolve projection and critical events with one common seed schedule."""

    coordinate = jnp.real(image_limb * jnp.conj(axis))
    lo = jnp.argmin(jnp.where(physical_mask, coordinate, jnp.inf), axis=1)
    hi = jnp.argmax(jnp.where(physical_mask, coordinate, -jnp.inf), axis=1)
    rows = jnp.arange(image_limb.shape[0])
    branch = jnp.concatenate([image_limb[rows, lo], image_limb[rows, hi]])
    branch_valid = jnp.concatenate([jnp.any(physical_mask, axis=1)] * 2)
    phases = jnp.arange(4) * image_limb.shape[1] // 4
    scouts = image_limb[:, phases].reshape(-1)
    # Rejected algebraic roots can still seed a physical projection event.
    scout_valid = jnp.ones(scouts.shape, dtype=bool)
    probe = _critical_event_image_seeds(
        image_limb, physical_mask, w_center, rho, s, q
    )
    seeds = jnp.concatenate([scouts, branch, probe.images])
    enabled = jnp.concatenate([scout_valid, branch_valid, probe.physical_mask])
    roots, valid, _ = _refine_support_events(
        seeds, w_center, rho, s, q, critical=False, axis=axis
    )
    observed, observed_valid, _ = _observed_critical_events(
        image_limb, physical_mask, w_center, rho, s, q, neighbors=neighbors
    )
    return (
        jnp.concatenate([roots, probe.critical_images, observed]),
        jnp.concatenate([valid & enabled, probe.critical_valid, observed_valid]),
    )


def _cartesian_support_cells_from_events(
    image_limb,
    physical_mask,
    *,
    w_center,
    rho,
    s,
    q,
    axis=1 + 0j,
    topology_uncertain,
    minimum_ghost_residual,
    limb_topology,
    maximum_extrema=20,
    neighbors=None,
    **unused,
):
    """Sort physical events and integrate only cells with image evidence.

    The fixed output has ``maximum_extrema + 1`` cells. Finite Newton seeds
    do not certify event completeness. A physical trace point therefore
    proves occupancy independently of the midpoint classifier; it must not
    be discarded when an unobserved projection turn lies inside a cell.
    """

    if neighbors is None:
        neighbors = tracked_limb_neighbors(image_limb, physical_mask)
    roots, valid = _cartesian_projection_events(
        image_limb, physical_mask, w_center, rho, s, q, axis,
        neighbors=neighbors,
    )
    coordinate = jnp.real(roots * jnp.conj(axis))
    order = jnp.argsort(jnp.where(valid, coordinate, jnp.inf))
    values = jnp.where(valid[order], coordinate[order], jnp.inf)
    scale = jnp.abs(roots[order]) + jnp.abs(s) + rho
    tolerance = 8 * jnp.finfo(values.dtype).eps * (scale[1:] + scale[:-1])
    duplicate = jnp.concatenate([
        jnp.asarray([False]), jnp.diff(values) <= tolerance
    ])
    unique = jnp.isfinite(values) & ~duplicate
    index = jnp.nonzero(unique, size=maximum_extrema + 2, fill_value=-1)[0]
    endpoints = jnp.where(index >= 0, values[jnp.maximum(index, 0)], jnp.inf)
    cells = jnp.stack([endpoints[:-1], endpoints[1:]], axis=-1)
    active = jnp.isfinite(cells[:, 1]) & (cells[:, 1] > cells[:, 0])

    projected = jnp.real(image_limb * jnp.conj(axis)).reshape(-1)
    witnessed = active & jnp.any(
        physical_mask.reshape(-1)[None, :]
        & (projected[None, :] > cells[:, 0, None])
        & (projected[None, :] < cells[:, 1, None]), axis=1
    )
    unproven = active & ~witnessed
    indices = jnp.nonzero(unproven, size=active.size, fill_value=0)[0]

    def classify(i, mask):
        cell = indices[i]
        midpoint = cells[cell, 0] + .5 * (cells[cell, 1] - cells[cell, 0])
        roots = binary_line_level_set_roots(
            midpoint * axis, 1j * axis, w_center, rho, s=s, q=q
        )
        real = (roots.imag == 0) & jnp.isfinite(roots.real)
        return mask.at[cell].set(jnp.sum(real) >= 2)

    occupied = jax.lax.fori_loop(
        0, jnp.sum(unproven, dtype=jnp.int32), classify, witnessed
    )
    # Capacity loss is structural uncertainty, even when the returned cells
    # happen to produce a finite integral. Preserve it through the existing
    # support/status contract without changing the selected integration route.
    truncated = jnp.sum(unique, dtype=jnp.int32) > maximum_extrema + 2
    return (cells, active & occupied, jnp.asarray(topology_uncertain) | truncated,
            minimum_ghost_residual, limb_topology)
