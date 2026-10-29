"""Physical angular cells for fixed-budget inverse-ray integration.

The trace support already refines angular tangencies. Its primary cuts are
retained, and critical/source-limb crossings partition narrow interior
features. Four common polar charts limit broad cell spans. No rejected-root
angle is used as a quadrature cut.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .angular_limb_dark import _physical_ray_intervals
from .angular_moment import AngularSupport, _angular_support_cells_from_trace
from .projection_support import _observed_critical_events
from .support import tracked_limb_neighbors
from .support_events import _critical_event_image_seeds, _event_equations


@jax.custom_jvp
def _critical_event_positions(roots, valid, w_center, rho, s, q):
    """Keep fixed-Newton positions, differentiating their defining equations."""

    return roots


@_critical_event_positions.defjvp
def _critical_event_positions_jvp(primals, tangents):
    roots, valid, w_center, rho, s, q = primals
    _, _, dw, drho, ds, dq = tangents
    xy = jnp.stack([roots.real, roots.imag], axis=-1)
    # Rejected candidates never become cuts. Their failed Newton trajectories
    # may have undefined derivatives even when the final positions are finite.
    safe_xy = jnp.where(valid[:, None], xy, jnp.asarray([.417, .263]))

    def differentiate(position):
        def equations(position, w_center, rho, s, q):
            return _event_equations(
                position, w_center, rho, s, q, critical=True, axis=1+0j
            )

        matrix = jax.jacfwd(equations)(position, w_center, rho, s, q)
        _, rhs = jax.jvp(
            lambda w, rho, s, q: equations(position, w, rho, s, q),
            (w_center, rho, s, q), (dw, drho, ds, dq),
        )
        # For H=(source-limb level, det lens Jacobian)=0, differentiation
        # gives H_xy dz + H_parameters dp = 0. This retains the moving-cut
        # contribution without differentiating sixteen auxiliary Newton steps.
        determinant = matrix[0, 0]*matrix[1, 1] - matrix[1, 0]*matrix[0, 1]
        dx = (-matrix[1, 1]*rhs[0] + matrix[0, 1]*rhs[1]) / determinant
        dy = (matrix[1, 0]*rhs[0] - matrix[0, 0]*rhs[1]) / determinant
        return dx + 1j*dy

    derivative = jax.vmap(differentiate)(safe_xy)
    return roots, jnp.where(valid, derivative, 0j)


def _angular_support_cells_from_events(
    w_center,
    rho,
    *,
    s,
    q,
    physical_limb,
    physical_mask,
    topology_uncertain,
    minimum_ghost_residual,
    limb_topology,
    neighbors=None,
    edge_activity=None,
):
    """Build 49 fixed cells from the existing trace and physical events.

    Finite Newton seeds cannot certify event completeness. A physical trace
    point inside an occupied trace envelope therefore proves cell occupancy
    independently of the midpoint ray. Only cells lacking this witness need
    the extra ray solve. This is one common policy for every angular case.
    """

    if neighbors is None:
        neighbors = tracked_limb_neighbors(physical_limb, physical_mask)
    primary = _angular_support_cells_from_trace(
        w_center, rho, s=s, q=q,
        physical_limb=physical_limb, physical_mask=physical_mask,
        topology_uncertain=topology_uncertain,
        minimum_ghost_residual=minimum_ghost_residual,
        limb_topology=limb_topology, neighbors=neighbors,
        edge_activity=edge_activity,
        refine_all_candidates=True,
    )
    roots = jnp.exp(1j * jnp.concatenate([
        primary.cells[:, 0], primary.cells[-1:, 1]
    ]))
    valid = jnp.isfinite(roots)
    probe = _critical_event_image_seeds(
        physical_limb, physical_mask, w_center, rho, s, q
    )
    probe_positions = _critical_event_positions(
        probe.critical_images, probe.critical_valid, w_center, rho, s, q
    )
    roots = jnp.concatenate([roots, probe_positions])
    valid = jnp.concatenate([valid, probe.critical_valid])
    critical, critical_valid, _ = _observed_critical_events(
        physical_limb, physical_mask, w_center, rho, s, q,
        neighbors=neighbors,
    )
    critical = _critical_event_positions(
        critical, critical_valid, w_center, rho, s, q
    )
    roots = jnp.concatenate([roots, critical])
    valid = jnp.concatenate([valid, critical_valid])

    occupied = primary.active & (primary.cells[:, 1] > primary.cells[:, 0])
    roots = jnp.concatenate([
        roots, jnp.asarray([1j, -1+0j, -1j], dtype=roots.dtype)
    ])
    valid = jnp.concatenate([valid, jnp.ones(3, dtype=bool)])

    angles = jnp.mod(jnp.angle(roots), 2 * jnp.pi)
    order = jnp.argsort(jnp.where(valid, angles, jnp.inf))
    values = jnp.where(valid[order], angles[order], jnp.inf)
    scale = (jnp.abs(roots[order]) + jnp.abs(s) + rho) / jnp.maximum(
        jnp.abs(roots[order]), jnp.finfo(rho.dtype).tiny
    )
    tolerance = 8 * jnp.finfo(rho.dtype).eps * (scale[1:] + scale[:-1])
    duplicate = jnp.concatenate([
        jnp.asarray([False]), jnp.diff(values) <= tolerance
    ])
    unique = jnp.isfinite(values) & ~duplicate
    maximum_extrema = 48
    overflow = jnp.sum(unique) > maximum_extrema
    index = jnp.nonzero(unique, size=maximum_extrema, fill_value=-1)[0]
    endpoints = jnp.where(index >= 0, values[jnp.maximum(index, 0)], 2 * jnp.pi)
    endpoints = jnp.concatenate([
        jnp.zeros(1, dtype=rho.dtype), endpoints,
        jnp.full(1, 2 * jnp.pi, dtype=rho.dtype),
    ])
    cells = jnp.stack([endpoints[:-1], endpoints[1:]], axis=-1)
    midpoint = jnp.mean(cells, axis=-1)
    nonempty = cells[:, 1] > cells[:, 0]
    inherited = jnp.any(
        occupied[None, :]
        & (midpoint[:, None] >= primary.cells[None, :, 0])
        & (midpoint[:, None] <= primary.cells[None, :, 1]), axis=-1
    )
    trace_angle = jnp.mod(jnp.angle(physical_limb.reshape(-1)), 2*jnp.pi)
    contains_trace = jnp.any(
        physical_mask.reshape(-1)[None, :]
        & (trace_angle[None, :] >= cells[:, 0, None])
        & (trace_angle[None, :] <= cells[:, 1, None]), axis=-1
    )
    inherited = inherited & contains_trace
    needed = nonempty & ~inherited

    def classify(inputs):
        angle, enabled = inputs

        def evaluate(_):
            lower, upper, inside, invalid = _physical_ray_intervals(
                angle[None], w_center, rho, s=s, q=q
            )
            return jnp.any(inside & (upper > lower)), invalid[0]

        return jax.lax.cond(
            enabled, evaluate,
            lambda _: (jnp.asarray(False), jnp.int32(0)), None,
        )

    midpoint_inside, invalid = jax.lax.map(classify, (midpoint, needed))
    active = nonempty & (midpoint_inside | inherited)
    # This records known capacity/interval failures, not a completeness proof.
    tangencies_valid = jnp.full(
        (maximum_extrema,), ~overflow & jnp.all(invalid == 0)
    )
    return AngularSupport(
        cells, active, topology_uncertain, minimum_ghost_residual,
        limb_topology, tangencies_valid,
    )
