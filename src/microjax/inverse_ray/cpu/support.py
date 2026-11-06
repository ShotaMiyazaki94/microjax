"""Tracked binary/triple image limbs shared by the two CPU charts."""

from __future__ import annotations
from itertools import permutations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from microjax.point_source import lens_eq
from microjax.poly_solver import poly_roots
from .lens import binary_lens, lens_map, lens_shear, source_polynomial
from ..geometry.lens import binary_geometry

Array = jnp.ndarray
_BINARY_ROOT_PERMUTATIONS = np.asarray(tuple(permutations(range(5))), dtype=np.int32)


class LimbTrace(NamedTuple):
    """Canonical source-limb trace shared by CPU geometry consumers.

    The polynomial roots and their physical mask must be treated as one object:
    rebuilding either one downstream can make support endpoints refer to a
    different set of images than the root certificate.  The incoming and
    outgoing edge masks are deliberately stricter than ``connected`` and only
    permit edges whose two phases have the same number of physical images.  A
    change in image multiplicity is represented by two disconnected trace
    pieces.  Both directions are stored explicitly because the periodic closing
    edge can permute algebraic root slots.

    Coordinates and masks come from the same polished roots.
    """

    image_limb: Array
    physical_mask: Array
    previous_limb: Array
    following_limb: Array
    previous_mask: Array
    following_mask: Array
    image_counts: Array
    residual: Array
    incoming_edge_active: Array
    outgoing_edge_active: Array
    angular_step: Array
    radial_step: Array


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

    initial_residual = lens_eq(images_midpoint, nlenses=2, a=a, e1=secondary_mass) - source_midpoint
    eligible = (
        jnp.isfinite(initial_residual.real)
        & jnp.isfinite(initial_residual.imag)
        & (jnp.abs(initial_residual) < 1.0e-4 * residual_scale)
    )

    polished = images_midpoint
    for _ in range(2):
        residual = lens_eq(polished, nlenses=2, a=a, e1=secondary_mass) - source_midpoint
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
        candidate_residual = lens_eq(candidate, nlenses=2, a=a, e1=secondary_mass) - source_midpoint
        improves = (
            safe
            & jnp.isfinite(candidate.real)
            & jnp.isfinite(candidate.imag)
            & (jnp.abs(candidate_residual) < jnp.abs(residual))
        )
        polished = jnp.where(improves, candidate, polished)

    return polished + midpoint_to_com


def physical_image_mask(images, source, lens):
    """Keep residual-resolved triple images; retain the validated binary rule."""
    if lens.positions.shape[0] == 2:
        return _physical_image_mask(images, source, lens.s, lens.q)
    residual = jnp.abs(lens_map(images, lens) - source)
    return jnp.isfinite(residual) & (residual <= 1.0e-7 * jnp.maximum(1.0, jnp.abs(source)))


def polish_images(images, source, lens):
    """Polish near-physical roots without admitting distant algebraic ghosts."""
    if lens.positions.shape[0] == 2:
        return _polish_binary_images(images, source, lens.s, lens.q)
    residual_scale = jnp.maximum(1.0, jnp.abs(source))
    residual = lens_map(images, lens) - source
    eligible = jnp.isfinite(residual) & (jnp.abs(residual) <= 1.0e-6 * residual_scale)
    determinant_floor = jnp.sqrt(jnp.finfo(images.real.dtype).eps)
    for _ in range(2):
        residual = lens_map(images, lens) - source
        shear = lens_shear(images, lens)
        shear_abs = jnp.abs(shear)
        determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
        safe = eligible & (jnp.abs(determinant) > determinant_floor)
        step = (-residual + shear * jnp.conj(residual)) / jnp.where(safe, determinant, 1.0)
        candidate = images + jnp.where(safe, step, 0.0j)
        improves = safe & jnp.isfinite(candidate) & (jnp.abs(lens_map(candidate, lens) - source) < jnp.abs(residual))
        images = jnp.where(improves, candidate, images)
    return images


def trace_source_limb(
    w_center: complex | Array,
    rho: float | Array,
    *,
    lens,
    n_limb: int,
) -> tuple[Array, Array]:
    """Continue and polish the five or ten roots around one source limb.

    The returned image coordinates are the guarded Newton-polished roots used
    to construct the returned physical mask.  Keeping one coordinate set for
    both purposes is essential: all later support extrema and branch
    continuation must see the same roots that passed the lens-equation
    residual test.

    The source is sampled at uniform angular phases.
    """

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    phases = 2.0 * jnp.pi * jnp.arange(n_limb, dtype=real_dtype) / n_limb
    source_limb = w_center + jnp.asarray(rho, dtype=real_dtype) * jnp.exp(1.0j * phases)
    coefficients, image_shift = source_polynomial(source_limb, lens)

    first_roots = poly_roots(coefficients[:1])[0]

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
        coefficients[1:],
    )
    roots = jnp.concatenate((first_roots[None, :], remaining_roots), axis=0)
    images_com = roots + image_shift
    polished_images = jax.vmap(lambda images, source: polish_images(images, source, lens))(images_com, source_limb)
    image_mask = jax.vmap(lambda images, source: physical_image_mask(images, source, lens))(
        polished_images, source_limb
    )
    # The mask is evaluated on the polished roots.  Return those same roots
    # to every downstream consumer (support extrema, branch neighbours, and
    # strip/radial seeds); returning ``images_com`` here would make the trace
    # geometry and its physical-root certificate refer to different points.
    return jnp.moveaxis(polished_images, 0, 1), jnp.moveaxis(image_mask, 0, 1)


def trace_source_limb_record(
    w_center: complex | Array,
    rho: float | Array,
    *,
    lens,
    n_limb: int,
) -> LimbTrace:
    """Attach residuals and same-multiplicity edges to one polished trace."""

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    phases = 2.0 * jnp.pi * jnp.arange(n_limb, dtype=real_dtype) / n_limb
    image_limb, physical_mask = trace_source_limb(
        w_center,
        rho,
        lens=lens,
        n_limb=n_limb,
    )
    previous_limb, following_limb, previous_mask, following_mask = tracked_limb_neighbors(image_limb, physical_mask)

    source_limb = w_center + jnp.asarray(rho, dtype=real_dtype) * jnp.exp(1.0j * phases)
    if lens.positions.shape[0] == 2:
        binary = binary_geometry(lens.s, lens.q)
        residual = jnp.abs(
            lens_eq(image_limb - binary.shifted, nlenses=2, a=binary.a, e1=binary.e1)
            - (source_limb[None, :] - binary.shifted)
        )
    else:
        residual = jnp.abs(lens_map(image_limb, lens) - source_limb[None, :])
    image_counts = jnp.sum(physical_mask, axis=0, dtype=jnp.int32)
    connected = physical_mask & following_mask
    incoming_connected = physical_mask & previous_mask
    same_previous_multiplicity = image_counts == jnp.roll(image_counts, 1)
    same_following_multiplicity = image_counts == jnp.roll(image_counts, -1)
    incoming_edge_active = incoming_connected & same_previous_multiplicity[None, :]
    outgoing_edge_active = connected & same_following_multiplicity[None, :]
    radius = jnp.abs(image_limb)
    following_radius = jnp.abs(following_limb)
    angular_step = jnp.abs(jnp.angle(following_limb * jnp.conjugate(image_limb)))
    radial_step = jnp.abs(following_radius - radius)
    return LimbTrace(
        image_limb,
        physical_mask,
        previous_limb,
        following_limb,
        previous_mask,
        following_mask,
        image_counts,
        residual,
        incoming_edge_active,
        outgoing_edge_active,
        angular_step,
        radial_step,
    )


def _minimum_cost_assignment(cost):
    """Exactly assign ten roots using 2**10 subsets, not 10! permutations.

    Each row extends the optimum for subsets one element smaller. Only ten
    vectorized updates and one ten-step backtrack are needed for a triple
    trace. Assignment is discrete and does not contribute an AD tangent.
    """
    cost = jax.lax.stop_gradient(cost)
    n = cost.shape[0]
    cost = jnp.where(jnp.isfinite(cost), cost, jnp.finfo(cost.dtype).max / (4 * n))
    subsets = jnp.arange(1 << n, dtype=jnp.int32)
    bits = jnp.left_shift(jnp.int32(1), jnp.arange(n, dtype=jnp.int32))
    occupied = (subsets[:, None] & bits[None, :]) != 0
    parents = subsets[:, None] ^ bits[None, :]
    size = jax.lax.population_count(subsets)
    initial = jnp.full(subsets.shape, jnp.inf, dtype=cost.dtype).at[0].set(0.0)

    def extend(row, state):
        optimum, choices = state
        candidates = jnp.where(occupied, optimum[parents] + cost[row, :], jnp.inf)
        selected = jnp.argmin(candidates, axis=1).astype(jnp.int32)
        values = jnp.take_along_axis(candidates, selected[:, None], axis=1)[:, 0]
        active = size == row + 1
        return jnp.where(active, values, optimum), jnp.where(active, selected, choices)

    _, choices = jax.lax.fori_loop(0, n, extend, (initial, jnp.zeros_like(subsets)))

    def backtrack(index, state):
        subset, permutation = state
        selected = choices[subset]
        return subset ^ bits[selected], permutation.at[n - index - 1].set(selected)

    _, permutation = jax.lax.fori_loop(0, n, backtrack, (jnp.int32((1 << n) - 1), jnp.zeros((n,), dtype=jnp.int32)))
    return permutation


def tracked_limb_neighbors(
    image_limb: Array,
    image_mask: Array,
) -> tuple[Array, Array, Array, Array]:
    """Return periodic neighbors while preserving tracked root branches.

    Sequential polynomial solves keep each root in its previous slot locally,
    but analytic continuation around a source loop can permute those slots.
    The last-to-first edge therefore needs an explicit root assignment;
    a plain ``roll`` can connect unrelated images exactly where a caustic is
    enclosed. The 120 possible binary-root permutations are small enough to
    score exhaustively once per trace. Ten-root traces use an exact subset
    assignment with the same squared-distance objective.
    """

    first = image_limb[:, 0]
    last = image_limb[:, -1]
    if image_limb.shape[0] == 5:
        root_permutations = jnp.asarray(_BINARY_ROOT_PERMUTATIONS)
        assignment_cost = jnp.sum(jnp.abs(last[root_permutations] - first[None, :]) ** 2, axis=1)
        selected = jnp.argmin(jax.lax.stop_gradient(assignment_cost))
        closing_permutation = root_permutations[selected]
    else:
        closing_permutation = _minimum_cost_assignment(jnp.abs(first[:, None] - last[None, :]) ** 2)
    inverse_permutation = (
        jnp.zeros_like(closing_permutation)
        .at[closing_permutation]
        .set(jnp.arange(image_limb.shape[0], dtype=jnp.int32))
    )

    previous = jnp.roll(image_limb, 1, axis=1).at[:, 0].set(last[closing_permutation])
    following = jnp.roll(image_limb, -1, axis=1).at[:, -1].set(first[inverse_permutation])
    previous_mask = jnp.roll(image_mask, 1, axis=1).at[:, 0].set(image_mask[:, -1][closing_permutation])
    following_mask = jnp.roll(image_mask, -1, axis=1).at[:, -1].set(image_mask[:, 0][inverse_permutation])
    return previous, following, previous_mask, following_mask


def trace_binary_source_limb(w_center, rho, *, s, q, n_limb):
    return trace_source_limb(w_center, rho, lens=binary_lens(s, q), n_limb=n_limb)


def trace_binary_source_limb_record(w_center, rho, *, s, q, n_limb):
    return trace_source_limb_record(w_center, rho, lens=binary_lens(s, q), n_limb=n_limb)
