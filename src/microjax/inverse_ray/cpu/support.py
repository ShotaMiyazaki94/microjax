"""Tracked binary-lens image limbs shared by the two CPU charts."""

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
_BINARY_ROOT_PERMUTATIONS = np.asarray(tuple(permutations(range(5))), dtype=np.int32)


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


def trace_binary_source_limb(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    n_limb: int,
) -> tuple[Array, Array]:
    """Solve and polish a source limb from q-aware quintic coefficients.

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
    polished_images = jax.vmap(lambda images, source: _polish_binary_images(images, source, s, q))(
        images_com, source_limb
    )
    image_mask = jax.vmap(lambda images, source: _physical_image_mask(images, source, s, q))(
        polished_images, source_limb
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
    phases = 2.0 * jnp.pi * jnp.arange(n_limb, dtype=real_dtype) / n_limb
    image_limb, physical_mask = trace_binary_source_limb(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
    )
    previous_limb, following_limb, previous_mask, following_mask = tracked_limb_neighbors(image_limb, physical_mask)

    lens = binary_geometry(s, q)
    source_limb = w_center + jnp.asarray(rho, dtype=real_dtype) * jnp.exp(1.0j * phases)
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
    inverse_permutation = (
        jnp.zeros_like(closing_permutation).at[closing_permutation].set(jnp.arange(5, dtype=jnp.int32))
    )

    previous = jnp.roll(image_limb, 1, axis=1).at[:, 0].set(last[closing_permutation])
    following = jnp.roll(image_limb, -1, axis=1).at[:, -1].set(first[inverse_permutation])
    previous_mask = jnp.roll(image_mask, 1, axis=1).at[:, 0].set(image_mask[:, -1][closing_permutation])
    following_mask = jnp.roll(image_mask, -1, axis=1).at[:, -1].set(image_mask[:, 0][inverse_permutation])
    return previous, following, previous_mask, following_mask
