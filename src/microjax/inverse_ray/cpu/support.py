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


def radial_support_from_limb(
    image_limb: Array,
    image_mask: Array,
) -> RadialSupport:
    """Project five tracked image limbs into at most nine disjoint cells."""

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
    "RadialSupport",
    "build_radial_support",
    "radial_support_from_limb",
    "tracked_limb_neighbors",
    "trace_binary_source_limb",
]
