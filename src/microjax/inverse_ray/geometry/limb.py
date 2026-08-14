"""Trace the circular source limb into fixed-shape image branches.

This module performs the point-source solves used to discover finite-source
image support.  It returns ``(image_slot, limb_phase)`` arrays; chart selection,
topology construction, and area integration belong to higher layers.
"""

import jax.numpy as jnp
from microjax.point_source import lens_eq, _images_point_source
from microjax.poly_solver import poly_roots
from microjax.lens_geometry import triple_lens_geometry
from .coefficients import binary_quintic_coefficients
from .lens import binary_geometry
from typing import Tuple

Array = jnp.ndarray
_TRIPLE_POLISH_STEPS = 2


def _polish_binary_limb_images(
    image: Array,
    mask: Array,
    w_limb_shift: Array,
    a: Array,
    e1: Array,
) -> Tuple[Array, Array]:
    """Polish near-physical binary roots before constructing limb topology.

    The degree-five polynomial can leave the small planetary image with a
    lens-equation residual just above the historical ``1e-6`` mask threshold.
    Alternating that true third image on and off creates hundreds of false
    radial phases.  Two fixed real-Newton steps use the exact binary lens
    Jacobian and accept only residual-decreasing updates.  Roots farther than
    ``1e-4`` from the lens equation are not moved, preventing the two algebraic
    non-images from being pulled into an unrelated physical solution.
    """

    needs_repair = jnp.any(jnp.sum(mask, axis=0) < 3)
    residual = lens_eq(image, nlenses=2, a=a, e1=e1) - w_limb_shift[None, :]
    eligible = jnp.abs(residual) < jnp.asarray(1.0e-4, dtype=image.real.dtype)
    determinant_floor = jnp.sqrt(jnp.finfo(image.real.dtype).eps)
    polished = image
    for _ in range(2):
        residual = lens_eq(polished, nlenses=2, a=a, e1=e1) - w_limb_shift[None, :]
        shear = (
            e1 / (jnp.conjugate(polished) - a) ** 2
            + (1.0 - e1) / (jnp.conjugate(polished) + a) ** 2
        )
        shear_abs = jnp.abs(shear)
        determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
        step = (-residual + shear * jnp.conjugate(residual)) / jnp.where(
            jnp.abs(determinant) > determinant_floor,
            determinant,
            1.0,
        )
        candidate = polished + jnp.where(
            eligible & (jnp.abs(determinant) > determinant_floor),
            step,
            0.0 + 0.0j,
        )
        candidate_residual = (
            lens_eq(candidate, nlenses=2, a=a, e1=e1) - w_limb_shift[None, :]
        )
        improves = (
            jnp.isfinite(candidate.real)
            & jnp.isfinite(candidate.imag)
            & (jnp.abs(candidate_residual) < jnp.abs(residual))
        )
        polished = jnp.where(improves, candidate, polished)

    final_residual = jnp.abs(
        lens_eq(polished, nlenses=2, a=a, e1=e1) - w_limb_shift[None, :]
    )
    polished_mask = jnp.isfinite(final_residual) & (final_residual < 1.0e-6)
    return (
        jnp.where(needs_repair, polished, image),
        jnp.where(needs_repair, polished_mask, mask),
    )


def _polish_triple_limb_images(
    image: Array,
    mask: Array,
    w_limb_shift: Array,
    a: Array,
    e1: Array,
    e2: Array,
    r3: Array,
    psi: Array,
) -> Tuple[Array, Array]:
    """Polish degree-ten roots with the exact triple-lens Jacobian.

    Only roots already classified as physical are refined.  A triple-lens
    polynomial also contains ghost roots; allowing those roots into the Newton
    basin can collapse several slots onto one physical image and corrupt limb
    topology.
    """

    lens_params = {"a": a, "e1": e1, "e2": e2, "r3": r3, "psi": psi}
    lens_positions = jnp.asarray([a, -a, r3 * jnp.exp(1.0j * psi)])
    lens_masses = jnp.asarray([e1, e2, 1.0 - e1 - e2])
    determinant_floor = jnp.sqrt(jnp.finfo(image.real.dtype).eps)
    polished = image

    for _ in range(_TRIPLE_POLISH_STEPS):
        residual = lens_eq(polished, nlenses=3, **lens_params) - w_limb_shift[None, :]
        shear = jnp.sum(
            lens_masses[:, None, None]
            / (
                jnp.conjugate(polished)[None, :, :]
                - jnp.conjugate(lens_positions)[:, None, None]
            )
            ** 2,
            axis=0,
        )
        shear_abs = jnp.abs(shear)
        determinant = (1.0 - shear_abs) * (1.0 + shear_abs)
        nonsingular = jnp.abs(determinant) > determinant_floor
        step = (-residual + shear * jnp.conjugate(residual)) / jnp.where(
            nonsingular, determinant, 1.0
        )
        candidate = polished + jnp.where(mask & nonsingular, step, 0.0 + 0.0j)
        candidate_residual = (
            lens_eq(candidate, nlenses=3, **lens_params) - w_limb_shift[None, :]
        )
        improves = (
            jnp.isfinite(candidate.real)
            & jnp.isfinite(candidate.imag)
            & (jnp.abs(candidate_residual) < jnp.abs(residual))
        )
        polished = jnp.where(improves, candidate, polished)

    final_residual = jnp.abs(
        lens_eq(polished, nlenses=3, **lens_params) - w_limb_shift[None, :]
    )
    polished_mask = mask & jnp.isfinite(final_residual) & (final_residual < 1.0e-6)
    return polished, polished_mask


def calc_source_limb(
    w_center: complex,
    rho: float,
    Nlimb: int = 100,
    nlenses: int = 2,
    **_params,
) -> Tuple[Array, Array]:
    """Map uniformly sampled source-limb points to image plane and mask real images.

    Returns
    - image_limb: (nimg, Nlimb) complex array of images shifted back to the lens frame.
    - mask: boolean array indicating which images are real at each limb angle.
    """
    w_limb = w_center + jnp.asarray(
        rho * jnp.exp(1.0j * jnp.linspace(0.0, 2 * jnp.pi, Nlimb)),
        dtype=jnp.asarray(w_center).dtype,
    )
    if nlenses == 2:
        s, q = _params["s"], _params["q"]
        geometry = binary_geometry(s, q)
        shifted = geometry.shifted
        _params = {"q": q, "s": s, "a": geometry.a, "e1": geometry.e1}
    elif nlenses == 3:
        s, q, q3, r3, psi = (
            _params["s"],
            _params["q"],
            _params["q3"],
            _params["r3"],
            _params["psi"],
        )
        geometry = triple_lens_geometry(s, q, q3, r3, psi)
        shifted = geometry.shifted
        _params = {
            "a": geometry.a,
            "r3": r3,
            "e1": geometry.e1,
            "e2": geometry.e2,
            "q": q,
            "s": s,
            "q3": q3,
            "psi": psi,
        }
    else:
        raise ValueError("Only 2 or 3 lenses are supported.")

    w_limb_shift = w_limb - shifted
    if nlenses == 2:
        # Construct the binary quintic at its physical low-q scale while
        # retaining one independent batched solve per limb point.  This is the
        # accelerator path: unlike the CPU continuation tracer, it exposes the
        # complete limb dimension to XLA/GPU parallelism.
        quintic = binary_quintic_coefficients(w_limb, s=s, q=q)
        image_com = jnp.moveaxis(poly_roots(quintic.coefficients), -1, 0)
        image_com = image_com + quintic.image_shift
        image = image_com - shifted
        residual = jnp.abs(
            lens_eq(image, nlenses=2, a=_params["a"], e1=_params["e1"])
            - w_limb_shift[None, :]
        )
        mask = jnp.isfinite(residual) & (residual < 1.0e-6)
        image, mask = _polish_binary_limb_images(
            image, mask, w_limb_shift, _params["a"], _params["e1"]
        )
    else:
        image, mask = _images_point_source(w_limb_shift, nlenses=nlenses, **_params)
        image, mask = _polish_triple_limb_images(
            image, mask, w_limb_shift, geometry.a, geometry.e1, geometry.e2, r3, psi
        )
    image_limb = image + shifted
    return image_limb, mask
