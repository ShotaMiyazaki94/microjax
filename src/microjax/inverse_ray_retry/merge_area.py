"""Region construction for inverse-ray polar integration in image space.

This module identifies and refines radial and angular subregions where images
of the source limb appear, to focus sampling near caustics and reduce wasted
integration. It clusters the mapped limb points and expands ranges by margins.
"""

import jax.numpy as jnp
from microjax.point_source import lens_eq, _images_point_source
from microjax.lens_geometry import triple_lens_geometry
from typing import Tuple

Array = jnp.ndarray


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
    eligible = jnp.abs(residual) < jnp.asarray(
        1.0e-4, dtype=image.real.dtype
    )
    determinant_floor = jnp.sqrt(jnp.finfo(image.real.dtype).eps)
    polished = image
    for _ in range(2):
        residual = (
            lens_eq(polished, nlenses=2, a=a, e1=e1)
            - w_limb_shift[None, :]
        )
        shear = e1 / (jnp.conjugate(polished) - a) ** 2 + (
            1.0 - e1
        ) / (jnp.conjugate(polished) + a) ** 2
        determinant = 1.0 - jnp.abs(shear) ** 2
        step = (
            -residual + shear * jnp.conjugate(residual)
        ) / jnp.where(
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
            lens_eq(candidate, nlenses=2, a=a, e1=e1)
            - w_limb_shift[None, :]
        )
        improves = (
            jnp.isfinite(candidate.real)
            & jnp.isfinite(candidate.imag)
            & (jnp.abs(candidate_residual) < jnp.abs(residual))
        )
        polished = jnp.where(improves, candidate, polished)

    final_residual = jnp.abs(
        lens_eq(polished, nlenses=2, a=a, e1=e1)
        - w_limb_shift[None, :]
    )
    polished_mask = jnp.isfinite(final_residual) & (final_residual < 1.0e-6)
    return (
        jnp.where(needs_repair, polished, image),
        jnp.where(needs_repair, polished_mask, mask),
    )

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
        rho
        * jnp.exp(
            1.0j * jnp.linspace(0.0, 2 * jnp.pi, Nlimb)
        ),
        dtype=jnp.asarray(w_center).dtype,
    )
    if nlenses == 2:
        s, q = _params["s"], _params["q"]
        a = 0.5 * s
        e1 = q / (1.0 + q)
        shifted = a * (1.0 - q) / (1.0 + q)
        _params = {"q": q, "s": s, "a": a, "e1": e1}
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
    image, mask = _images_point_source(w_limb_shift, nlenses=nlenses, **_params)
    if nlenses == 2:
        image, mask = _polish_binary_limb_images(
            image, mask, w_limb_shift, _params["a"], _params["e1"]
        )
    image_limb = image + shifted
    return image_limb, mask
