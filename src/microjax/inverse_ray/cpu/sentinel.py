"""Four-reference-point sentinel for caustics hidden inside a source disk."""

from __future__ import annotations

import jax.numpy as jnp

from microjax.coeffs import _poly_coeffs_critical_binary
from microjax.point_source import lens_eq
from microjax.poly_solver import poly_roots

from ..geometry.lens import binary_geometry

Array = jnp.ndarray


def binary_critical_caustic_reference_points(
    *,
    s: float | Array,
    q: float | Array,
) -> tuple[Array, Array]:
    """Return four critical images and their mapped caustic references."""

    lens = binary_geometry(s, q)
    real_dtype = jnp.asarray(lens.s).dtype
    phase = jnp.asarray(jnp.pi, dtype=real_dtype)
    coefficients = _poly_coeffs_critical_binary(phase, lens.a, lens.e1)
    critical = poly_roots(coefficients[None, :])[0]
    caustic = (
        lens_eq(
            critical,
            nlenses=2,
            a=lens.a,
            e1=lens.e1,
        )
        + lens.shifted
    )
    return critical + lens.shifted, caustic


def binary_caustic_reference_points(
    *,
    s: float | Array,
    q: float | Array,
) -> Array:
    """Map one critical-curve quartic to four caustic reference points."""

    _, caustic = binary_critical_caustic_reference_points(s=s, q=q)
    return caustic


def nearest_caustic_limb_reference(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
) -> tuple[Array, Array]:
    """Return the critical image nearest the source-limb caustic contact."""

    critical, caustic = binary_critical_caustic_reference_points(s=s, q=q)
    rho = jnp.asarray(rho)
    safe_rho = jnp.maximum(rho, jnp.finfo(rho.dtype).tiny)
    clearance = jnp.abs(jnp.abs(caustic - jnp.asarray(w_center)) - rho) / safe_rho
    selected = jnp.argmin(clearance)
    return critical[selected], clearance[selected]


def hidden_caustic_candidate(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    limb_transition: bool | Array,
) -> Array:
    """Flag a buried caustic not exposed by a 3-to-5 limb transition."""

    references = binary_caustic_reference_points(s=s, q=q)
    contains_reference = jnp.any(
        jnp.abs(references - jnp.asarray(w_center)) <= jnp.asarray(rho)
    )
    return contains_reference & ~jnp.asarray(limb_transition)


def caustic_reference_limb_clearance(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
) -> Array:
    """Return the nearest reference-point distance from the source limb.

    The value is dimensionless: zero places a caustic reference point on the
    circular source boundary, while values well above zero distinguish a
    caustic contained deeply inside the source from an unresolved near-contact.
    """

    references = binary_caustic_reference_points(s=s, q=q)
    rho = jnp.asarray(rho)
    safe_rho = jnp.maximum(
        rho,
        jnp.finfo(rho.dtype).tiny,
    )
    limb_distance = jnp.abs(
        jnp.abs(references - jnp.asarray(w_center)) - rho
    )
    return jnp.min(limb_distance / safe_rho)


__all__ = [
    "binary_caustic_reference_points",
    "binary_critical_caustic_reference_points",
    "caustic_reference_limb_clearance",
    "hidden_caustic_candidate",
    "nearest_caustic_limb_reference",
]
