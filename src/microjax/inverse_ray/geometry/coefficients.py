"""Numerically stable binary-lens polynomial coefficients.

The coefficient factorization follows Equation 6 of Wang, Wang & Dong
(2025, ApJS 276, 40), expressed in microJAX's public binary centre-of-mass
frame. Terms whose true scale is the secondary mass fraction remain explicitly
proportional to that fraction instead of being formed by subtracting
order-unity quantities.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

Array = jnp.ndarray


class BinaryQuintic(NamedTuple):
    """A planet-centred quintic and its shift to the public COM frame."""

    coefficients: Array
    image_shift: Array


def binary_quintic_coefficients(
    w: complex | Array,
    *,
    s: float | Array,
    q: float | Array,
) -> BinaryQuintic:
    """Construct q-aware binary-lens quintic coefficients."""

    w = jnp.asarray(w)
    real_dtype = w.real.dtype
    s = jnp.asarray(s, dtype=real_dtype)
    q = jnp.asarray(q, dtype=real_dtype)

    primary_mass = 1.0 / (1.0 + q)
    secondary_mass = q * primary_mass
    image_shift = s * primary_mass
    source_planet = w - image_shift
    source_planet_conjugate = jnp.conjugate(source_planet)

    secondary_scale = secondary_mass * s
    source_primary_conjugate = jnp.conjugate(w) + secondary_scale
    factored_scale = (1.0 - s) * (1.0 + s) + s * source_primary_conjugate

    source_real = source_planet.real
    source_imag = source_planet.imag
    source_norm = jnp.real(source_planet * source_planet_conjugate)

    c5 = -source_planet_conjugate * source_primary_conjugate
    c4 = (
        source_norm - 1.0 - 2.0 * s * source_planet_conjugate
    ) * source_primary_conjugate + secondary_scale
    c3 = (2.0 * source_planet - s) * (
        factored_scale * source_primary_conjugate + 2.0j * source_imag - secondary_scale
    ) + 4.0j * source_imag * (secondary_scale - source_planet)
    c2 = (
        factored_scale.real
        * (
            factored_scale.real * source_primary_conjugate.real
            + 1.0j * (1.0 - s * source_real) * source_imag
        )
        + s * source_imag**2 * (2.0 + s * source_primary_conjugate)
        + secondary_scale
        * (2.0 * source_norm + s * (-source_real + 3.0j * source_imag) - 1.0)
    )
    c1 = secondary_scale * (
        (s + 2.0 * source_planet) * factored_scale
        + s * (2.0j * source_imag * s - secondary_mass)
    )
    c0 = secondary_scale**2 * source_planet

    coefficients = jnp.stack((c5, c4, c3, c2, c1, c0), axis=-1)
    return BinaryQuintic(coefficients, image_shift)


__all__ = ["BinaryQuintic", "binary_quintic_coefficients"]
