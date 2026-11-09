"""Lens geometry shared by the CPU image-plane integration kernels."""

from typing import NamedTuple

import jax.numpy as jnp

from microjax.lens_geometry import triple_lens_geometry
from ..geometry import binary_geometry
from ..coefficients import binary_quintic_coefficients


class CpuLens(NamedTuple):
    s: jnp.ndarray
    q: jnp.ndarray
    positions: jnp.ndarray
    masses: jnp.ndarray


def binary_lens(s, q):
    geometry = binary_geometry(s, q)
    positions = jnp.asarray([-geometry.a, geometry.a]) + geometry.shifted
    masses = jnp.asarray([1.0 - geometry.e1, geometry.e1])
    return CpuLens(jnp.asarray(s), jnp.asarray(q), positions, masses)


def triple_lens(s, q, q3, r3, psi):
    geometry = triple_lens_geometry(s, q, q3, r3, psi)
    positions = jnp.asarray([-geometry.a, geometry.a, geometry.r3_complex]) + geometry.shifted
    masses = jnp.asarray([geometry.e2, geometry.e1, geometry.e3])
    return CpuLens(jnp.asarray(s), jnp.asarray(q), positions, masses)


def lens_map(z, lens):
    """Map public-frame image coordinates into the public source plane."""
    difference = jnp.conj(jnp.asarray(z)[..., None] - lens.positions)
    return z - jnp.sum(lens.masses / difference, axis=-1)


def lens_shear(z, lens):
    difference = jnp.conj(jnp.asarray(z)[..., None] - lens.positions)
    return jnp.sum(lens.masses / difference**2, axis=-1)


def _polynomial_product_ascending(left, right):
    """Multiply tiny polynomials, broadcasting leading axes and unrolling sums."""
    terms = []
    for output_index in range(left.shape[-1] + right.shape[-1] - 1):
        products = [
            left[..., index] * right[..., output_index - index]
            for index in range(left.shape[-1])
            if 0 <= output_index - index < right.shape[-1]
        ]
        value = products[0]
        for product in products[1:]:
            value = value + product
        terms.append(value)
    return jnp.stack(terms, axis=-1)


def source_polynomial(w, lens):
    """Return descending source-image coefficients and the public image shift.

    Binary coefficients retain their mass-ratio-aware factorization. Triple
    elimination is assembled about the lens nearest the source in its local
    Einstein-radius scale. This resolves the relevant small image cluster
    without subtracting large global monomials. A limb batch shares one origin.
    The returned shift always restores the established public binary-COM frame.
    """
    if lens.positions.shape[0] == 2:
        return binary_quintic_coefficients(w, s=lens.s, q=lens.q)
    w = jnp.asarray(w)
    diagonal = jnp.eye(lens.positions.size, dtype=bool)
    separation = jnp.conj(lens.positions[:, None] - lens.positions[None, :])
    background = lens.positions - jnp.sum(
        jnp.where(diagonal, 0.0j, lens.masses[None, :] / jnp.where(diagonal, 1.0, separation)), axis=-1
    )
    score = jnp.abs(jnp.mean(w) - background) ** 2 / lens.masses
    shifted = lens.positions[jnp.argmin(score)]
    multiply = _polynomial_product_ascending
    factors = [jnp.stack((shifted - position, jnp.ones_like(shifted))) for position in lens.positions]
    denominator = multiply(multiply(factors[0], factors[1]), factors[2])
    deflection = sum(
        lens.masses[index] * multiply(factors[(index + 1) % 3], factors[(index + 2) % 3]) for index in range(3)
    )
    # Eliminating conjugate(z) gives T_j=(conj(w)-conj(z_j))*D+R,
    # then P=(z-w)*product(T_j)-D*sum(m_j*product(T_k,k!=j)).
    terms = [
        jnp.conj(w - position)[..., None] * denominator + jnp.pad(deflection, (0, 1)) for position in lens.positions
    ]
    numerator = multiply(
        jnp.stack((shifted - w, jnp.ones_like(w)), axis=-1), multiply(multiply(terms[0], terms[1]), terms[2])
    )
    eliminated = sum(
        lens.masses[index] * multiply(terms[(index + 1) % 3], terms[(index + 2) % 3]) for index in range(3)
    )
    product = multiply(denominator, eliminated)
    padding = [(0, 0)] * (product.ndim - 1) + [(0, 1)]
    return (numerator - jnp.pad(product, padding))[..., ::-1], shifted
