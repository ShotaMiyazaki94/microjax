"""Shared lens geometry and coordinate transforms.

Public triple-lens coordinates retain the centre of mass of the first two
lenses. This is the useful perturbative frame: varying the third body does not
translate an otherwise fixed binary event. Polynomial and rational lens
equations use the midpoint of those first two lenses. The total three-body
centre of mass is retained as internal geometry, but is not the public origin.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class TripleLensGeometry(NamedTuple):
    """Mass fractions and internal/public coordinate displacements."""

    a: jnp.ndarray
    e1: jnp.ndarray
    e2: jnp.ndarray
    e3: jnp.ndarray
    r3_complex: jnp.ndarray
    shifted: jnp.ndarray
    total_shifted: jnp.ndarray


def triple_lens_geometry(
    s: float,
    q: float,
    q3: float,
    r3: float,
    psi: float,
) -> TripleLensGeometry:
    """Return the canonical geometry for the public triple-lens parameters.

    The first (primary) and second lenses are at ``-a`` and ``+a`` in the
    midpoint frame, with mass fractions ``e2`` and ``e1`` respectively.  The
    third lens is at ``r3 * exp(1j*psi)`` and has fraction ``e3``.

    ``shifted`` follows the established public binary-COM convention:

    ``coordinate_midpoint = coordinate_binary_com - shifted``.

    It is deliberately independent of ``q3``, ``r3``, and ``psi`` so a triple
    lens can be studied as a perturbation of a fixed binary event.
    ``total_shifted`` gives the corresponding displacement from the total
    three-body centre of mass for internal diagnostics; it must not silently
    change the public source-coordinate frame.
    """

    total_mass = 1.0 + q + q3
    a = 0.5 * s
    e1 = q / total_mass
    e2 = 1.0 / total_mass
    e3 = q3 / total_mass
    r3_complex = r3 * jnp.exp(1j * psi)
    shifted = a * (1.0 - q) / (1.0 + q)
    total_center_of_mass_midpoint = -a * e2 + a * e1 + e3 * r3_complex
    total_shifted = -total_center_of_mass_midpoint
    return TripleLensGeometry(
        a, e1, e2, e3, r3_complex, shifted, total_shifted
    )
