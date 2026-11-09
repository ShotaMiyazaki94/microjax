"""Compact derived lens geometry shared by inverse-ray layers."""

from typing import NamedTuple

import jax.numpy as jnp


class BinaryGeometry(NamedTuple):
    """Binary lens parameters in the midpoint calculation frame."""

    s: jnp.ndarray
    q: jnp.ndarray
    a: jnp.ndarray
    e1: jnp.ndarray
    shifted: jnp.ndarray


def binary_geometry(s: float, q: float) -> BinaryGeometry:
    """Derive midpoint-frame geometry from separation and mass ratio."""

    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    return BinaryGeometry(s, q, a, e1, shifted)


__all__ = ["BinaryGeometry", "binary_geometry"]
