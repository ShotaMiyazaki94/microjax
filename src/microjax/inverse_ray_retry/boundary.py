"""Boundary and membership helpers for inverse-ray integration.

Provides JAX-compatible utilities for classifying points relative to the source
disk and for computing smooth boundary weights used in finite-source angular
integration. Custom JVPs avoid zero gradients at discontinuities.
"""

import jax.numpy as jnp
from microjax.point_source import lens_eq

Array = jnp.ndarray

def distance_from_source(
    r0: float,
    th_values: Array,
    w_center_shifted: complex,
    shifted: float,
    nlenses: int = 2,
    **_params,
) -> Array:
    """Distance to source center for a fixed radius and set of angles."""
    x_th = r0 * jnp.cos(th_values)
    y_th = r0 * jnp.sin(th_values)
    z_th = x_th + 1j * y_th
    image_mesh = lens_eq(z_th - shifted, nlenses=nlenses, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    return distances
