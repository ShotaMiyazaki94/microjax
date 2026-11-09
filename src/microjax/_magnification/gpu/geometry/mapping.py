"""Map image-chart quadrature nodes back to source-plane distance.

This is the lowest geometry layer used by radial brightness profiles.  A local
chart node ``z = chart_center + r * exp(i theta)`` is evaluated through the
lens equation and measured from the source centre.  It does not select image
components or perform quadrature.
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
    chart_center: complex = 0.0 + 0.0j,
    **_params,
) -> Array:
    """Distance to source center for a fixed radius and set of angles."""
    x_th = r0 * jnp.cos(th_values)
    y_th = r0 * jnp.sin(th_values)
    z_th = jnp.asarray(chart_center) + x_th + 1j * y_th
    image_mesh = lens_eq(z_th - shifted, nlenses=nlenses, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    return distances
