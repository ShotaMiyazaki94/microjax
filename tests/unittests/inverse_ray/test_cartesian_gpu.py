"""GPU Cartesian chart algebra and low-q conditioning regressions."""

import jax
import jax.numpy as jnp
import numpy as np

from microjax.inverse_ray.geometry.lens import binary_geometry
from microjax.inverse_ray.integrators.cartesian_gpu import (
    binary_line_level_set_coefficients_gpu,
)
from microjax.inverse_ray.roots.level_set import binary_level_set


jax.config.update("jax_enable_x64", True)


def test_q_aware_line_sextic_matches_direct_low_q_level_set():
    q = jnp.asarray(1.0e-6)
    s = jnp.asarray(0.75)
    rho = jnp.asarray(3.0e-5)
    w_center = jnp.asarray(-0.58334 - 0.00202j)
    lens = binary_geometry(s, q)
    axis = w_center / jnp.abs(w_center)
    offset = jnp.asarray(-0.58331 - 0.00201j)
    direction = 1.0j * axis
    coefficients = binary_line_level_set_coefficients_gpu(
        offset,
        direction,
        w_center,
        rho,
        s=s,
        q=q,
    )
    ordinate = jnp.linspace(-2.0e-3, 2.0e-3, 257)
    polynomial = jax.vmap(lambda value: jnp.polyval(coefficients, value))(ordinate)
    direct = binary_level_set(
        offset + ordinate * direction,
        w_center - lens.shifted,
        rho,
        lens.shifted,
        a=lens.a,
        e1=lens.e1,
    )
    scale = jnp.vdot(polynomial, direct) / jnp.vdot(polynomial, polynomial)
    residual = np.asarray(direct - scale * polynomial)
    assert np.max(np.abs(residual)) <= 5.0e-12 * np.max(np.abs(np.asarray(direct)))
