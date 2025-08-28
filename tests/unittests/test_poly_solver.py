import numpy as _np
import pytest
from microjax.poly_solver import poly_roots


def _sort_by_angle(z, jnp):
    # Stable sort for complex roots: by angle then magnitude
    ang = jnp.angle(z)
    mag = jnp.abs(z)
    order = jnp.lexsort((mag, ang))
    return z[order]


def test_poly_roots_quadratic():
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    from jax import config
    import jax.numpy as jnp
    config.update("jax_enable_x64", True)

    # (z-1)(z+2) = z^2 + z - 2 -> coeffs [1, 1, -2]
    try:
        coeffs = jnp.array(_np.array([1.0 + 0j, 1.0 + 0j, -2.0 + 0j]))
        roots = _sort_by_angle(poly_roots(coeffs[None, :])[0], jnp)
        expected = _sort_by_angle(jnp.array(_np.array([1.0 + 0j, -2.0 + 0j])), jnp)
        assert jnp.allclose(roots, expected, atol=1e-9)
    except RuntimeError as e:
        if "Unable to initialize backend" in str(e):
            pytest.skip("Skipping on non-CPU JAX backend environment")
        raise


def test_poly_roots_cubic():
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    from jax import config
    import jax.numpy as jnp
    config.update("jax_enable_x64", True)

    # (z-1)(z-2)(z-3) = z^3 - 6z^2 + 11z - 6
    try:
        coeffs = jnp.array(_np.array([1.0 + 0j, -6.0 + 0j, 11.0 + 0j, -6.0 + 0j]))
        roots = _sort_by_angle(poly_roots(coeffs[None, :])[0], jnp)
        expected = _sort_by_angle(jnp.array(_np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])), jnp)
        assert jnp.allclose(roots, expected, atol=1e-8)
    except RuntimeError as e:
        if "Unable to initialize backend" in str(e):
            pytest.skip("Skipping on non-CPU JAX backend environment")
        raise
