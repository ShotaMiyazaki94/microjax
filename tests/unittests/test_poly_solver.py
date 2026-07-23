import numpy as _np
import pytest
from microjax.poly_solver import poly_roots, poly_roots_self_inversive_fixed


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


def test_fixed_self_inversive_roots_and_implicit_jvp():
    import jax
    import jax.numpy as jnp

    expected = _np.asarray(
        [
            _np.exp(0.3j),
            _np.exp(-0.8j),
            2.0 * _np.exp(0.5j),
            0.5 * _np.exp(0.5j),
            3.0 * _np.exp(-1.0j),
            (1.0 / 3.0) * _np.exp(-1.0j),
        ]
    )
    coeffs = jnp.asarray(_np.poly(expected), dtype=jnp.complex128)
    roots = poly_roots_self_inversive_fixed(coeffs[None, :])[0]
    relative_residual = jnp.abs(jnp.polyval(coeffs, roots)) / jnp.maximum(
        jnp.polyval(jnp.abs(coeffs), jnp.abs(roots)),
        jnp.finfo(jnp.float64).tiny,
    )
    assert float(jnp.max(relative_residual)) < 1e-12

    tangent = jnp.asarray(
        [0.2 - 0.1j, -0.3 + 0.4j, 0.1j, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.complex128,
    )

    def root_sum(values):
        return jnp.sum(poly_roots_self_inversive_fixed(values[None, :])[0])

    _, computed = jax.jvp(root_sum, (coeffs,), (tangent,))
    expected_tangent = -(
        tangent[1] * coeffs[0] - coeffs[1] * tangent[0]
    ) / coeffs[0] ** 2
    assert jnp.allclose(computed, expected_tangent, rtol=1e-10, atol=1e-10)


def test_fixed_self_inversive_solver_supports_degree_eight():
    import jax.numpy as jnp

    expected = _np.asarray(
        [
            _np.exp(0.2j),
            _np.exp(-0.7j),
            _np.exp(1.4j),
            _np.exp(-2.2j),
            2.0 * _np.exp(0.4j),
            0.5 * _np.exp(0.4j),
            3.0 * _np.exp(-1.0j),
            (1.0 / 3.0) * _np.exp(-1.0j),
        ]
    )
    coefficients = jnp.asarray(
        _np.poly(expected), dtype=jnp.complex128
    )
    roots = poly_roots_self_inversive_fixed(coefficients[None, :])[0]
    relative_residual = jnp.abs(jnp.polyval(coefficients, roots)) / jnp.maximum(
        jnp.polyval(jnp.abs(coefficients), jnp.abs(roots)),
        jnp.finfo(jnp.float64).tiny,
    )
    assert float(jnp.max(relative_residual)) < 1e-12
