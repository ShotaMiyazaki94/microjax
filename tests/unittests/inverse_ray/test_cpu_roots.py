import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.roots import (
    companion_roots,
    polished_real_companion_roots,
    real_companion_roots,
)
from microjax.poly_solver import poly_roots

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.fast


def test_companion_roots_match_ehrlich_aberth_polynomial_residuals():
    coefficients = jnp.asarray([1.0, -0.4, 0.7, -1.2, 0.3, 0.2, -0.1])
    roots = companion_roots(coefficients)
    reference = poly_roots(coefficients[None, :])[0]
    residual = jax.vmap(lambda root: jnp.abs(jnp.polyval(coefficients, root)))(roots)
    reference_residual = jax.vmap(
        lambda root: jnp.abs(jnp.polyval(coefficients, root))
    )(reference)
    assert float(jnp.max(residual)) < 1e-12
    pairwise_distance = np.abs(
        np.asarray(roots)[:, None] - np.asarray(reference)[None, :]
    )
    assert np.max(np.min(pairwise_distance, axis=1)) < 1e-10
    assert float(jnp.max(reference_residual)) < 1e-12


def test_companion_root_implicit_jvp_satisfies_linearized_polynomial():
    coefficients = jnp.asarray([1.0, -0.4, 0.7, -1.2, 0.3, 0.2, -0.1])
    tangent = jnp.asarray([0.0, 0.1, -0.2, 0.05, 0.03, -0.04, 0.02])
    roots, root_tangent = jax.jvp(companion_roots, (coefficients,), (tangent,))
    derivative = jnp.polyval(coefficients[:-1] * jnp.arange(6, 0, -1), roots)
    exponents = jnp.arange(6, -1, -1)
    coefficient_term = (roots[:, None] ** exponents[None, :]) @ tangent
    np.testing.assert_allclose(
        np.asarray(derivative * root_tangent + coefficient_term),
        0.0,
        rtol=1e-9,
        atol=1e-9,
    )


def test_real_companion_roots_match_complex_companion_roots():
    coefficients = jnp.asarray([1.0, -0.4, 0.7, -1.2, 0.3, 0.2, -0.1])
    roots = real_companion_roots(coefficients)
    reference = companion_roots(coefficients)
    residual = jnp.abs(jnp.polyval(coefficients, roots))
    assert float(jnp.max(residual)) < 1e-12
    pairwise_distance = np.abs(
        np.asarray(roots)[:, None] - np.asarray(reference)[None, :]
    )
    assert np.max(np.min(pairwise_distance, axis=1)) < 1e-10


def test_real_companion_root_implicit_jvp_satisfies_linearized_polynomial():
    coefficients = jnp.asarray([1.0, -0.4, 0.7, -1.2, 0.3, 0.2, -0.1])
    tangent = jnp.asarray([0.0, 0.1, -0.2, 0.05, 0.03, -0.04, 0.02])
    roots, root_tangent = jax.jvp(
        real_companion_roots, (coefficients,), (tangent,)
    )
    derivative = jnp.polyval(coefficients[:-1] * jnp.arange(6, 0, -1), roots)
    exponents = jnp.arange(6, -1, -1)
    coefficient_term = (roots[:, None] ** exponents[None, :]) @ tangent
    np.testing.assert_allclose(
        np.asarray(derivative * root_tangent + coefficient_term),
        0.0,
        rtol=1e-9,
        atol=1e-9,
    )


def test_polished_real_companion_roots_match_complex_companion_roots():
    coefficients = jnp.asarray([1.0, -0.4, 0.7, -1.2, 0.3, 0.2, -0.1])
    roots = polished_real_companion_roots(coefficients)
    reference = companion_roots(coefficients)
    residual = jnp.abs(jnp.polyval(coefficients, roots))
    assert float(jnp.max(residual)) < 1e-12
    pairwise_distance = np.abs(
        np.asarray(roots)[:, None] - np.asarray(reference)[None, :]
    )
    assert np.max(np.min(pairwise_distance, axis=1)) < 1e-10


def test_polished_real_companion_root_jvp_satisfies_linearized_polynomial():
    coefficients = jnp.asarray([1.0, -0.4, 0.7, -1.2, 0.3, 0.2, -0.1])
    tangent = jnp.asarray([0.0, 0.1, -0.2, 0.05, 0.03, -0.04, 0.02])
    roots, root_tangent = jax.jvp(
        polished_real_companion_roots,
        (coefficients,),
        (tangent,),
    )
    derivative = jnp.polyval(coefficients[:-1] * jnp.arange(6, 0, -1), roots)
    exponents = jnp.arange(6, -1, -1)
    coefficient_term = (roots[:, None] ** exponents[None, :]) @ tangent
    np.testing.assert_allclose(
        np.asarray(derivative * root_tangent + coefficient_term),
        0.0,
        rtol=1e-9,
        atol=1e-9,
    )
