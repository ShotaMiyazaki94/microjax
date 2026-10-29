"""Fixed projection rules preserve their measure, diagnostics, and bounds AD."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.projection_quadrature import (
    _RADIAL21_COARSE,
    _RADIAL21_W,
    _RADIAL21_X,
    _integrate_radial21,
    _three_panel_linear_rule,
    _three_panel_sin2_rule,
)


@pytest.mark.parametrize("order", [21, 24])
def test_transformed_rule_integrates_square_root_endpoints(order):
    nodes, weights = _three_panel_sin2_rule(order, jnp.float64)
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    assert nodes.size == order
    assert np.all(np.diff(nodes) > 0)
    assert np.all((nodes > 0) & (nodes < 1))
    assert np.all(weights > 0)
    # These are moments in the physical coordinate, including its Jacobian.
    for degree in range(3):
        np.testing.assert_allclose(weights @ nodes**degree, 1 / (degree + 1),
                                   rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(weights @ np.sqrt(nodes * (1 - nodes)),
                               np.pi / 8, rtol=2e-12, atol=2e-13)


def test_radial21_embedded_rule_has_local_cubic_moments():
    # Undo the physical Jacobian to check each rule in its Gauss coordinate.
    theta = np.arcsin(np.sqrt(_RADIAL21_X)).reshape(3, 7)
    local = 12 * theta / np.pi - (2 * np.arange(3)[:, None] + 1)
    jacobian = np.pi / 12 * np.sin(2 * theta)
    high = _RADIAL21_W.reshape(3, 7) / jacobian
    low = _RADIAL21_COARSE.reshape(3, 7) / jacobian
    assert np.count_nonzero(low) == 12
    for weights, max_degree in ((high, 13), (low, 3)):
        for degree in range(max_degree + 1):
            expected = 2 / (degree + 1) if degree % 2 == 0 else 0.0
            np.testing.assert_allclose(np.sum(weights * local**degree, axis=1),
                                       expected, rtol=2e-12, atol=2e-12)


def test_linear_composite24_integrates_polynomial_moments():
    nodes, weights = _three_panel_linear_rule(24, jnp.float64)
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    assert nodes.size == 24
    assert np.all(weights > 0)
    for degree in range(16):
        np.testing.assert_allclose(weights @ nodes**degree, 1 / (degree + 1),
                                   rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize("subdivisions", [1, 3])
@pytest.mark.parametrize("kind", ["polynomial", "exponential"])
def test_radial21_smooth_integrals_match_antiderivatives(kind, subdivisions):
    lower, upper = -0.4, 1.3
    if kind == "polynomial":
        function = lambda radius: radius**2 + 2 * radius + 3
        primitive = lambda radius: radius**3 / 3 + radius**2 + 3 * radius
        expected = primitive(upper) - primitive(lower)
    else:
        function = jnp.exp
        expected = np.exp(upper) - np.exp(lower)

    def integrand(radius):
        return function(radius), jnp.asarray(0.0), jnp.int32(0)

    result = jax.jit(lambda bounds: _integrate_radial21(
        integrand, bounds, jnp.asarray([True]), subdivisions=subdivisions))(
            jnp.asarray([[lower, upper]]))
    np.testing.assert_allclose(result.value, expected, rtol=2e-11, atol=2e-12)
    assert float(result.error) >= 0
    assert int(result.status) == 0
    assert int(result.n_nodes) == 21 * subdivisions


def test_radial21_preserves_scaled_square_root_endpoint_integral():
    lower, upper = 2.0, 5.0

    def integrand(radius):
        value = jnp.sqrt(jnp.maximum(0.0, (radius - lower) * (upper - radius)))
        return value, jnp.asarray(0.0), jnp.int32(0)

    result = _integrate_radial21(integrand, jnp.asarray([[lower, upper]]),
                                jnp.asarray([True]))
    np.testing.assert_allclose(result.value, np.pi * (upper - lower)**2 / 8,
                               rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("subdivisions", [1, 2])
def test_radial21_masks_noncontiguous_cells_and_propagates_inner_error(subdivisions):
    bounds = jnp.asarray([[2.0, 2.8], [jnp.nan, jnp.nan], [4.0, 5.2]])
    active = jnp.asarray([True, False, True])

    def integrand(radius):
        padding = radius == 1.0
        value = jnp.where(padding, 1e12, radius**2)
        error = jnp.where(padding, 1e12, 0.01)
        status = jnp.where(padding, jnp.int32(128),
                           jnp.where(radius < 3.0, jnp.int32(1), jnp.int32(4)))
        return value, error, status

    def without_inner_error(radius):
        value, _, status = integrand(radius)
        return value, jnp.asarray(0.0), status

    result = jax.jit(lambda cells: _integrate_radial21(
        integrand, cells, active, subdivisions=subdivisions))(bounds)
    outer_only = _integrate_radial21(without_inner_error, bounds, active,
                                     subdivisions=subdivisions)
    expected = (2.8**3 - 2.0**3 + 5.2**3 - 4.0**3) / 3
    np.testing.assert_allclose(result.value, expected, rtol=2e-12, atol=2e-12)
    # Constant pointwise absolute error accumulates with the physical measure.
    np.testing.assert_allclose(result.error - outer_only.error,
                               0.01 * (2.8 - 2.0 + 5.2 - 4.0),
                               rtol=2e-10, atol=2e-12)
    assert int(result.status) == 5
    assert int(result.n_nodes) == 42 * subdivisions


def test_radial21_all_inactive_cells_have_zero_value_error_and_status():
    def invalid_integrand(radius):
        return jnp.asarray(jnp.nan), jnp.asarray(jnp.inf), jnp.int32(128)

    result = jax.jit(lambda bounds: _integrate_radial21(
        invalid_integrand, bounds, jnp.asarray([False, False])))(
            jnp.asarray([[jnp.nan, jnp.nan], [-jnp.inf, jnp.inf]]))
    assert float(result.value) == 0.0
    assert float(result.error) == 0.0
    assert int(result.status) == 0
    assert int(result.n_nodes) == 0


def test_radial21_bounds_jvp_matches_leibniz_rule_with_inactive_padding():
    active = jnp.asarray([True, False, True])

    def integrate(bounds):
        def integrand(radius):
            return jnp.exp(radius), jnp.asarray(0.0), jnp.int32(0)
        return _integrate_radial21(integrand, bounds, active, subdivisions=2).value

    bounds = jnp.asarray([[-0.4, 0.6], [jnp.nan, jnp.nan], [0.8, 1.3]])
    tangent = jnp.asarray([[0.2, -0.3], [100.0, -100.0], [-0.4, 0.7]])
    value, derivative = jax.jit(lambda cells, direction:
        jax.jvp(integrate, (cells,), (direction,)))(bounds, tangent)
    expected_value = np.exp(0.6) - np.exp(-0.4) + np.exp(1.3) - np.exp(0.8)
    expected_derivative = (-0.2 * np.exp(-0.4) - 0.3 * np.exp(0.6)
                           + 0.4 * np.exp(0.8) + 0.7 * np.exp(1.3))
    np.testing.assert_allclose(value, expected_value, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=2e-11, atol=2e-12)
