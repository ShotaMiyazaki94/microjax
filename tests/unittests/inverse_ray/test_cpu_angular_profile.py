"""Brightness quadrature must use physical endpoints on the periodic circle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.angular_profile import (
    _integrate_interval,
    integrate_mu_residual,
)
from microjax.inverse_ray.roots.angular import (
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
    AngularIntervals,
)


def _intervals(bounds, status=0):
    return AngularIntervals(
        jnp.zeros((4, 2), dtype=jnp.float64).at[:len(bounds)].set(
            jnp.asarray(bounds, dtype=jnp.float64).reshape(-1, 2)
        ),
        jnp.int32(len(bounds)),
        jnp.asarray(0.0),
        jnp.int32(status),
    )


def _semicircle(theta, center, half_width):
    distance = jnp.arctan2(jnp.sin(theta - center), jnp.cos(theta - center))
    normalized = distance / half_width
    return jnp.sqrt(jnp.maximum(0.0, (1.0 - normalized) * (1.0 + normalized)))


def _assert_scalar_and_batched_rule_agree(result, old_value, old_error):
    # Scalar and batched fusion can round the same quadrature differently.
    # The embedded error subtracts two integral values, so its roundoff is
    # measured on the integral scale even when the error itself is near zero.
    roundoff = 4.0 * np.finfo(np.float64).eps * abs(float(old_value))
    np.testing.assert_allclose(result.value, old_value, rtol=0.0, atol=roundoff)
    np.testing.assert_allclose(result.error, old_error, rtol=0.0, atol=roundoff)


@pytest.mark.parametrize("center", [-0.13, 0.0, 0.13])
def test_periodic_arc_matches_rotation_and_analytic_integral(center):
    half_width = 0.4
    split = _intervals([(0.0, center + half_width),
                        (2.0 * np.pi + center - half_width, 2.0 * np.pi)])
    rotated_center = 2.0
    rotated = _intervals([(rotated_center - half_width, rotated_center + half_width)])
    evaluate_split = jax.jit(lambda arcs: integrate_mu_residual(
        lambda theta: _semicircle(theta, center, half_width), arcs))
    evaluate_rotated = jax.jit(lambda arcs: integrate_mu_residual(
        lambda theta: _semicircle(theta, rotated_center, half_width), arcs))
    result = evaluate_split(split)
    control = evaluate_rotated(rotated)
    expected = np.pi * half_width / 2.0
    np.testing.assert_allclose(result.value, expected, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(result.value, control.value, rtol=2e-13, atol=2e-14)
    assert int(result.status) == 0


def test_joined_arc_preserves_separate_inside_component():
    half_width, middle_width, amplitude = 0.4, 0.18, 0.3
    arcs = _intervals([(0.0, half_width),
                       (np.pi - middle_width, np.pi + middle_width),
                       (2.0 * np.pi - half_width, 2.0 * np.pi)])
    result = jax.jit(lambda intervals: integrate_mu_residual(
        lambda theta: (_semicircle(theta, 0.0, half_width)
                       + amplitude * _semicircle(theta, np.pi, middle_width)),
        intervals))(arcs)
    expected = np.pi * (half_width + amplitude * middle_width) / 2.0
    np.testing.assert_allclose(result.value, expected, rtol=2e-13, atol=2e-14)
    assert int(result.status) == 0


def test_full_ring_retains_the_existing_legendre_rule_and_status():
    arcs = _intervals([(0.0, 2.0 * np.pi)], status=ANGULAR_DEGENERATE)
    mu = lambda theta: 0.6 + 0.1 * jnp.cos(theta) + 0.2 * jnp.sin(theta)
    result = integrate_mu_residual(mu, arcs)
    old_value, old_error, _ = _integrate_interval(mu, arcs.intervals[0, 0], arcs.intervals[0, 1])
    np.testing.assert_allclose(result.value, 1.2 * np.pi, rtol=2e-14, atol=2e-14)
    _assert_scalar_and_batched_rule_agree(result, old_value, old_error)
    assert int(result.status) == ANGULAR_DEGENERATE


def test_empty_arcs_preserve_diagnostic_and_do_not_evaluate_brightness():
    result = jax.jit(lambda arcs: integrate_mu_residual(
        lambda theta: jnp.asarray(jnp.nan), arcs))(_intervals([], status=ANGULAR_ROOT_FAILURE))
    assert float(result.value) == 0.0
    assert float(result.error) == 0.0
    assert int(result.status) == ANGULAR_ROOT_FAILURE


@pytest.mark.parametrize("center", [0.4, 1.5])
def test_single_nonwrapping_arc_keeps_existing_rule(center):
    half_width = 0.4
    arcs = _intervals([(center - half_width, center + half_width)])
    mu = lambda theta: _semicircle(theta, center, half_width)
    result = integrate_mu_residual(mu, arcs)
    old_value, old_error, _ = _integrate_interval(mu, arcs.intervals[0, 0], arcs.intervals[0, 1])
    np.testing.assert_allclose(result.value, np.pi * half_width / 2.0, rtol=2e-13, atol=2e-14)
    _assert_scalar_and_batched_rule_agree(result, old_value, old_error)
    assert int(result.status) == 0


def test_periodic_arc_width_and_translation_jvp_match_analytic_integral():
    def integral(parameters):
        center, half_width = parameters
        bounds = jnp.zeros((4, 2), dtype=jnp.float64)
        bounds = bounds.at[0].set(jnp.stack((jnp.asarray(0.0), center + half_width)))
        bounds = bounds.at[1].set(jnp.stack((2.0 * jnp.pi + center - half_width,
                                           jnp.asarray(2.0 * jnp.pi))))
        arcs = AngularIntervals(bounds, jnp.int32(2), jnp.asarray(0.0), jnp.int32(0))
        return integrate_mu_residual(
            lambda theta: _semicircle(theta, center, half_width), arcs).value

    parameters = jnp.asarray([0.1, 0.4])
    jacobian = jax.jit(jax.jacfwd(integral))(parameters)
    np.testing.assert_allclose(jacobian, [0.0, np.pi / 2.0], rtol=2e-12, atol=2e-12)
