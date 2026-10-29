"""Numerical and physical contracts for the reusable moment integrator."""

from math import erf, pi, sqrt
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from microjax.inverse_ray.cpu.projection_moments import (
    integrate_moments,
    _GL5_X,
    _GL5_W,
    _E3_W,
    _sum_status,
)


def test_embedded_rule_polynomial_exactness():
    assert np.all(_GL5_W > 0)
    assert np.all(_E3_W >= 0)
    for weights, degree in [(_GL5_W, 9), (_E3_W, 3)]:
        for k in range(degree + 1):
            expected = 0.0 if k % 2 else 2.0 / (k + 1)
            np.testing.assert_allclose(
                np.dot(weights, _GL5_X**k), expected, rtol=0, atol=1e-14
            )


def test_two_separated_profile_moments_match_analytic_integrals():
    cells = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    active = jnp.array([True, True, True])

    def profile(x):
        y = jnp.stack(
            [jnp.exp(-40 * (x - 0.4) ** 2), jnp.exp(-160 * (x - 2.7) ** 2)], axis=-1
        )
        return y, jnp.zeros_like(y), jnp.zeros(x.shape, dtype=jnp.int32)

    coarse = jax.jit(lambda: integrate_moments(profile, cells, active, budget=18))()
    integral = jax.jit(lambda: integrate_moments(profile, cells, active, budget=144))()

    def exact(rate, center):
        return (
            sqrt(pi / rate)
            / 2
            * (erf(sqrt(rate) * (3 - center)) + erf(sqrt(rate) * center))
        )

    expected = np.asarray([exact(40, 0.4), exact(160, 2.7)])
    assert np.max(np.abs(np.asarray(integral.value) - expected)) < np.max(
        np.abs(np.asarray(coarse.value) - expected)
    )
    np.testing.assert_allclose(integral.value, expected, rtol=1e-7, atol=1e-10)


def test_work_bound_and_empty_support():
    cells = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    def profile(x):
        y = jnp.stack([x, x * x], axis=-1)
        return y, jnp.zeros_like(y), jnp.zeros(x.shape, dtype=jnp.int32)

    for count in range(5):
        result = jax.jit(
            lambda active: integrate_moments(profile, cells, active, budget=18)
        )(jnp.arange(4) < count)
        assert int(result.n_nodes) <= 18 * count
        np.testing.assert_allclose(
            result.value, [count**2 / 2, count**3 / 3], rtol=1e-7, atol=1e-7
        )
        if count == 0:
            np.testing.assert_array_equal(result.error, [0.0, 0.0])
            assert result.status == 0


def test_moving_support_derivative_matches_analytic_moments():
    def value(bounds):
        def profile(x):
            y = jnp.stack([jnp.ones_like(x), jnp.sqrt(x)], axis=-1)
            return y, jnp.zeros_like(y), jnp.zeros(x.shape, dtype=jnp.int32)

        return integrate_moments(
            profile, bounds.reshape(1, 2), jnp.array([True]), budget=72
        ).value

    bounds = jnp.array([0.2, 1.7])
    jacobian = jax.jit(jax.jacfwd(value))(bounds)
    expected = np.array([[-1.0, 1.0], [-np.sqrt(0.2), np.sqrt(1.7)]])
    np.testing.assert_allclose(jacobian, expected, rtol=1e-8, atol=1e-9)


def test_status_count_and_flag_semantics_are_distinct():
    cells = jnp.array([[0.0, 1.0], [1.0, 2.0]])
    active = jnp.array([True, True])

    def profile(x):
        y = jnp.ones((x.size, 2))
        return y, jnp.zeros_like(y), jnp.full(x.shape, 2, dtype=jnp.int32)

    counts = jax.jit(
        lambda: integrate_moments(
            profile, cells, active, budget=18, status_reducer=_sum_status
        )
    )()
    flags = jax.jit(lambda: integrate_moments(profile, cells, active, budget=18))()
    assert int(counts.status) == 50  # Five live panels × five rays × two bad roots.
    assert int(flags.status) == 2
    assert int(counts.n_nodes) == 30  # Includes the discarded initial parent panel.
