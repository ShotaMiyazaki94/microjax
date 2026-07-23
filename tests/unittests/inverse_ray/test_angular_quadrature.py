import jax
import jax.numpy as jnp
import numpy as np

from microjax.inverse_ray.roots.angular import ANGULAR_OK, AngularIntervals
from microjax.inverse_ray.quadrature.angular import integrate_angular_profile
from microjax.inverse_ray.profiles import linear_limb_intensity


def test_angular_profile_quadrature_integrates_active_intervals_only():
    intervals = AngularIntervals(
        intervals=jnp.asarray(
            [[0.2, 1.3], [2.0, 2.5], [10.0, 11.0], [20.0, 21.0]]
        ),
        n_intervals=jnp.int32(2),
        error=jnp.asarray(2e-9),
        status=jnp.int32(ANGULAR_OK),
    )
    result = integrate_angular_profile(
        lambda theta: 2.0 + 0.0 * theta,
        intervals,
        endpoint_value_bound=jnp.asarray(2.0),
    )

    assert int(result.status) == ANGULAR_OK
    assert np.isclose(float(result.value), 3.2, rtol=0.0, atol=1e-14)
    assert 4e-9 <= float(result.error) < 5e-9


def test_inactive_angular_interval_does_not_poison_reverse_mode():
    intervals = AngularIntervals(
        intervals=jnp.asarray(
            [[0.2, 0.4], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ),
        n_intervals=jnp.int32(1),
        error=jnp.asarray(0.0),
        status=jnp.int32(ANGULAR_OK),
    )

    def integral(scale):
        return integrate_angular_profile(
            lambda theta: jnp.sqrt(scale * theta), intervals
        ).value

    forward = jax.jacfwd(integral)(jnp.asarray(1.0))
    reverse = jax.grad(integral)(jnp.asarray(1.0))
    assert np.isfinite(float(reverse))
    assert np.isclose(float(reverse), float(forward), rtol=0.0, atol=1e-13)


def test_linear_limb_intensity_has_finite_clipped_reverse_rule():
    distances = jnp.asarray([0.999, 1.0, 1.0 + 1e-12, 2.0])
    u1 = jnp.asarray(0.5)

    def total(values):
        return jnp.sum(linear_limb_intensity(values, u1=u1))

    values = linear_limb_intensity(distances, u1=u1)
    reverse = jax.grad(total)(distances)
    normalization = 3.0 / (jnp.pi * (3.0 - u1))
    assert np.all(np.isfinite(np.asarray(reverse)))
    assert np.isclose(
        float(values[1]), float(normalization * (1.0 - u1)), atol=1e-15
    )
    assert np.array_equal(np.asarray(values[2:]), np.zeros(2))


def test_angular_profile_fixed_subdivision_reduces_oscillatory_error():
    intervals = AngularIntervals(
        intervals=jnp.asarray(
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ),
        n_intervals=jnp.int32(1),
        error=jnp.asarray(0.0),
        status=jnp.int32(ANGULAR_OK),
    )
    integrand = lambda theta: jnp.sin(100.0 * theta)
    coarse = integrate_angular_profile(integrand, intervals, subdivisions=1)
    refined = integrate_angular_profile(integrand, intervals, subdivisions=4)
    exact = (1.0 - np.cos(100.0)) / 100.0

    assert abs(float(refined.value) - exact) < abs(float(coarse.value) - exact)
    assert float(refined.error) < float(coarse.error)
