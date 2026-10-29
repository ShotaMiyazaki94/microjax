"""End-to-end forward gradients of a smooth angle-first source."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.one_shot import (
    ONE_SHOT_POLAR_THETA_FIRST, mag_limb_dark_cpu_one_shot,
)

jax.config.update('jax_enable_x64', True)


@pytest.mark.fast
def test_smooth_angular_source_all_forward_derivatives_match_finite_differences():
    """A finite gradient alone misses the linear-outer-rule regression.

    A smooth angular brightness integral can have endpoint-singular parameter
    derivatives. Its fixed outer rule must resolve those derivatives as well
    as the magnification. The historical implementation has a 1.25e-3 relative
    rho discrepancy here, so the 2e-3 tolerance retains that existing accuracy
    floor. The rejected linear rule instead gave 6--10% errors in x/y/s/q
    and a wrong-sign rho derivative (about -26.4 versus +0.404).
    """

    point = jnp.asarray([.1, .2, .01, 1., .3, .5])

    @jax.jit
    def evaluate(parameters):
        x, y, rho, s, q, u1 = parameters
        result = mag_limb_dark_cpu_one_shot(
            x+1j*y, rho, s=s, q=q, u1=u1, n_limb=56,
        )
        return jnp.asarray([result.magnification, result.stage, result.status])

    value = np.asarray(evaluate(point))
    assert int(value[1]) == ONE_SHOT_POLAR_THETA_FIRST
    assert int(value[2]) == 0
    forward = np.asarray(jax.jit(jax.jacfwd(lambda p: evaluate(p)[0]))(point))
    steps = np.asarray([.01, .01, .01, .01, .3, 1.])*1e-3
    finite_difference = []
    for dimension, step in enumerate(steps):
        upper = np.asarray(evaluate(point.at[dimension].add(step)))
        lower = np.asarray(evaluate(point.at[dimension].add(-step)))
        np.testing.assert_array_equal(upper[1:], value[1:])
        np.testing.assert_array_equal(lower[1:], value[1:])
        finite_difference.append((upper[0]-lower[0])/(2*step))
    assert np.isfinite(forward).all()
    np.testing.assert_allclose(forward, finite_difference, rtol=2e-3, atol=1e-6)
