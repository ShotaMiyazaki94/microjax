"""Forward differentiation for a formerly radial LD regression case."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.one_shot import (
    ONE_SHOT_POLAR_THETA_FIRST,
    mag_limb_dark_cpu_one_shot,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.fast
def test_zero_brightness_samples_keep_all_forward_derivatives_finite():
    """A zero clipped radicand must not introduce zero-times-infinity JVPs."""
    parameters = jnp.asarray([
        0.16039205046000904, -8.75250608210678e-5,
        0.00021142727844446993, 1.125, 0.0003727593720314938, .5,
    ])

    def value(p):
        result = mag_limb_dark_cpu_one_shot(
            p[0] + 1j * p[1], p[2], s=p[3], q=p[4], u1=p[5], n_limb=56,
        )
        return result.magnification, (result.magnification, result.stage)

    jacobian, (magnification, stage) = jax.jit(jax.jacfwd(value, has_aux=True))(parameters)
    assert int(stage) == ONE_SHOT_POLAR_THETA_FIRST
    assert np.isfinite(np.asarray(jacobian)).all()
    # Independent VBMicrolensing reference: Tol=RelTol=1e-9 and 1e-10
    # agree within 3e-11 relative. Check physical accuracy, rather than
    # freezing the previous quadrature's value (39.434088360655984).
    np.testing.assert_allclose(magnification, 39.440227256801045, rtol=1e-4, atol=0)

    magnitude = jax.jit(lambda p: value(p)[0])
    uniform = magnitude(parameters.at[5].set(0.0))
    fully_darkened = magnitude(parameters.at[5].set(1.0))
    # Linear surface brightness and its 3-u1 normalization give this exact
    # derivative, independent of image geometry or finite-difference steps.
    expected = 6 * (fully_darkened - uniform) / (3 - parameters[5])**2
    np.testing.assert_allclose(jacobian[5], expected, rtol=1e-9, atol=1e-8)
