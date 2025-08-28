import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from microjax.fastlens.special import j0, j1, j2, j1p5, ellipk, ellipe
from scipy.special import j0 as j0_s, j1 as j1_s, jn as jn_s, jv as jv_s
from scipy.special import ellipk as ellipk_s, ellipe as ellipe_s


def test_bessel_at_zero_and_small():
    # Small arguments agree with SciPy; include zero
    xs = jnp.array([0.0, 1e-6, 1e-4, 1e-2])
    # Known limits at zero where defined
    assert np.isclose(np.array(j0(xs))[0], 1.0, atol=1e-15)
    assert np.isclose(np.array(j1(xs))[0], 0.0, atol=1e-15)
    assert jnp.allclose(j0(xs), j0_s(xs), atol=1e-12)
    assert jnp.allclose(j1(xs), j1_s(xs), atol=1e-12)
    xs_nz = jnp.array([1e-12, 1e-8, 1e-4, 1e-2])
    assert jnp.allclose(j2(xs_nz), jn_s(2, xs_nz), atol=1e-12)
    assert float(j2(xs_nz)[0]) < 1e-9
    assert jnp.allclose(j1p5(xs_nz), jv_s(1.5, xs_nz), atol=1e-12)


def test_elliptic_edges():
    # m=0: both are pi/2 (use scalar to avoid carry shape mismatch path)
    m0 = 0.0
    assert np.isclose(np.array(ellipk(m0)), ellipk_s(m0), atol=1e-14)
    assert np.isclose(np.array(ellipe(m0)), ellipe_s(m0), atol=1e-14)

    # small +/- m should match SciPy closely; iterate scalars
    for m in [-1e-6, -1e-4, 1e-6, 1e-4]:
        assert np.isclose(np.array(ellipk(m)), ellipk_s(m), rtol=1e-10, atol=1e-12)
        assert np.isclose(np.array(ellipe(m)), ellipe_s(m), rtol=1e-10, atol=1e-12)
