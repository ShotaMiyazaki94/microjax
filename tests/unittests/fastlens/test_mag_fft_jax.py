import numpy as np

from microjax.fastlens.mag_fft_jax import magnification_disk as jx_magnification_disk
from microjax.fastlens.mag_fft_jax import A_point as jx_A_point


def test_jax_magnification_disk_A0_large_rho_matches():
    rho = 1e-2
    mag = jx_magnification_disk()
    a0_expected = np.sqrt(rho**2 + 4) / rho
    a = np.array(mag.A(0.0, rho))
    assert np.allclose(a, a0_expected, rtol=1e-10, atol=1e-12)


def test_jax_magnification_disk_small_rho_limits():
    rho = 1e-6
    mag = jx_magnification_disk()
    u = 100 * rho
    a = np.array(mag.A(u, rho))
    assert np.allclose(a, np.array(jx_A_point(u)), rtol=1e-6, atol=1e-8)


def test_jax_A_point_monotone_to_one():
    u = np.array([10.0, 30.0])
    ap = np.array(jx_A_point(u))
    assert np.all(ap > 1.0)
    assert (ap[1] - 1.0) < (ap[0] - 1.0)

