import numpy as np

from microjax.fastlens.mag_fft_jax import magnification_limb1, magnification_limb2


def test_mag_limb1_A0_matches_A_at_zero_for_large_rho():
    rho = 1e-2
    mag = magnification_limb1()
    a0 = np.array(mag.A0(rho))
    a = np.array(mag.A(0.0, rho))
    assert np.allclose(a, a0, rtol=1e-10, atol=1e-12)


def test_mag_limb2_A0_matches_A_at_zero_for_large_rho():
    rho = 1e-2
    mag = magnification_limb2()
    a0 = np.array(mag.A0(rho))
    a = np.array(mag.A(0.0, rho))
    assert np.allclose(a, a0, rtol=1e-10, atol=1e-12)

