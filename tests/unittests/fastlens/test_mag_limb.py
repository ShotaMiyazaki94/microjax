import numpy as np

from microjax.fastlens.mag_fft import magnification_limb


def test_mag_limb_A0_matches_A_at_zero_for_large_rho():
    # n_limb=1 and 2 for coverage
    for n in (1, 2):
        rho = 1e-2
        mag = magnification_limb(n_limb=n)
        a0 = mag.A0(rho)
        a = mag.A(0.0, rho)
        assert np.allclose(a, a0, rtol=1e-10, atol=1e-12)


def test_mag_limb_small_rho_point_source_limit():
    rho = 1e-6
    u = 1e2 * rho
    mag = magnification_limb(n_limb=1)
    # For small rho and large u relative to rho, A approaches 1 (point source ~1 at large u)
    a = mag.A(u, rho)
    assert np.all(a > 1.0)
    # As u increases further, approach 1
    a2 = mag.A(3e2 * rho, rho)
    assert (a2 - 1.0) < (a - 1.0)

