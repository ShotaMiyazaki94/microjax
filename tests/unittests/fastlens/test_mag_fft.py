import numpy as np

from microjax.fastlens.mag_fft import magnification_disk, A_point


def test_magnification_disk_A0_large_rho_matches():
    # Use rho larger than rho_switch to hit the large-rho FFT path
    rho = 1e-2
    mag = magnification_disk()
    a0_expected = np.sqrt(rho**2 + 4) / rho
    # At u=0, large-rho branch returns exactly A0(rho)
    a = mag.A(0.0, rho)
    assert np.allclose(a, a0_expected, rtol=1e-10, atol=1e-12)


def test_magnification_disk_small_rho_limits():
    # Small rho uses approximation: point-source for u >= u_switch * rho
    rho = 1e-6
    mag = magnification_disk()
    u = 100 * rho  # greater than u_switch * rho (default u_switch=10)
    a = mag.A(u, rho)
    assert np.allclose(a, A_point(u), rtol=1e-6, atol=1e-8)


def test_A_point_asymptotics():
    # Sanity: A_point(u) -> 1 + 1/(2 u^4) as u -> inf; test near 1
    u = np.array([10.0, 30.0])
    ap = A_point(u)
    assert np.all(ap > 1.0)
    # Closer to 1 for larger u
    assert (ap[1] - 1.0) < (ap[0] - 1.0)

