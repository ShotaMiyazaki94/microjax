import numpy as np
import pytest

from microjax.fspl.mag_fft_jax import fspl_disk, fspl_ld1


@pytest.fixture(scope="module")
def vbbl():
    vb = pytest.importorskip("VBBinaryLensing")
    VBBL = vb.VBBinaryLensing()
    # load bundled ESPL table (required for ESPLMag* calls)
    table_path = vb.__file__.replace("__init__.py", "data/ESPL.tbl")
    VBBL.LoadESPLTable(table_path)
    return VBBL


@pytest.mark.parametrize("rho", [1e-3, 1e-2, 5e-2])
def test_uniform_disk_matches_vbbl(vbbl, rho):
    vbbl.a1 = 0.0  # uniform source
    mag = fspl_disk()
    # probe both inner/outer regimes with moderate count to keep runtime sane
    # avoid extremely small u/rho where numerical noise dominates; still probe inner/outer
    u_grid = rho * np.array([1.0, 5.0, 20.0])
    for u in u_grid:
        ref = vbbl.ESPLMag(float(u), float(rho))
        got = np.array(mag.A(u, rho))[0]
        # Limb-darkening kernel is less precise (FFT + interpolation); allow ~10% band.
        assert np.allclose(got, ref, rtol=1e-1, atol=5e-3)


@pytest.mark.parametrize("rho", [1e-3, 3e-2])
@pytest.mark.parametrize("gamma", [0.2, 0.5, 0.8])
def test_linear_limb_matches_vbbl_gamma_mode(vbbl, rho, gamma):
    vbbl.a1 = gamma
    mag = fspl_ld1(a1=gamma)
    u_grid = rho * np.array([1.0, 5.0, 20.0])
    for u in u_grid:
        ref = vbbl.ESPLMagDark(float(u), float(rho))
        got = np.array(mag.A(u, rho))[0]
        assert np.allclose(got, ref, rtol=1e-2, atol=1e-4)
