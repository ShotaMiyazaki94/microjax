import os
import numpy as np
import pytest

from microjax.fastlens.mag_fft_jax import fspl_disk, fspl_ld1


@pytest.fixture(scope="module")
def vbbl():
    vb = pytest.importorskip("VBBinaryLensing")
    V = vb.VBBinaryLensing()
    table_path = vb.__file__.replace("__init__.py", "data/ESPL.tbl")
    V.LoadESPLTable(table_path)
    return V


# Fast grid (always run): modest coverage to keep runtime low.
RHO_FAST = np.logspace(-3, 2, 4)      # 1e-3 .. 1e2, 4 pts
UOVER_FAST = np.logspace(-2, 2, 9)    # 0.01 .. 100, 9 pts

# Slow grid (full coverage): only when FASTLENS_SLOW=1
RHO_SLOW = np.logspace(-3, 2, 8)      # 1e-3 .. 1e2, 8 pts
UOVER_SLOW = np.logspace(-2, 2, 25)   # 0.01 .. 100, 25 pts
RUN_SLOW = os.getenv("FASTLENS_SLOW") == "1"


def _max_rel_err_disk(vbbl, rhos, u_over, n_fft=2048, n_pad=2048):
    mag = fspl_disk(fft_logumin=-6, fft_logumax=3, N_fft=n_fft, normalize_sk=True)
    errs = []
    for rho in rhos:
        for r in u_over:
            u = rho * r
            ref = vbbl.ESPLMag(float(u), float(rho))
            got = float(np.array(mag.A(u, rho))[0])
            errs.append(abs(got - ref) / ref)
    return max(errs)


def _max_rel_err_limb(vbbl, a1, rhos, u_over, n_fft=2048, n_pad=2048):
    mag = fspl_ld1(a1=a1, fft_logumin=-6, fft_logumax=3, N_fft=n_fft, normalize_sk=True)
    vbbl.a1 = a1
    errs = []
    for rho in rhos:
        for r in u_over:
            u = rho * r
            ref = vbbl.ESPLMagDark(float(u), float(rho))
            got = float(np.array(mag.A(u, rho))[0])
            errs.append(abs(got - ref) / ref)
    return max(errs)


def test_disk_vs_vbbl_fast_grid(vbbl):
    max_err = _max_rel_err_disk(vbbl, RHO_FAST, UOVER_FAST, n_fft=2048)
    assert max_err < 5e-3  # 0.5%


@pytest.mark.parametrize(
    "a1, tol",
    [
        (0.2, 0.012),  # ~1% observed; small margin
        (0.5, 0.02),   # ~1.6% observed
        (0.8, 0.013),  # ~1.1% observed
    ],
)
def test_limb_vs_vbbl_fast_grid(vbbl, a1, tol):
    max_err = _max_rel_err_limb(vbbl, a1, RHO_FAST, UOVER_FAST, n_fft=2048)
    assert max_err < tol


@pytest.mark.skipif(not RUN_SLOW, reason="set FASTLENS_SLOW=1 to run slow wide-grid validation")
def test_disk_vs_vbbl_wide_grid(vbbl):
    max_err = _max_rel_err_disk(vbbl, RHO_SLOW, UOVER_SLOW, n_fft=2048)
    assert max_err < 2e-3  # 0.2%


@pytest.mark.skipif(not RUN_SLOW, reason="set FASTLENS_SLOW=1 to run slow wide-grid validation")
@pytest.mark.parametrize(
    "a1, tol",
    [
        (0.2, 0.011),
        (0.5, 0.018),
        (0.8, 0.012),
    ],
)
def test_limb_vs_vbbl_wide_grid(vbbl, a1, tol):
    max_err = _max_rel_err_limb(vbbl, a1, RHO_SLOW, UOVER_SLOW, n_fft=2048)
    assert max_err < tol
