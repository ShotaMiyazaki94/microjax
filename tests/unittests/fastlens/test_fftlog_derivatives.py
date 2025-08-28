import numpy as np
import pytest

from microjax.fastlens.fftlog import fftlog as np_fftlog
from microjax.fastlens.fftlog_jax import fftlog as jx_fftlog


def make_input():
    x = np.logspace(-2, 2, 256)
    fx = np.exp(-x**2)
    return x, fx


@pytest.mark.parametrize("ell", [0, 1, 2])
def test_fftlog_derivative_variants_numpy(ell):
    x, fx = make_input()
    fl = np_fftlog(x, fx, nu=1.5, N_extrap_low=0, N_extrap_high=0, N_pad=64)
    # Basic calls shouldn't raise and should yield finite arrays
    for method in (fl.fftlog_dj, fl.fftlog_ddj, fl.fftlog_jsqr):
        y, Fy = method(ell)
        assert y.shape == Fy.shape
        assert np.all(np.diff(y) > 0)
        assert np.all(np.isfinite(Fy))


@pytest.mark.parametrize("ell", [0, 1, 2])
def test_fftlog_derivative_variants_jax_vs_numpy_close(ell):
    x, fx = make_input()
    np_fl = np_fftlog(x, fx, nu=1.5, N_extrap_low=0, N_extrap_high=0, N_pad=64)
    jx_fl = jx_fftlog(x, fx, nu=1.5, N_extrap_low=0, N_extrap_high=0, N_pad=64)

    pairs = [
        (np_fl.fftlog_dj, jx_fl.fftlog_dj),
        (np_fl.fftlog_ddj, jx_fl.fftlog_ddj),
        (np_fl.fftlog_jsqr, jx_fl.fftlog_jsqr),
    ]
    for np_m, jx_m in pairs:
        y_np, Fy_np = np_m(ell)
        y_jx, Fy_jx = jx_m(ell)
        assert np.allclose(y_np, y_jx, rtol=1e-6, atol=1e-8)
        sl = slice(10, -10)
        assert np.allclose(Fy_np[sl], np.array(Fy_jx)[sl], rtol=5e-3, atol=5e-4)

