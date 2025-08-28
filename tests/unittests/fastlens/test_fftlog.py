import numpy as np
import pytest

from microjax.fastlens.fftlog import hankel as np_hankel
from microjax.fastlens.fftlog_jax import hankel as jax_hankel


def make_gaussian():
    # Log-spaced grid and Gaussian function
    x = np.logspace(-2, 2, 256)
    fx = np.exp(-x**2)
    return x, fx


@pytest.mark.parametrize("n", [0, 1])
def test_hankel_numpy_vs_jax_close(n):
    x, fx = make_gaussian()
    nu = 1.5
    # Prefer zero extrapolation and modest zero-padding to avoid edge NaNs
    np_h = np_hankel(x, fx, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=64)
    jx_h = jax_hankel(x, fx, nu=nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=64)

    y_np, Fy_np = np_h.hankel(n)
    y_jx, Fy_jx = jx_h.hankel(n)

    # Grids should match closely
    assert np.allclose(y_np, y_jx, rtol=1e-6, atol=1e-8)
    # Transforms should be close up to small numerical differences
    # Compare only on the central region to avoid extrap edge effects
    sl = slice(10, -10)
    assert np.allclose(Fy_np[sl], np.array(Fy_jx)[sl], rtol=5e-3, atol=5e-4)


def test_hankel_basic_properties():
    x = np.logspace(-2, 2, 256)
    fx = np.exp(-0.5 * x**2)
    h = np_hankel(x, fx, nu=1.5, N_extrap_low=0, N_extrap_high=0, N_pad=64)
    y, Fy = h.hankel(0)
    # Shapes align and y is increasing
    assert y.shape == Fy.shape
    assert np.all(np.diff(y) > 0)
    # Positive input should yield finite output
    assert np.all(np.isfinite(Fy))
