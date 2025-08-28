import numpy as np
import jax.numpy as jnp
import pytest
from tests.utils.gpu import gpu_tests_enabled, has_cuda, is_a100

from microjax.inverse_ray.lightcurve import mag_triple


def make_trajectory(u0, tE, t0, alpha, n=24, span=2.0):
    t = t0 + jnp.linspace(-span * tE, span * tE, n)
    tau = (t - t0) / tE
    y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
    y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    return w_points


@pytest.mark.gpu
def test_triple_shapes_and_finiteness():
    if not (gpu_tests_enabled() and has_cuda() and is_a100()):
        pytest.skip("GPU/A100 not available or MICROJAX_GPU_TESTS not enabled")
    s, q, q3 = 0.9, 0.2, 0.1
    rho = 0.01
    alpha = jnp.deg2rad(20.0)
    w = make_trajectory(u0=0.05, tE=10.0, t0=0.0, alpha=alpha, n=24, span=1.0)
    mags = mag_triple(
        w,
        rho,
        s=s,
        q=q,
        q3=q3,
        r_resolution=64,
        th_resolution=64,
        Nlimb=64,
        bins_r=16,
        bins_th=32,
        margin_r=0.5,
        margin_th=0.5,
        MAX_FULL_CALLS=200,
        chunk_size=24,
    )
    arr = np.array(mags)
    assert arr.shape == (w.shape[0],)
    assert np.all(np.isfinite(arr))
