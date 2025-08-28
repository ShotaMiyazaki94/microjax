import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp
import pytest
from tests.utils.gpu import gpu_tests_enabled, has_cuda, is_a100

from microjax.point_source import _images_point_source
from microjax.multipole import _mag_hexadecapole
from microjax.inverse_ray.lightcurve import mag_binary


def make_trajectory(u0, tE, t0, alpha, n=32, span=3.0):
    t = t0 + jnp.linspace(-span * tE, span * tE, n)
    tau = (t - t0) / tE
    y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
    y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    return t, w_points


def test_far_field_uses_multipole_matches_internal():
    # Choose far from caustics: large |w| and tiny q
    s, q, rho = 1.2, 1e-3, 1e-3
    a = 0.5 * s
    e1 = q / (1.0 + q)
    x_cm = a * (1.0 - q) / (1.0 + q)
    # Trajectory points far away
    _, w = make_trajectory(u0=5.0, tE=10.0, t0=0.0, alpha=jnp.deg2rad(30), n=16, span=1.0)
    w_shift = w - x_cm
    z, z_mask = _images_point_source(w_shift, nlenses=2, a=a, e1=e1)
    mu_multi, _ = _mag_hexadecapole(z, z_mask, rho, nlenses=2, a=a, e1=e1, s=s, q=q)

    mags = mag_binary(
        w,
        rho,
        s=s,
        q=q,
        r_resolution=64,
        th_resolution=64,
        Nlimb=64,
        bins_r=16,
        bins_th=32,
        margin_r=0.5,
        margin_th=0.5,
        MAX_FULL_CALLS=0,
        chunk_size=16,
    )
    # With MAX_FULL_CALLS=0, all points should use multipole path
    assert np.allclose(np.array(mags), np.array(mu_multi), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "s,q,u0,tE,rho,alpha",
    [
        (1.0, 1e-2, 0.05, 20.0, 5e-3, np.deg2rad(15.0)),
    ],
)
@pytest.mark.gpu
def test_binary_lightcurve_matches_vbbl(s, q, u0, tE, rho, alpha):
    if not (gpu_tests_enabled() and has_cuda() and is_a100()):
        pytest.skip("GPU/A100 not available or MICROJAX_GPU_TESTS not enabled")
    VB = pytest.importorskip("VBBinaryLensing")
    VBBL = VB.VBBinaryLensing()
    VBBL.a1 = 0.0
    VBBL.RelTol = 1e-5

    t0 = 0.0
    npts = 50
    t, w = make_trajectory(u0=u0, tE=tE, t0=t0, alpha=alpha, n=npts, span=2.0)
    params_vb = [jnp.log(s), jnp.log(q), u0, alpha - jnp.pi, jnp.log(rho), jnp.log(tE), t0]
    mag_vb, _, _ = jnp.array(VBBL.BinaryLightCurve(params_vb, t))

    mags = mag_binary(
        w,
        rho,
        s=s,
        q=q,
        r_resolution=128,
        th_resolution=128,
        Nlimb=128,
        bins_r=24,
        bins_th=48,
        margin_r=0.5,
        margin_th=0.5,
        MAX_FULL_CALLS=200,
        chunk_size=25,
    )

    diff = np.array(mags) - np.array(mag_vb)
    # Allow a modest tolerance due to discretization and algorithmic differences
    assert np.max(np.abs(diff)) < 2e-3
