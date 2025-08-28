import numpy as np
import jax.numpy as jnp
import pytest

pytest.importorskip("VBMicrolensing")

from microjax.trajectory.parallax import (
    peri_vernal,
    set_parallax,
    compute_parallax,
)


def compute_microjax_pspl_parallax(t, u0, t0, tE, piEN, piEE, RA, Dec):
    tref = t0
    tperi, tvernal = peri_vernal(tref)
    parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)
    dtn, dum = compute_parallax(t, piEN, piEE, parallax_params)
    tau = (t - t0) / tE
    um = u0 + dum
    tm = tau + dtn
    u2 = um**2 + tm**2
    u = jnp.sqrt(u2)
    A = (u2 + 2.0) / (u * jnp.sqrt(u2 + 4.0))
    return A, tm, um


@pytest.mark.parametrize(
    "coords,u0,tE,t0,piEN,piEE",
    [
        ("17:45:40 -29:00:28", 0.01, 30.0, 8000.0, 0.1, 0.1),
        ("17:45:40 -29:00:28", 0.30, 20.0, 7500.0, -0.2, 0.05),
    ],
)
def test_pspl_parallax_matches_vbbl(coords, u0, tE, t0, piEN, piEE):
    import VBMicrolensing
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg))
    RA = c.ra.deg
    Dec = c.dec.deg

    t = t0 + np.linspace(-5.0 * tE, 5.0 * tE, 200)
    log_tE = np.log(tE)

    # VBBinaryLensing via VBMicrolensing wrapper
    VBM = VBMicrolensing.VBMicrolensing()
    VBM.parallaxsystem = 1
    VBM.t0_par_fixed = 1
    VBM.t0_par = t0
    VBM.SetObjectCoordinates(coords)
    VBM.RelTol = 1e-04
    VBM.Tol = 1e-04
    param = [u0, log_tE, t0, piEN, piEE]
    mag_vbbl, x, y = VBM.PSPLLightCurveParallax(param, t)

    # microJAX implementation
    mag_jax, tm, um = compute_microjax_pspl_parallax(t, u0, t0, tE, piEN, piEE, RA, Dec)

    # Compare magnifications
    diff_mag = np.array(mag_vbbl) - np.array(mag_jax)
    assert np.max(np.abs(diff_mag)) < 5e-4

    # Compare source-plane trajectories (sign conventions differ; use -x, -y per module __main__ demo)
    x_vbbl, y_vbbl = -np.array(x), -np.array(y)
    diff_pos = np.sqrt((x_vbbl - np.array(tm)) ** 2 + (y_vbbl - np.array(um)) ** 2)
    # Position paths can differ slightly in convention/numerics; allow small tolerance
    assert np.max(diff_pos) < 2e-3
