import numpy as np
import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("MulensModel")

from microjax.trajectory.parallax import (
    EarthOrbitalParallaxProjector,
    compute_parallax,
    compute_parallax_ephem,
    load_builtin_earth_ephemeris,
    peri_vernal,
    set_parallax,
)


def microjax_kepler_trajectory(t, u0, t0, tE, piEN, piEE, RA, Dec):
    tperi, tvernal = peri_vernal(t0)
    parallax_params = set_parallax(t0, tperi, tvernal, RA, Dec)
    dtn, dum = compute_parallax(t, piEN, piEE, parallax_params)
    tau = (t - t0) / tE + dtn
    beta = u0 + dum
    return tau, beta


def microjax_ephem_trajectory(t, u0, t0, tE, piEN, piEE, RA, Dec):
    eph = load_builtin_earth_ephemeris()
    proj = EarthOrbitalParallaxProjector(eph, RA, Dec, t0 + 2_450_000.0, use_HJD=True)
    dtn, dum = compute_parallax_ephem(t, piEN, piEE, proj)
    tau = (t - t0) / tE + dtn
    beta = u0 + dum
    return tau, beta


@pytest.mark.parametrize(
    "coords,u0,tE,t0,piEN,piEE",
    [
        # pi_E magnitude ~1 to stress parallax
        ("17:45:40 -29:00:28", 0.01, 30.0, 8000.0, 0.7, -0.7),
        ("17:45:40 -29:00:28", 0.30, 20.0, 7500.0, -0.8, 0.6),
    ],
)
def test_pspl_parallax_matches_mulens(coords, u0, tE, t0, piEN, piEE):
    import MulensModel as mm
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg))
    RA = c.ra.deg
    Dec = c.dec.deg

    t = t0 + np.linspace(-5.0 * tE, 5.0 * tE, 200)

    # MulensModel reference (geocentric parallax)
    params = {
        "t_0": t0 + 2450000,
        "t_0_par": t0 + 2450000,
        "u_0": u0,
        "t_E": tE,
        "pi_E_N": piEN,
        "pi_E_E": piEE,
    }
    model = mm.Model(params, coords=coords)
    model.parallax(earth_orbital=True, satellite=False)
    traj = model.get_trajectory(times=t + 2450000)

    # microJAX (Kepler)
    tau_k, beta_k = microjax_kepler_trajectory(t, u0, t0, tE, piEN, piEE, RA, Dec)
    diff_k = np.sqrt((np.array(tau_k) - traj.x) ** 2 + (np.array(beta_k) - traj.y) ** 2)

    # microJAX (ephemeris)
    tau_e, beta_e = microjax_ephem_trajectory(t, u0, t0, tE, piEN, piEE, RA, Dec)
    diff_e = np.sqrt((np.array(tau_e) - traj.x) ** 2 + (np.array(beta_e) - traj.y) ** 2)

    # 合理的な精度閾値（Kepler ~1e-2, ephem ~1e-3）
    assert diff_k.max() < 1.1e-2
    assert diff_k.mean() < 3e-3

    assert diff_e.max() < 1e-3
    assert diff_e.mean() < 3e-4
