import numpy as np
import jax.numpy as jnp
import pytest

pytest.importorskip("VBBinaryLensing")

from microjax.trajectory.lom import circular_orbital_motion_3d
from microjax.trajectory.lom import elliptic_orbital_motion_3d


@pytest.fixture(scope="module")
def vbbl():
    import VBBinaryLensing as vb

    solver = vb.VBBinaryLensing()
    # The 3.7 light-curve wrappers otherwise inherit an effectively exact
    # default that can make even a few finite-source samples take minutes.
    solver.Tol = 1e-8
    return solver


def _vbbl_lightcurve_magnification(method, params, times):
    """Call either the current two-argument or legacy parallax-array API."""

    times_list = times.tolist()
    try:
        result = method(params, times_list)
    except TypeError:
        zeros = [0.0] * times.size
        result = method(params, times_list, zeros, zeros, zeros)

    values = np.asarray(result)
    # VBBinaryLensing 3.7 returns [magnification, y1, y2, separation].
    # Older wrappers returned the magnification vector directly.
    return values[0] if values.ndim == 2 else values


def _mags_from_microjax_orbital_state(
    vbbl,
    times,
    *,
    s0,
    q,
    u0,
    alpha0,
    rho,
    tE,
    t0,
    t0_par,
    w1,
    w2,
    w3,
):
    """Rebuild VBBL orbital magnification from microjax orbital state."""
    t = jnp.asarray(times, dtype=jnp.float64)
    s_t, alpha_t, _ = circular_orbital_motion_3d(
        t=t, s0=s0, alpha0=alpha0, w1=w1, w2=w2, w3=w3, tref=t0_par
    )

    tn = (t - t0) / tE
    u = jnp.full_like(tn, u0)
    y1 = -tn * jnp.cos(alpha_t) + u * jnp.sin(alpha_t)
    y2 = -u * jnp.cos(alpha_t) - tn * jnp.sin(alpha_t)

    return np.array(
        [
            vbbl.BinaryMag2(float(si), float(q), float(xi), float(yi), float(rho))
            for si, xi, yi in zip(np.array(s_t), np.array(y1), np.array(y2))
        ]
    )


def _times_uniform(t0: float, n_times: int = 1000) -> np.ndarray:
    """Uniform time grid over the full comparison window."""
    return np.linspace(t0 - 20.0, t0 + 20.0, n_times)


def _mags_from_microjax_kepler_state(
    vbbl,
    times,
    *,
    s0,
    q,
    u0,
    alpha0,
    rho,
    tE,
    t0,
    t0_par,
    w1,
    w2,
    w3,
    szs,
    ar,
):
    """Rebuild VBBL Keplerian magnification from microjax orbital state."""
    t = jnp.asarray(times, dtype=jnp.float64)
    s_t, alpha_t, _ = elliptic_orbital_motion_3d(
        t=t,
        s0=s0,
        alpha0=alpha0,
        w1=w1,
        w2=w2,
        w3=w3,
        szs=szs,
        ar=ar,
        tref=t0_par,
    )

    tn = (t - t0) / tE
    u = jnp.full_like(tn, u0)
    y1 = -tn * jnp.cos(alpha_t) + u * jnp.sin(alpha_t)
    y2 = -u * jnp.cos(alpha_t) - tn * jnp.sin(alpha_t)

    return np.array(
        [
            vbbl.BinaryMag2(float(si), float(q), float(xi), float(yi), float(rho))
            for si, xi, yi in zip(np.array(s_t), np.array(y1), np.array(y2))
        ]
    )


@pytest.mark.parametrize(
    "w1,w2,w3",
    [
        (1.0e-2, -2.0e-2, 3.0e-2),
        (-3.0e-2, 1.0e-2, -2.0e-2),  # exercises VBBL's internal w3 floor branch
    ],
)
def test_circular_orbital_motion_3d_matches_vbbl_orbital_lightcurve(vbbl, w1, w2, w3):
    s0 = 1.2
    q = 0.1
    u0 = 0.03
    alpha0 = 0.4
    rho = 1.0e-3
    tE = 30.0
    t0 = 5000.0
    t0_par = t0
    times = _times_uniform(t0)

    vbbl.t0_par = t0_par

    params = [
        np.log(s0),
        np.log(q),
        u0,
        alpha0,
        np.log(rho),
        np.log(tE),
        t0,
        0.0,  # pi_E parallel
        0.0,  # pi_E perpendicular
        w1,
        w2,
        w3,
    ]
    mags_vb = _vbbl_lightcurve_magnification(
        vbbl.BinaryLightCurveOrbital,
        params,
        times,
    )

    mags_mj = _mags_from_microjax_orbital_state(
        vbbl,
        times,
        s0=s0,
        q=q,
        u0=u0,
        alpha0=alpha0,
        rho=rho,
        tE=tE,
        t0=t0,
        t0_par=t0_par,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    np.testing.assert_allclose(mags_mj, mags_vb, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "w1,w2,w3,szs,ar",
    [
        (1.0e-2, -2.0e-2, 3.0e-2, 0.2, 0.85),
        (-8.0e-3, 1.1e-2, -6.0e-3, -0.15, 0.95),
    ],
)
def test_elliptic_orbital_motion_3d_matches_vbbl_kepler_lightcurve(
    vbbl, w1, w2, w3, szs, ar
):
    s0 = 1.2
    q = 0.1
    u0 = 0.03
    alpha0 = 0.4
    rho = 1.0e-3
    tE = 30.0
    t0 = 5000.0
    t0_par = t0
    times = _times_uniform(t0)

    vbbl.t0_par = t0_par

    params = [
        np.log(s0),
        np.log(q),
        u0,
        alpha0,
        np.log(rho),
        np.log(tE),
        t0,
        0.0,  # pi_E parallel
        0.0,  # pi_E perpendicular
        w1,
        w2,
        w3,
        szs,
        ar,
    ]
    mags_vb = _vbbl_lightcurve_magnification(
        vbbl.BinaryLightCurveKepler,
        params,
        times,
    )

    mags_mj = _mags_from_microjax_kepler_state(
        vbbl,
        times,
        s0=s0,
        q=q,
        u0=u0,
        alpha0=alpha0,
        rho=rho,
        tE=tE,
        t0=t0,
        t0_par=t0_par,
        w1=w1,
        w2=w2,
        w3=w3,
        szs=szs,
        ar=ar,
    )

    np.testing.assert_allclose(mags_mj, mags_vb, rtol=0.0, atol=1e-8)


def test_uniform_grid_has_1000_samples():
    t0 = 5000.0
    times = _times_uniform(t0)
    assert times.size == 1000
