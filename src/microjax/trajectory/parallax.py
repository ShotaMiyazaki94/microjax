"""Microlensing annual parallax utilities built on JAX.

This module provides helpers to compute Earth's projected position and the
resulting annual parallax corrections for microlensing trajectories in an
equatorial (ICRS) frame. Computations are JAX-friendly (autodiff/JIT) and use
lightweight fixed-iteration numerical methods where needed.

Functions
---------
- ``peri_vernal(tref)``: Select perihelion and vernal-equinox epochs nearest
  the reference time.
- ``getpsi(phi, ecc)``: Solve Kepler's equation for the eccentric anomaly via
  a fixed-iteration Newton method.
- ``prepare_projection_basis(rotaxis_deg, psi_offset, RA, Dec)``: Build the
  orbital→equatorial rotation and sky-plane (north/east) basis vectors.
- ``project_earth_position(t, tperi, period, ecc, R, north, east)``: Project
  Earth's Sun-centered orbit position onto the target tangent plane.
- ``set_parallax(tref, tperi, tvernal, RA, Dec, ...)``: Precompute quantities
  (position, local velocity, bases) around a reference epoch.
- ``compute_parallax(t, piEN, piEE, parallax_params)``: Compute the parallax
  offsets to add to dimensionless time (``tau``) and impact parameter (``u``).

Conventions
-----------
- Times are in JD-2450000 unless stated otherwise.
- Right ascension and declination are in degrees (ICRS).
- Tangent-plane basis is orthonormal and orthogonal to the line-of-sight (LOS).
- Default Earth orbital constants: obliquity 23.44 deg, eccentricity 0.0167,
  sidereal year 365.25636 days.

Sign Conventions
----------------
- Basis construction is right-handed: ``east = z_eq × los`` and
  ``north = los × east`` where ``z_eq`` is the equatorial north pole.
- Positive ``east`` increases right ascension; positive ``north`` increases
  declination on the sky.
- Parallax offsets apply as ``tm = tau + dtn`` and ``um = u0 + dum`` where
  ``tau = (t - t0)/tE`` and ``u0`` is the impact parameter.

Example
-------
>>> tperi, tvernal = peri_vernal(8000.0)
>>> params = set_parallax(8000.0, tperi, tvernal, RA=266.4168, Dec=-29.0078)
>>> dtn, dum = compute_parallax(8000.0 + jnp.linspace(-50, 50, 100), 0.1, 0.1, params)

References
----------
- Meeus, J., Astronomical Algorithms, 2nd ed., Willmann–Bell.
- Seidelmann, P. K. (ed.), Explanatory Supplement to the Astronomical Almanac.
- Gould, A. (2004), Resolution of the Microlens Parallax Degeneracy, ApJ, 606, 319.
"""

from typing import Tuple, Union

import jax.numpy as jnp

# Lightweight array alias for readability (consistent with inverse_ray style)
Array = jnp.ndarray

def peri_vernal(tref: Union[float, Array]) -> Tuple[Array, Array]:
    """Return perihelion and vernal-equinox times closest to ``tref``.

    This utility selects, from pre-tabulated epochs, the perihelion time and
    the vernal equinox time that are closest to the provided reference time.

    The function accepts both absolute Julian Date (JD) and JD-2450000. If the
    input is larger than ``2_450_000``, it is internally shifted by subtracting
    ``2_450_000`` so it can be compared against the tables below, which are in
    JD-2450000.

    Parameters
    ----------
    tref : float or array-like
        Reference time(s) in JD or JD-2450000.

    Returns
    -------
    tperi : float
        Perihelion time (JD-2450000), closest to ``tref``.
    tvernal : float
        Vernal equinox time (JD-2450000), closest to ``tref``.

    Notes
    -----
    - The returned values are selected by nearest-neighbor search in the
      provided tables and are not interpolated.
    - If ``tref`` is an array, the nearest entry is found based on the array
      broadcasting rules of JAX, and a single pair is returned as JAX scalars.

    Examples
    --------
    >>> tperi, tvernal = peri_vernal(2458000.0)
    >>> float(tperi) > 0 and float(tvernal) > 0
    True
    """
    tref = jnp.where(tref > 2_450_000.0, tref - 2_450_000.0, tref)
    peris = jnp.array([
        1546.70833, 1913.87500, 2277.08333, 2643.70833, 3009.25000, 3372.54167,
        3740.16667, 4104.33333, 4468.50000, 4836.12500, 5199.50000, 5565.29167,
        5931.54167, 6294.70833, 6662.00000, 7026.79167, 7390.45833, 7758.08333,
        8121.75000, 8486.70833, 8853.83333
    ])
    vernals = jnp.array([
        1623.81597, 1989.06319, 2354.30278, 2719.54167, 3084.78403, 3450.02292,
        3815.26806, 4180.50486, 4545.74167, 4910.98889, 5276.23056, 5641.47292,
        6006.71806, 6371.95972, 6737.20625, 7102.44792, 7467.68750, 7832.93611,
        8198.17708, 8563.41528, 8928.65903
    ])
    dperi = jnp.abs(peris - tref)
    imin = jnp.argmin(dperi)
    return peris[imin], vernals[imin]

def getpsi(phi: Union[float, Array], ecc: float) -> Array:
    """Solve Kepler's equation for the eccentric anomaly ``psi``.

    The equation solved is ``psi - e * sin(psi) = phi`` using a fixed small
    number of Newton–Raphson iterations (5) with an empirical initial guess.
    This routine is differentiable under JAX and works with scalars or arrays.

    Parameters
    ----------
    phi : float or jax.Array
        Mean anomaly in radians. May be scalar or array-like.
    ecc : float
        Orbital eccentricity, ``0 <= ecc < 1``.

    Returns
    -------
    psi : jax.Array
        Eccentric anomaly in radians, with the same broadcasted shape as
        ``phi``.

    Notes
    -----
    - The initial guess is ``phi + sign(sin(phi)) * 0.85 * ecc`` which works
      well for moderate eccentricities without branching.
    - The iteration count is fixed to keep control-flow JIT friendly.
    """
    psi = phi + jnp.sign(jnp.sin(phi)) * 0.85 * ecc # empirical init
    for _ in range(5):
        f = psi - ecc * jnp.sin(psi) - phi
        f_prime = 1.0 - ecc * jnp.cos(psi)
        psi -= f / f_prime
    return psi

def prepare_projection_basis(rotaxis_deg: float, psi_offset: float, RA: float, Dec: float) -> Tuple[Array, Array, Array]:
    """Build rotation and on-sky projection bases.

    Constructs the rotation matrix that maps the Sun–Earth orbital plane
    coordinates to the equatorial frame and returns the orthonormal basis
    vectors on the sky plane at the target direction: ``north`` and ``east``.

    Parameters
    ----------
    rotaxis_deg : float
        Obliquity of the ecliptic (tilt between equatorial and ecliptic
        planes) in degrees.
    psi_offset : float
        Eccentric-anomaly angle between perihelion and the vernal equinox in
        radians. This aligns the orbital x-axis with the vernal direction.
    RA : float
        Right ascension of the target in degrees (ICRS).
    Dec : float
        Declination of the target in degrees (ICRS).

    Returns
    -------
    R : jax.Array, shape (3, 3)
        Rotation matrix from orbital coordinates to equatorial coordinates.
    north : jax.Array, shape (3,)
        Unit vector pointing to celestial north on the tangent plane at the
        target position.
    east : jax.Array, shape (3,)
        Unit vector pointing to celestial east on the tangent plane.

    Notes
    -----
    The line-of-sight unit vector is derived from (RA, Dec). The returned
    ``east`` and ``north`` are orthonormal and orthogonal to the line of sight.
    """
    # orbital frame -> ecliptic frame
    # psi_offset is an angle from perihelion to vernal equinox
    # (rotate about the z-axis to align with the x-axis with the vernal)
    Rz = jnp.array([
        [jnp.cos(-psi_offset), -jnp.sin(-psi_offset), 0],
        [jnp.sin(-psi_offset),  jnp.cos(-psi_offset), 0],
        [0,                   0,                    1]
    ])
    # ecliptic -> equatorial (rotate about x-axis)
    # rotaxis is an inclunation angle from equational to ecliptic frame
    rotaxis = jnp.deg2rad(rotaxis_deg)
    Rx = jnp.array([
        [1, 0,               0],
        [0, jnp.cos(rotaxis), -jnp.sin(rotaxis)],
        [0, jnp.sin(rotaxis),  jnp.cos(rotaxis)]
    ])
    R = Rx @ Rz

    alpha, delta = jnp.deg2rad(RA), jnp.deg2rad(Dec)
    los = jnp.array([
        jnp.cos(alpha) * jnp.cos(delta),
        jnp.sin(alpha) * jnp.cos(delta),
        jnp.sin(delta)
    ])
    z_eq = jnp.array([0.0, 0.0, 1.0])
    east = jnp.cross(z_eq, los)
    east /= jnp.linalg.norm(east)
    north = jnp.cross(los, east)
    north /= jnp.linalg.norm(north)

    return R, north, east

def project_earth_position(
    t: Union[float, Array],
    tperi: float,
    period: float,
    ecc: float,
    R: Array,
    north: Array,
    east: Array,
) -> Array:
    """Project Earth's position onto the target's tangent plane.

    Computes the Sun-centered position of Earth in its (elliptical) orbit at
    time ``t`` using the eccentric anomaly and projects it onto the sky-plane
    basis defined by ``north`` and ``east``.

    Parameters
    ----------
    t : float or jax.Array
        Observation time(s) in JD-2450000; scalar or 1D array.
    tperi : float
        Time of perihelion in JD-2450000.
    period : float
        Orbital period in days (sidereal year).
    ecc : float
        Orbital eccentricity, ``0 <= ecc < 1``.
    R : jax.Array, shape (3, 3)
        Rotation matrix from orbital to equatorial frame.
    north : jax.Array, shape (3,)
        North unit vector on the tangent plane.
    east : jax.Array, shape (3,)
        East unit vector on the tangent plane.

    Returns
    -------
    q : jax.Array, shape (2, N)
        Stacked projected coordinates ``[q_north, q_east]`` where ``N`` is the
        number of time samples (``N = 1`` for scalar ``t``).

    Notes
    -----
    The orbital coordinates are computed in the orbital frame with x-axis
    toward perihelion, then rotated to the equatorial frame and projected onto
    the tangent-plane basis.
    """
    t = jnp.atleast_1d(t)
    N = t.shape[0]
    phi = 2.0 * jnp.pi * (t - tperi) / period
    psi = getpsi(phi, ecc)

    # Sun-centered position, x-axis aligning with perihelion, z-axis aligning with ecliptic north 
    x_orb = jnp.cos(psi) - ecc
    y_orb = jnp.sin(psi) * jnp.sqrt(1.0 - ecc**2)
    r_orb = jnp.array([x_orb, y_orb, jnp.zeros(N)])
    r_eq = R @ r_orb # (3, N) shape

    q_north = jnp.dot(north, r_eq)
    q_east = jnp.dot(east, r_eq)
    return jnp.array([q_north, q_east])

def set_parallax(
    tref: float,
    tperi: float,
    tvernal: float,
    RA: float,
    Dec: float,
    rotaxis_deg: float = 23.44,
    ecc: float = 0.0167,
    period: float = 365.25636,
    dt: float = 0.1,
) -> Tuple[Array, Array, Array, Array, Array, float, float, float, float]:
    """Precompute parallax parameters at a reference epoch.

    Precomputes quantities needed to evaluate the microlensing annual parallax
    signal around ``tref``. This includes the on-sky Earth position at ``tref``,
    a local linear velocity approximation (finite-difference over ``dt``), and
    the projection/rotation bases.

    If either ``tperi`` or ``tvernal`` is passed as 0, both values are
    automatically inferred using :func:`peri_vernal` at ``tref``.

    Parameters
    ----------
    tref : float
        Reference time in JD-2450000 at which the linearization is anchored.
    tperi : float
        Perihelion time in JD-2450000, or 0 to auto-select.
    tvernal : float
        Vernal equinox time in JD-2450000, or 0 to auto-select.
    RA : float
        Target right ascension in degrees (ICRS).
    Dec : float
        Target declination in degrees (ICRS).
    rotaxis_deg : float, optional
        Obliquity of the ecliptic in degrees. Default is 23.44.
    ecc : float, optional
        Orbital eccentricity of Earth. Default is 0.0167.
    period : float, optional
        Orbital period (sidereal year) in days. Default is 365.25636.
    dt : float, optional
        Time step (days) used to compute the finite-difference velocity.

    Returns
    -------
    parallax_params : tuple
        Tuple ``(qne0, vne0, R, north, east, tref, tperi, period, ecc)`` where
        each element is:
        - ``qne0``: jax.Array, shape (2,), Earth position [north, east] at ``tref``.
        - ``vne0``: jax.Array, shape (2,), approximate velocity d[q_north, q_east]/dt at ``tref``.
        - ``R``: jax.Array, shape (3, 3), rotation matrix orbital→equatorial.
        - ``north``: jax.Array, shape (3,), north basis vector.
        - ``east``: jax.Array, shape (3,), east basis vector.
        - ``tref``: float, the reference epoch.
        - ``tperi``: float, perihelion epoch used.
        - ``period``: float, orbital period used.
        - ``ecc``: float, eccentricity used.

    Notes
    -----
    The velocity is computed with a symmetric finite difference of width
    ``2*dt`` to reduce truncation error and preserve JAX differentiability.
    """
    info_0 = peri_vernal(tref)
    info = jnp.where(tperi * tvernal == 0,
                     jnp.array(info_0),
                     jnp.array([tperi, tvernal]))
    tperi, tvernal = info
    phi_offset = 2 * jnp.pi * (tvernal - tperi) / period
    psi_offset = getpsi(phi_offset, ecc)
    costh = (jnp.cos(psi_offset) - ecc) / (1 - ecc * jnp.cos(psi_offset))
    sinth = jnp.sqrt(1.0 - ecc**2) * jnp.sin(psi_offset) / (1 - ecc * jnp.cos(psi_offset))
    f_rot = jnp.mod(jnp.arctan2(sinth, costh), 2 * jnp.pi)
    R, north, east = prepare_projection_basis(rotaxis_deg, f_rot, RA, Dec)
    qne0 = project_earth_position(tref, tperi, period, ecc, R, north, east)
    qne1 = project_earth_position(tref - dt, tperi, period, ecc, R, north, east)
    qne2 = project_earth_position(tref + dt, tperi, period, ecc, R, north, east)
    vne0 = 0.5 * (qne2 - qne1) / dt
    parallax_params = (qne0, vne0, R, north, east, tref, tperi, period, ecc)
    return parallax_params

def compute_parallax(
    t: Union[float, Array],
    piEN: float,
    piEE: float,
    parallax_params: Tuple[Array, Array, Array, Array, Array, float, float, float, float],
) -> Tuple[Array, Array]:
    """Compute annual parallax offsets at times ``t``.

    Produces the parallax-induced shifts to microlensing trajectory parameters:
    ``dtn`` should be added to the dimensionless time coordinate ``tau``, and
    ``dum`` should be added to the impact parameter ``u`` (north–east frame).

    Parameters
    ----------
    t : float or jax.Array
        Time(s) in JD-2450000 at which to evaluate the parallax signal.
    piEN : float
        Parallax amplitude projected in the north direction.
    piEE : float
        Parallax amplitude projected in the east direction.
    parallax_params : tuple
        Output of :func:`set_parallax`.

    Returns
    -------
    dtn : jax.Array
        Offset to add to the dimensionless time coordinate(s) ``tau``; shape
        ``(N,)`` matching the number of time samples.
    dum : jax.Array
        Offset to add to the impact parameter coordinate(s) ``u``; shape
        ``(N,)``.

    Notes
    -----
    The mean linear motion around ``tref`` is removed using the precomputed
    velocity in ``parallax_params`` to isolate the purely annual parallax
    contribution.
    """
    qne0, vne0, R, north, east, tref, tperi, period, ecc = parallax_params
    qne = project_earth_position(t, tperi, period, ecc, R, north, east)
    dt_ref = t - tref
    # Vectorized form equivalent to the original comprehension over i in {0,1}
    qne_delta = qne - (qne0 + vne0 * dt_ref)
    dtn = piEN * qne_delta[0] + piEE * qne_delta[1]
    dum = piEN * qne_delta[1] - piEE * qne_delta[0]
    return dtn, dum


if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp
    import jax
    jax.config.update('jax_enable_x64', True)
    import matplotlib.pyplot as plt
    import VBMicrolensing

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    coords = "17:45:40 -29:00:28"
    c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg),)
    RA = c.ra.deg
    Dec = c.dec.deg

    u0 = 0.01
    log_tE = np.log(30.0)
    t0 = 8000.0 
    piEN = 0.3
    piEE = 0.3
    tE = np.exp(log_tE)
    t = t0 + np.linspace(-5.0*tE, 5.0*tE, 1000)

    # VBM
    VBM = VBMicrolensing.VBMicrolensing()
    VBM.parallaxsystem = 1
    VBM.t0_par_fixed = 1
    VBM.t0_par = t0
    VBM.SetObjectCoordinates(coords)
    VBM.RelTol = 1e-04
    VBM.Tol=1e-04
    param = [u0, log_tE, t0, piEN, piEE]
    mag_vbbl, x, y = VBM.PSPLLightCurveParallax(param, t)

    # MulensModel
    import MulensModel as mm
    params = dict()
    params['t_0'] = t0 + 2450000
    params['t_0_par'] = t0 + 2450000
    params['u_0'] = u0
    params['t_E'] = tE
    params['pi_E_N'] = piEN
    params['pi_E_E'] = piEE
    my_model = mm.Model(params, coords=coords)
    my_model.parallax(earth_orbital=True, satellite=False)
    mag_mulens = my_model.get_magnification(time=t+2450000,)

    # microJAX
    tref = t0
    tperi, tvernal = peri_vernal(tref)
    parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)
    dtn, dum = compute_parallax(t, piEN, piEE, parallax_params)
    tau = (t - t0)/tE
    um  = u0 + dum
    tm  = tau + dtn
    u2  = um**2 + tm**2
    u   = jnp.sqrt(u2)
    mag_jax = (u2 + 2.0) / (u * jnp.sqrt(u2 + 4.0))

    diff_vbbl_jax    = jnp.array(mag_vbbl) - jnp.array(mag_jax)
    diff_mulens_jax  = jnp.array(mag_mulens) - jnp.array(mag_jax)

    print("mag diff max VBM - MicroJAX:    ", jnp.max(diff_vbbl_jax))
    print("mag diff max Mulens - microJAX: ", jnp.max(diff_mulens_jax))

    trajectory_mulens = my_model.get_trajectory(times= t + 2450000.0)
    x_vbbl, y_vbbl = -jnp.array(x), -jnp.array(y)

    diff_vbbl_jax_xy = jnp.array([x_vbbl, y_vbbl]) - jnp.array([tm, um])
    diff_mulens_jax_xy = jnp.array([trajectory_mulens.x, trajectory_mulens.y]) - jnp.array([tm, um])
    print("diff pos max VBM - MicroJAX:    ", jnp.max(jnp.sqrt(diff_vbbl_jax_xy[0]**2 + diff_vbbl_jax_xy[1]**2)))
    print("diff pos max Mulens - microJAX: ", jnp.max(jnp.sqrt(diff_mulens_jax_xy[0]**2 + diff_mulens_jax_xy[1]**2)))
    exit(1)
