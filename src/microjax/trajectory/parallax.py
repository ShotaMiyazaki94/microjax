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
from importlib import resources

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

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


# ---------------------------------------------------------------------------
# Ephemeris-based parallax (jacscanomaly-compatible)
# ---------------------------------------------------------------------------

ARCSEC_TO_RAD = jnp.deg2rad(1.0 / 3600.0)
AU_C_DAY = 0.005775518331436995  # AU/c in days


def load_horizons_vectors_file(path: str) -> np.ndarray:
    """
    Read a Horizons 'GEOMETRIC cartesian states' table (CSV-like) and return
    a numeric array with columns:
        [t_jdtdb, x, y, z, vx, vy, vz]
    Units: t in days (JD TDB), position in AU, velocity in AU/day.

    Expected Horizons line format (example):
    2451544.500000000, A.D. 2000-Jan-01 00:00:00.0000, -1.7E-01, 8.8E-01, ... , RR,

    Notes:
    - Skips everything outside $$SOE ... $$EOE.
    - Ignores the Calendar Date column.
    - Ignores LT/RG/RR.
    """
    rows = []
    in_block = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not in_block:
                if s.startswith("$$SOE"):
                    in_block = True
                continue
            else:
                if s.startswith("$$EOE"):
                    break
                if not s or s.startswith("*"):
                    continue

                parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                if len(parts) < 8:
                    continue

                try:
                    t = float(parts[0])
                    x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                    vx = float(parts[5]); vy = float(parts[6]); vz = float(parts[7])
                except ValueError:
                    continue

                rows.append((t, x, y, z, vx, vy, vz))

    if not rows:
        raise ValueError("No ephemeris rows parsed. Check file format and $$SOE/$$EOE markers.")
    return np.asarray(rows, dtype=np.float64)


@jax.tree_util.register_pytree_node_class
class HeliocentricEphemeris:
    """Uniform ephemeris container (t must be uniform grid for interp_uniform_linear)."""
    def __init__(self, t: jnp.ndarray, r: jnp.ndarray, v: jnp.ndarray):
        self.t = t
        self.r = r
        self.v = v

    def tree_flatten(self):
        return (self.t, self.r, self.v), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @staticmethod
    def from_horizons_vectors_table(table_np: np.ndarray) -> "HeliocentricEphemeris":
        tab = jnp.asarray(table_np)
        t = tab[:, 0]
        r = tab[:, 1:4]
        v = tab[:, 4:7]
        order = jnp.argsort(t)
        return HeliocentricEphemeris(t[order], r[order], v[order])


def interp_uniform_linear(xq, x0, dt, y):
    """Fast linear interpolation on a uniform grid (JAX friendly)."""
    xq = jnp.atleast_1d(xq)
    u = (xq - x0) / dt
    i0 = jnp.floor(u).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, y.shape[0] - 2)
    w = u - i0.astype(u.dtype)
    y0 = y[i0]
    y1 = y[i0 + 1]
    return y0 + (y1 - y0) * (w[:, None] if y.ndim == 2 else w)


def get_north_east(RA_deg, Dec_deg):
    lam = jnp.deg2rad(RA_deg)
    bet = jnp.deg2rad(Dec_deg)

    pole = jnp.array([0.0, 0.0, 1.0], dtype=lam.dtype)
    los = jnp.array(
        [jnp.cos(lam) * jnp.cos(bet), jnp.sin(lam) * jnp.cos(bet), jnp.sin(bet)],
        dtype=lam.dtype,
    )
    east = jnp.cross(pole, los)
    east = east / jnp.linalg.norm(east)
    north = jnp.cross(los, east)
    return north, east


def event_unit_vector(RA_deg, Dec_deg, dtype=jnp.float64):
    ra = jnp.deg2rad(jnp.asarray(RA_deg, dtype=dtype))
    dec = jnp.deg2rad(jnp.asarray(Dec_deg, dtype=dtype))
    cd, sd = jnp.cos(dec), jnp.sin(dec)
    ca, sa = jnp.cos(ra), jnp.sin(ra)
    return jnp.array([cd * ca, cd * sa, sd], dtype=dtype)


def light_time_corrected_time(t, t0, dt, rv, n_hat, au_c_day: float = AU_C_DAY, n_iter: int = 5):
    """Iteratively solve for emission time accounting for light travel."""
    t = jnp.asarray(t)
    t_emit = t

    def body(_, t_emit_curr):
        rv_curr = interp_uniform_linear(t_emit_curr, t0, dt, rv)
        r_curr = rv_curr[..., :3]
        lt = jnp.sum(r_curr * n_hat, axis=-1) * au_c_day
        return t - lt

    return lax.fori_loop(0, n_iter, body, t_emit)


@jax.tree_util.register_pytree_node_class
class EarthOrbitalParallaxProjector:
    def __init__(self, eph: HeliocentricEphemeris, RA_deg, Dec_deg, tref, *,
                 use_HJD: bool = True, light_time_iters: int = 5, au_c_day: float = AU_C_DAY):
        dtype = eph.t.dtype
        self.t0 = eph.t[0]
        self.dt = eph.t[1] - eph.t[0]
        self.tref = jnp.asarray(tref, dtype=dtype)

        self.use_HJD = bool(use_HJD)
        self.light_time_iters = int(light_time_iters)
        self.au_c_day = jnp.asarray(au_c_day, dtype=dtype)

        self.sky_north, self.sky_east = get_north_east(RA_deg, Dec_deg)
        self.n_hat = event_unit_vector(RA_deg, Dec_deg, dtype=dtype)

        self.rv = jnp.concatenate([eph.r, eph.v], axis=-1)

        if self.use_HJD:
            tref_eval = light_time_corrected_time(
                self.tref[None], self.t0, self.dt, self.rv, self.n_hat,
                au_c_day=self.au_c_day, n_iter=self.light_time_iters
            )[0]
        else:
            tref_eval = self.tref

        rv_ref = interp_uniform_linear(tref_eval[None], self.t0, self.dt, self.rv)[0]
        r_ref, v_ref = rv_ref[:3], rv_ref[3:]

        self.E_ref = -jnp.stack([r_ref @ self.sky_east, r_ref @ self.sky_north])
        self.V_ref = -jnp.stack([v_ref @ self.sky_east, v_ref @ self.sky_north])

    def tree_flatten(self):
        children = (
            self.t0, self.dt, self.tref, self.au_c_day,
            self.sky_north, self.sky_east, self.n_hat,
            self.rv, self.E_ref, self.V_ref
        )
        aux = (self.use_HJD, self.light_time_iters)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (
            obj.t0, obj.dt, obj.tref, obj.au_c_day,
            obj.sky_north, obj.sky_east, obj.n_hat,
            obj.rv, obj.E_ref, obj.V_ref
        ) = children
        (obj.use_HJD, obj.light_time_iters) = aux
        return obj


def earth_orbital_parallax_offsets(t, piEN, piEE, P: EarthOrbitalParallaxProjector):
    """Ephemeris-based ``d_tau`` and ``d_beta`` offsets (JAX differentiable)."""
    t = jnp.asarray(t, dtype=P.tref.dtype)

    if P.use_HJD:
        t_eval = light_time_corrected_time(
            t, P.t0, P.dt, P.rv, P.n_hat,
            au_c_day=P.au_c_day, n_iter=P.light_time_iters
        )
    else:
        t_eval = t

    rv_t = interp_uniform_linear(t_eval, P.t0, P.dt, P.rv)
    r_t = rv_t[:, :3]

    E_t = -jnp.stack([r_t @ P.sky_east, r_t @ P.sky_north], axis=-1)
    ds = -((P.E_ref[None] - E_t) + P.V_ref[None] * (t - P.tref)[:, None])

    d_tau = piEN * ds[:, 1] + piEE * ds[:, 0]
    d_beta = -piEE * ds[:, 1] + piEN * ds[:, 0]
    return d_tau, d_beta


earth_orbital_parallax_offsets_jit = jax.jit(earth_orbital_parallax_offsets)


def load_builtin_earth_ephemeris() -> HeliocentricEphemeris:
    """Load bundled Earth ephemeris table."""
    try:
        path = resources.files("microjax.data").joinpath("earth_orbital_parallax_table.txt")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError("Bundled Horizons ephemeris not found.") from exc
    table = load_horizons_vectors_file(path)
    return HeliocentricEphemeris.from_horizons_vectors_table(table)


def set_parallax_ephem(
    tref: float,
    RA: float,
    Dec: float,
    *,
    eph: HeliocentricEphemeris | None = None,
    use_HJD: bool = True,
    light_time_iters: int = 5,
) -> EarthOrbitalParallaxProjector:
    """Convenience wrapper mirroring set_parallax but using ephemeris."""
    if eph is None:
        eph = load_builtin_earth_ephemeris()
    tref_abs = tref + 2_450_000.0
    return EarthOrbitalParallaxProjector(
        eph,
        RA,
        Dec,
        tref_abs,
        use_HJD=use_HJD,
        light_time_iters=light_time_iters,
    )


def compute_parallax_ephem(
    t: Union[float, Array],
    piEN: float,
    piEE: float,
    projector: EarthOrbitalParallaxProjector,
    *,
    times_are_absolute: bool = False,
) -> Tuple[Array, Array]:
    """Ephemeris-based parallax offsets with microjax-style signature."""
    t_eval = jnp.asarray(t, dtype=projector.tref.dtype)
    if not times_are_absolute:
        t_eval = t_eval + 2_450_000.0
    d_tau, d_beta = earth_orbital_parallax_offsets_jit(t_eval, piEN, piEE, projector)
    return d_tau, d_beta
