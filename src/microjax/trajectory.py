import jax.numpy as jnp

def peri_vernal(tref):
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

def getpsi(phi, ecc):
    """Solve Kepler's equation using Newton-Raphson method."""
    psi = phi + jnp.sign(jnp.sin(phi)) * 0.85 * ecc # empirical init
    for _ in range(5):
        f = psi - ecc * jnp.sin(psi) - phi
        f_prime = 1.0 - ecc * jnp.cos(psi)
        psi -= f / f_prime
    return psi

def prepare_projection_basis(rotaxis_deg, psi_offset, RA, Dec):
    """
    Precompute rotation and projection basis matrices used in sun projection calculation.

    Parameters:
        rotaxis_deg : float - obliquity of the ecliptic [deg]
        psi_offset : float - eccentric anomaly offset from perihelion to vernal point [rad]
        RA : float - right ascension of target [deg]
        Dec : float - declination of target [deg]

    Returns:
        R : (3,3) ndarray - total rotation matrix from orbital to equatorial frame
        north : (3,) ndarray - projection vector (north direction)
        east : (3,) ndarray - projection vector (east direction)
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

def project_earth_position(t, tperi, period, ecc, R, north, east):
    """
    Project position onto the tangent plane defined by (north, east),
    given precomputed rotation matrix R.

    Parameters:
        t : jnp.array or float - observation time [JD]
        tperi : float - time of perihelion [JD]
        period : float - orbital period [days]
        ecc : float - orbital eccentricity
        R : (3,3) ndarray - total rotation matrix from orbital to equatorial frame
        north : (3,) ndarray - unit vector defining projection north direction
        east : (3,) ndarray - unit vector defining projection east direction

    Returns:
        q_north, q_east : float - projected coordinates on the tangent plane
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

def set_parallax(tref, tperi, tvernal, RA, Dec,
                 rotaxis_deg=23.44, ecc=0.0167, period=365.25636, dt=0.1):
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

def compute_parallax(t, piEN, piEE, parallax_params):
    qne0, vne0, R, north, east, tref, tperi, period, ecc = parallax_params
    qne = project_earth_position(t, tperi, period, ecc, R, north, east)
    qne_delta = jnp.array([qne[i] - qne0[i] - vne0[i] * (t - tref) for i in range(2)])
    dtn = piEN * qne_delta[0] + piEE * qne_delta[1]
    dum = piEN * qne_delta[1] - piEE * qne_delta[0]
    return dtn, dum
