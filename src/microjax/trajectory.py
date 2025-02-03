import jax.numpy as jnp
from functools import partial
from jax import jit

def dtn_dum_parallax(t, piEN, piEE, t_peri, qne0, vne0, xpos, ypos, north, east, tref=0.0, ecc=0.0167, period = 365.25636):
    """
    Returns:
        dtn: offsets in the tn direction
        dum: offsets in the um direction
    """ 
    qne = _get_sun_proj(t, t_peri, period, ecc, xpos, ypos, north, east)
    qne = jnp.array([qne[i] - qne0[i] - vne0[i]*(t - tref) for i in range(2)])
    dtn = piEN * qne[0] + piEE * qne[1]
    dum = piEN * qne[1] - piEE * qne[0]
    return dtn, dum

def _get_info(RA=0.0, Dec=0.0, tref=0.0,
             rotaxis=23.44, ecc=0.0167, period = 365.25636, 
             t_peri=0.0, t_vernal=0.0, dt=0.1):
    if t_peri*t_vernal == 0:
        tptmp, tvtmp = _peri_vernal(tref)
    t_peri = t_peri if t_peri > 0 else tptmp
    t_vernal  = t_vernal if t_vernal > 0 else tvtmp
    offset   = t_vernal - t_peri
    # Get the perihelion (x) direction and the corresponding y direction
    # in the equatorial coordinate system (where x = vernal)
    xpos, ypos = _get_xy_in_peri(rotaxis, offset, period, ecc)
    # Calculate East and North vector on the event sky
    north, east = _get_north_east(RA, Dec)
    # Calculate Earth projected velocity at reference time tref
    qne1 = _get_sun_proj(tref - dt, t_peri, period, ecc,
                         xpos, ypos, north, east)
    qne2 = _get_sun_proj(tref + dt, t_peri, period, ecc,
                         xpos, ypos, north, east)
    vne0 = 0.5 * (qne2 - qne1) / dt # AU / day, minus projected Earth velocity
    qne0 = _get_sun_proj(tref, t_peri, period, ecc,
                         xpos, ypos, north, east)
    return t_peri, qne0, vne0, xpos, ypos, north, east 
    

def _get_north_east(RA, Dec):
    """
    get the north and east vectors in the equatorial frame
    Args:
        RA: right ascension
        Dec: declination
    Returns:
        north: north vector
        east: east vector
    """
    alpha  = jnp.deg2rad(RA)
    delta  = jnp.deg2rad(Dec)
    north = jnp.array([0.0, 0.0, 1.0])
    event = jnp.array([jnp.cos(alpha) * jnp.cos(delta),
                       jnp.sin(alpha) * jnp.cos(delta),
                       jnp.sin(delta)])
    east       = jnp.cross(north, event)
    east_norm  = east / jnp.sqrt(jnp.dot(east, east)) # normalize
    north = jnp.cross(event, east_norm)
    return north, east

def _get_sun_proj(t, tperi, period, ecc, xpos, ypos, north, east):
    """
    Get the projected position of the Sun at a given time.
    Args:
        t: time
        tperi: perihelion time
        period: orbital period
        ecc: eccentricity
        xpos: x-vector of the rotated frame in the equatorial frame (x, y, z)
        ypos: y-vector of the rotated frame in the equatorial frame (x, y, z)
        north: north vector
        east: east vector
    Returns:
        sun: projected position of the Sun
    """
    
    phi = (t - tperi) / period * 2.0 * jnp.pi # mean anomaly at t
    psi = _getpsi(phi, ecc) # eccentric anomaly at t
    sun = jnp.array([xpos[i] *(jnp.cos(psi) - ecc) \
            + ypos[i] * jnp.sin(psi) * jnp.sqrt(1.0 - ecc**2) for i in range(3)]).T
    return jnp.array([jnp.dot(sun, north), jnp.dot(sun, east)]) 


def _get_xy_in_peri(rotaxis, offset, period, ecc) :
    """
    Rotate equatorial frame (x: spring (vernal equinox), y: summer, z: north-pole)
    so that x-axis aligns perihelion and return x, y vector of the rotated frame in the original equatorial frame
    Args:
        rotaxis: rotation axis (Earth axis for parallax)
        offset: tvernal - tperi
        period: orbital period
        ecc: eccentricity
    Returns:
        xpos: x-vector of the rotated frame in the equatorial frame (x, y, z)
        ypos: y-vector of the rotated frame in the equatorial frame (x, y, z)
    """
    axisrad = jnp.deg2rad(rotaxis)
    spring = jnp.array([1.0, 0.0, 0.0])
    summer = jnp.array([0.0, jnp.cos(axisrad), jnp.sin(axisrad)])
    phi = (1.0 - offset/period)*2.0*jnp.pi; # mean anomaly M of perihelion for vernal
    psi = _getpsi(phi, ecc) # convert into eccentric anomaly E
    costh = (jnp.cos(psi) - ecc)/(1.0 - ecc*jnp.cos(psi))  # cos T (T: true anomaly)
    sinth = -jnp.sqrt(1.0 - costh**2)  # angle of vernal to perihelion is > pi
    xpos =  spring * costh + summer * sinth
    ypos = -spring * sinth + summer * costh
    return xpos, ypos

def _getpsi(phi, ecc):
    """
    Solve Kepler equation using Newton method.
    Calculate the eccentric anomly for a given mean anomaly.
    Arg:
        phi: mean anomaly
        ecc: eccentricity
    Return:
        psi: eccentric anomaly
    """
    psi = phi + jnp.sign(jnp.sin(phi)) * 0.85 * ecc # Solar System Dynamics, eq. (2.64)
    for i in range(4): # 4 iterations of Newton method
        fun = psi - ecc * jnp.sin(psi)    # E_i - e*sin E_i
        dif = phi - fun                 # dif = f (E_i) = M - E_i + e*sinE_i
        der = 1.0 - ecc * jnp.cos(psi)    # der = -f'(E_i) = 1 - e*cosE_i
        psi = psi + dif / der             # E_i+1 = E_i - f(E_i)/f'(E_i)
    return psi

def _peri_vernal(tref):
    """
    get the perihelion and vernal equinox times for a given reference time
    Arg:
        tref: reference time
    Return:
        peris: perihelion times
        vernals: vernal equinox times
    """
    peris = jnp.array([1546.70833, #2000/01/03 05:00:00.0  (UT)
                       1913.87500, #2001/01/04 09:00:00.0  (UT)
                       2277.08333, #2002/01/02 14:00:00.0  (UT)
                       2643.70833, #2003/01/04 05:00:00.0  (UT)
                       3009.25000, #2004/01/04 18:00:00.0  (UT)
                       3372.54167, #2005/01/02 01:00:00.0  (UT)
                       3740.16667, #2006/01/04 16:00:00.0  (UT)
                       4104.33333, #2007/01/03 20:00:00.0  (UT)
                       4468.50000, #2008/01/03 00:00:00.0  (UT)
                       4836.12500, #2009/01/04 15:00:00.0  (UT)
                       5199.50000, #2010/01/03 00:00:00.0  (UT)
                       5565.29167, #2011/01/03 19:00:00.0  (UT)
                       5931.54167, #2012/01/05 01:00:00.0  (UT)
                       6294.70833, #2013/01/02 05:00:00.0  (UT)
                       6662.00000, #2014/01/04 12:00:00.0  (UT)
                       7026.79167, #2015/01/04 07:00:00.0  (UT)
                       7390.45833, #2016/01/02 23:00:00.0  (UT)
                       7758.08333, #2017/01/04 14:00:00.0  (UT)
                       8121.75000, #2018/01/03 06:00:00.0  (UT)
                       8486.70833, #2019/01/03 05:00:00.0  (UT)
                       8853.83333, #2020/01/05 08:00:00.0  (UT)
    ])
    vernals = jnp.array([1623.81597, #2000/03/20 07:35:00.0  (UT)
                         1989.06319, #2001/03/20 13:31:00.0  (UT)
                         2354.30278, #2002/03/20 19:16:00.0  (UT)
                         2719.54167, #2003/03/21 01:00:00.0  (UT)
                         3084.78403, #2004/03/20 06:49:00.0  (UT)
                         3450.02292, #2005/03/20 12:33:00.0  (UT)
                         3815.26806, #2006/03/20 18:26:00.0  (UT)
                         4180.50486, #2007/03/21 00:07:00.0  (UT)
                         4545.74167, #2008/03/20 05:48:00.0  (UT)
                         4910.98889, #2009/03/20 11:44:00.0  (UT)
                         5276.23056, #2010/03/20 17:32:00.0  (UT)
                         5641.47292, #2011/03/20 23:21:00.0  (UT)
                         6006.71806, #2012/03/20 05:14:00.0  (UT)
                         6371.95972, #2013/03/20 11:02:00.0  (UT)
                         6737.20625, #2014/03/20 16:57:00.0  (UT)
                         7102.44792, #2015/03/20 22:45:00.0  (UT)
                         7467.68750, #2016/03/20 04:30:00.0  (UT)
                         7832.93611, #2017/03/20 10:28:00.0  (UT)
                         8198.17708, #2018/03/20 16:15:00.0  (UT)
                         8563.41528, #2019/03/20 21:58:00.0  (UT)
                         8928.65903, #2020/03/20 03:49:00.0  (UT)
    ])
    index_min = jnp.argmin(jnp.abs(peris - tref))
    return peris[index_min], vernals[index_min]