from jax import config 
config.update('jax_enable_x64', True) 
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from microjax.inverse_ray.lightcurve import mag_lc, mag_lc_vmap
from microjax.trajectory import dtn_dum_parallax, _get_info_parallax

data = pd.read_csv("example/data/ogle-2014-blg-0124/phot.dat", delim_whitespace=True,header=None, names=["HJD", "mag", "mage", "seeing", "sky"])
data = data[(data.HJD>2.4563e+6)&(data.mage<0.4)]
#data = data[(data.HJD>2.4566e+6)&(data.mage<0.4)]
data["HJD"] -= 2450000
mag0 = 18.0
data["flux"] = 10**(-0.4*(data.mag - mag0))
data["fluxe"] = data.flux * 0.4 * np.log(10) * data.mage
# initial guess
_t_0, _u_0, _t_E = 6.83640951e+03, 2.24211333e-01, 1.33559958e+02 
_s, _q, _alpha = 9.16157288e-01, 5.87559438e-04, 1.00066409e+02
_rho, _pi_E_N, _pi_E_E = 2.44003713e-03, 1.82341182e-01,9.58542572e-02
_fs, _fb = 8.06074085e-01, 8.62216897e-01
_f_sum = _fs + _fb

from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
coords_deg = (coords.ra.deg, coords.dec.deg)
tref = 6836.0
info_parallax = _get_info_parallax(RA=coords_deg[0], Dec=coords_deg[1], tref=tref)
t_peri, qne0, vne0, xpos, ypos, north, east = info_parallax

t = jnp.linspace(6650, 7000, 2000)
_alpha_rad = jnp.deg2rad(_alpha)
dtn, dum = dtn_dum_parallax(t, _pi_E_N, _pi_E_E, t_peri, qne0, vne0, xpos, ypos, north, east)
print("!!!:",dtn, dum)
tau = (t - _t_0)/_t_E 
um  = _u_0 
#tau = (t - _t_0)/_t_E + dtn
#um  = _u_0 + dum 
y1 = -um*jnp.sin(_alpha_rad) + tau*jnp.cos(_alpha_rad)
y2 = um*jnp.cos(_alpha_rad) + tau*jnp.sin(_alpha_rad)
w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
mags = mag_lc_vmap(w_points, _rho, s=_s, q=_q, nlenses=2, cubic=True, r_resolution=1000, th_resolution=4000)
_f = _fs * mags + _fb

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)
import seaborn as sns
sns.set_theme(style='ticks', font_scale=1.4,) 
plt.plot(t, _f, "-", color='red', label="2L1S (microjax)")
plt.errorbar(data.HJD, data.flux, yerr=data.fluxe, fmt='.', color='black', label="OGLE-I")
plt.title("OGLE-2014-BLG-0124")
plt.xlabel("HJD - 2,450,000")
plt.ylabel("flux")
plt.xlim(6650,7000)
plt.legend()
plt.savefig("example/ob140124/ob140124_plot.pdf")
plt.close()

#print(coords_deg)
#print(t_peri, qne0, vne0, xpos, ypos, north, east)
