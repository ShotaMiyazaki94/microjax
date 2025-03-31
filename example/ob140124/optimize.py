import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MulensModel as mm
plt.rcParams["figure.figsize"] = (12,5)
import seaborn as sns
sns.set_theme(style='ticks', font_scale=1.2,)
from matplotlib import rc

data = pd.read_csv("example/data/ogle-2014-blg-0124/phot.dat", 
                   sep='\s+',header=None, names=["HJD", "mag", "mage", "seeing", "sky"])
data["HJD"] -= 2450000
data = data[data.HJD>6300]
#data = data[(data.HJD>2.4566e+6)&(data.mage<0.4)]
mag0 = 18.0
data["flux"] = 10**(-0.4*(data.mag - mag0))
data["fluxe"] = data.flux * 0.4 * np.log(10) * data.mage

import jax
from jax import lax, vmap, jit
import jax.numpy as jnp
from microjax.inverse_ray.lightcurve import mag_lc_uniform
from microjax.multipole import _mag_hexadecapole
from microjax.point_source import _images_point_source
from microjax.inverse_ray.cond_extended import _caustics_proximity_test, _planetary_caustic_test
jax.config.update("jax_enable_x64", True)

# initial guess
t0, u0, tE = 6.83640951e+03, 2.24211333e-01, 1.33559958e+02 
s, q, alpha = 9.16157288e-01, 5.87559438e-04, jnp.deg2rad(1.00066409e+02)
rho = 2.44003713e-03
fs, fb = 8.06074085e-01, 8.62216897e-01
f_sum_ = fs + fb

nlenses = 2
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1}
x_cm = a * (1 - q) / (1 + q)


Nlimb = 500
r_resolution  = 500
th_resolution = 500
MAX_FULL_CALLS = 200

cubic = True
bins_r = 50
bins_th = 120
margin_r = 1.0
margin_th= 1.0

t = jnp.array(data.HJD)
#t = jnp.linspace(6650, 7000, 500)
tau = (t - t0)/tE 
um  = u0
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)  
w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

mags = mag_lc_uniform(w_points, rho, s=s, q=q, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic, 
                       Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th, margin_r=margin_r, margin_th=margin_th, MAX_FULL_CALLS=MAX_FULL_CALLS)
f_ = fs * mags + fb

z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
test1 = _caustics_proximity_test(
    w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params 
    )
test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)
test = lax.cond(
    q < 0.01, 
    lambda:test1 & test2,
    lambda:test1,
)
print("num: %d"%len(w_points))
print("full num: %d"%jnp.sum(~test))


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)
import seaborn as sns
sns.set_theme(style='ticks', font_scale=1.4,) 
plt.plot(t, f_, "-", color='red', label="2L1S (microjax)")
plt.errorbar(data.HJD, data.flux, yerr=data.fluxe, fmt='.', color='black', label="OGLE-I")
plt.title("OGLE-2014-BLG-0124")
plt.xlabel("HJD - 2,450,000")
plt.ylabel("flux")
plt.xlim(6300,7000)
plt.legend()
plt.savefig("example/ob140124/ob140124_plot.pdf", bbox_inches="tight")
plt.close()