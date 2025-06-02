import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_triple
from scipy.interpolate import interp1d
from microjax.trajectory.parallax import peri_vernal, set_parallax, compute_parallax

from astropy.coordinates import SkyCoord
import astropy.units as u
coords = "17:55:00.57 -31:28:08.6"
c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg),)
RA = c.ra.deg
Dec = c.dec.deg

# Load the data
data_moa = np.loadtxt("example/OGLE-2014-BLG-1722/data/Ian2.dat.flux.norm")
data_ogle = np.loadtxt("example/OGLE-2014-BLG-1722/data/OGLE-2014-BLG-1722.dat.flux.norm")

data_moa = data_moa[data_moa[:, 0] > 6250]
data_ogle = data_ogle[data_ogle[:, 0] > 6250]

plt.figure(figsize=(15, 6))
plt.errorbar(data_moa[:, 0], data_moa[:, 1], yerr=data_moa[:, 2], fmt=".", label="MOA")
plt.errorbar(data_ogle[:, 0], data_ogle[:, 1], yerr=data_ogle[:, 2], fmt=".", label="OGLE")
#plt.xlim(6860, 6940)
plt.grid(ls="--")
plt.savefig("example/OGLE-2014-BLG-1722/data.png", dpi=200)
plt.close()


time = jnp.linspace(6860, 6940, 1000)
t0 = 6900.224
tE = 23.819
u0 = -0.131
q = 4.468e-4
s = 0.754
alpha = -0.228 # rad
rho = 5e-3
q3 = 6.388e-4
s2 = 0.851
s2 = 0.9
psi = -2.196 # rad
piEN = 0.199
piEE = 0.092

tref = t0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)

from microjax.point_source import critical_and_caustic_curves

params = jnp.array([t0, tE, u0, q, s, alpha, rho, q3, s2, psi, piEN, piEE])

@jax.jit
def magn_triple(time, params, parallax_params):
    t0, tE, u0, q, s, alpha, rho, q3, s2, psi, piEN, piEE = params
    dtn, dum = compute_parallax(time, piEN, piEE, parallax_params)
    tau = (time - t0) / tE + dtn
    um = u0 + dum
    y1 = tau * jnp.cos(alpha) - um * jnp.sin(alpha)
    y2 = tau * jnp.sin(alpha) + um * jnp.cos(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    magn = mag_triple(w_points, rho, q=q, s=s, q3=q3, r3=s2, psi=psi)
    return magn, w_points    


from microjax.likelihood import linear_chi2
crit, cau = critical_and_caustic_curves(nlenses=3, s=s, q=q, q3=q3, r3=s2, psi=psi, npts=5000)
_params = {"q": q, "s": s, "q3": q3, "r3": s2, "psi": psi}

#data_moa = data_moa[(data_moa[:, 0] <= 6940)&(data_moa[:, 0] >= 6860)]
#data_ogle = data_ogle[(data_ogle[:, 0] <= 6940)&(data_ogle[:, 0] >= 6860)]
magn_moa = magn_triple(data_moa[:, 0], params, parallax_params)[0]
fluxes_moa = linear_chi2(magn_moa, data_moa[:, 1], data_moa[:, 2])
fs_moa = fluxes_moa[0]
fb_moa = fluxes_moa[2]
print(f"MOA fluxes: fs = {fs_moa}, fb = {fb_moa}")
#magn_ogle = magn_triple(data_ogle[:, 0], params)[0]
#fs_ogle, fb_ogle = linear_chi2(magn_ogle, data_ogle[:, 1], data_ogle[:, 2])
time = jnp.linspace(6860, 6940, 1000)
magn, w_points = magn_triple(time, params, parallax_params)
model_moa = magn * fs_moa + fb_moa 

plt.errorbar(data_moa[:, 0], data_moa[:, 1], yerr=data_moa[:, 2], fmt=".", label="MOA data")
plt.plot(time, model_moa, label="MOA model", color="orange", zorder=10)
plt.savefig("example/OGLE-2014-BLG-1722/model_moa.png", dpi=200)
plt.close()

plt.scatter(w_points.real, w_points.imag, c=np.arange(len(w_points)), 
            s=10, label="source trajectory")
plt.scatter(cau.real, cau.imag, s=0.1, label="caustic", color="red")
plt.axis("equal")
plt.savefig("example/OGLE-2014-BLG-1722/geometry.png", dpi=200)
plt.close()

magn = mag_triple(w_points, rho, **_params) 
plt.plot(time, magn, label="magnification")
plt.xlabel("Time [HJD - 2450000]")
plt.ylabel("Magnification")
plt.savefig("example/OGLE-2014-BLG-1722/magn.png", dpi=200)
plt.close()