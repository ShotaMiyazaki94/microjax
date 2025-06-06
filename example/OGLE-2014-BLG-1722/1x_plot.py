import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

params = np.load("example/OGLE-2014-BLG-1722/params_final.npy")
loss_file = np.load("example/OGLE-2014-BLG-1722/loss_curve.npy")

print(params)

plt.plot(loss_file, label="Loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(ls="--")
plt.yscale("log")
plt.savefig("example/OGLE-2014-BLG-1722/loss_curve.png", dpi=200, bbox_inches="tight")
plt.close()

import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_triple
from microjax.trajectory.parallax import peri_vernal, set_parallax, compute_parallax
from microjax.point_source import critical_and_caustic_curves, mag_point_source
from astropy.coordinates import SkyCoord
import astropy.units as u
from microjax.likelihood import linear_chi2, nll_ulens

coords = "17:55:00.57 -31:28:08.6"
c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg),)
RA = c.ra.deg
Dec = c.dec.deg
tref = 6900.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)

@jax.jit
def mag_time(time, params, parallax_params):
    t0, tE, u0, log_q, log_s, alpha, log_q3, log_s2, psi, piEN, piEE = params
    q = 10**log_q
    s = 10**log_s
    q3 = 10**log_q3
    s2 = 10**log_s2
    dtn, dum = compute_parallax(time, piEN=piEN, piEE=piEE, parallax_params=parallax_params)
    tau = (time - t0) / tE
    #um = u0 
    #tm = tau 
    um = u0 + dum
    tm = tau + dtn

    y1 = tm * jnp.cos(alpha) - um * jnp.sin(alpha)
    y2 = tm * jnp.sin(alpha) + um * jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=jnp.complex128)

    _params = {"q": q, "s": s, "q3": q3, "r3": s2, "psi": psi}
    magn = mag_point_source(w_points, nlenses=3, **_params)
    return magn, w_points


"""
@jax.jit
def mag_time(time, params, parallax_params):
    t0, tE, u0, q, s, alpha, rho, q3, s2, psi, piEN, piEE = params
    dtn, dum = compute_parallax(time, piEN=piEN, piEE=piEE, parallax_params=parallax_params)
    tau = (time - t0) / tE
    um = u0 + dum
    tm = tau + dtn

    y1 = tm * jnp.cos(alpha) - um * jnp.sin(alpha)
    y2 = tm * jnp.sin(alpha) + um * jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=jnp.complex128)

    _params = {"q": q, "s": s, "q3": q3, "r3": s2, "psi": psi}
    magn = mag_triple(w_points, rho=rho, **_params,
                      r_resolution=500, th_resolution=500, Nlimb=500)
    return magn, w_points
"""

data_moa = np.loadtxt("example/OGLE-2014-BLG-1722/data/Ian2.dat.flux.norm")
data_ogle = np.loadtxt("example/OGLE-2014-BLG-1722/data/OGLE-2014-BLG-1722.dat.flux.norm")
data_moa = data_moa[data_moa[:, 0] > 6250]
data_ogle = data_ogle[data_ogle[:, 0] > 6250]

magn_moa = mag_time(data_moa[:, 0], params, parallax_params)[0]
fluxes_moa = linear_chi2(magn_moa, data_moa[:, 1], data_moa[:, 2])
fs_moa = fluxes_moa[0]
fb_moa = fluxes_moa[2]
print(f"MOA fluxes: fs = {fs_moa}, fb = {fb_moa}")
magn_ogle = mag_time(data_ogle[:, 0], params, parallax_params)[0]
fluxes_ogle = linear_chi2(magn_ogle, data_ogle[:, 1], data_ogle[:, 2])
fs_ogle = fluxes_ogle[0]
fb_ogle = fluxes_ogle[2]
print(f"OGLE fluxes: fs = {fs_ogle}, fb = {fb_ogle}")

def magn_plot(flux, fluxe, fs, fb):
    magn_data = (flux - fb) / fs
    magne_data = fluxe / fs
    return magn_data, magne_data

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
time = jnp.linspace(6860, 6940, 1000)
magn, w_points = mag_time(time, params, parallax_params)
magn_moa, magne_moa = magn_plot(data_moa[:, 1], data_moa[:, 2], fs_moa, fb_moa)
magn_ogle, magne_ogle = magn_plot(data_ogle[:, 1], data_ogle[:, 2], fs_ogle, fb_ogle)
#model_moa = magn * fs_moa + fb_moa 
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(data_moa[:, 0], magn_moa, yerr=magne_moa, fmt=".", label="MOA data", color="gray", capsize=3)
ax.errorbar(data_ogle[:, 0], magn_ogle, yerr=magne_ogle, fmt=".", label="OGLE data", color="r", capsize=3)
ax.plot(time, magn, label="Model", color="k", zorder=10)
ax.set_xlim(6860, 6940)
ax.set_ylim(0, 8.3)
ax.set_xlabel("Time (HJD - 2450000)")
ax.set_ylabel("Magnification")

axins1 = inset_axes(ax, width="30%", height="30%", loc="upper left", borderpad=1)
x1, x2 = 6880, 6893  # 拡大範囲（例）
y1, y2 = 1, 3  # 拡大範囲（例）
valid = (data_moa[:, 0] > x1) & (data_moa[:, 0] < x2)
print(jnp.sum(valid))
valid = (data_ogle[:, 0] > x1) & (data_ogle[:, 0] < x2)
print(jnp.sum(valid))
axins1.errorbar(data_moa[:, 0], magn_moa, yerr=magne_moa, fmt=".", color="gray", capsize=2)
axins1.errorbar(data_ogle[:, 0], magn_ogle, yerr=magne_ogle, fmt=".", color="r", capsize=2)
axins1.plot(time, magn, color="k", zorder=10)
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.yaxis.tick_right()
axins1.yaxis.set_label_position("right")
mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")

axins2 = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=1)
x3, x4 = 6898, 6902  # 別の拡大範囲（例）
y3, y4 = 6, 8
valid = (data_moa[:, 0] > x3) & (data_moa[:, 0] < x4)
print(jnp.sum(valid))
valid = (data_ogle[:, 0] > x3) & (data_ogle[:, 0] < x4)
print(jnp.sum(valid))
axins2.errorbar(data_moa[:, 0], magn_moa, yerr=magne_moa, fmt=".", color="gray", capsize=2)
axins2.errorbar(data_ogle[:, 0], magn_ogle, yerr=magne_ogle, fmt=".", color="r", capsize=2)
axins2.plot(time, magn, color="k", zorder=10)
axins2.set_xlim(x3, x4)
axins2.set_ylim(y3, y4)
mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")

ax.legend(loc="lower right", fontsize=12)
plt.savefig("example/OGLE-2014-BLG-1722/model_light_curve.pdf", dpi=200, bbox_inches="tight")
plt.close()

from microjax.point_source import critical_and_caustic_curves
q, s, q3, s2, psi = 10**params[3], 10**params[4], 10**params[6], 10**params[7], params[8]
crit, cau = critical_and_caustic_curves(nlenses=3, s=s, q=q, q3=q3, r3=s2, psi=psi, npts=5000)
plt.scatter(w_points.real, w_points.imag, c=np.arange(len(w_points)), 
            s=10, label="source trajectory")
plt.scatter(cau.real, cau.imag, s=0.1, label="caustic", color="red")
plt.axis("equal")
plt.savefig("example/OGLE-2014-BLG-1722/geometry_v2.png", dpi=200, bbox_inches="tight")
plt.close()


