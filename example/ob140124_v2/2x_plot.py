import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.2)

import optax
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, random, jacfwd
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.trajectory.parallax import compute_parallax, set_parallax, peri_vernal
from microjax.likelihood import linear_chi2, nll_ulens
from model import mag_time, wpoints_time
from astropy.coordinates import SkyCoord
import arviz as az

coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
RA, Dec = coords.ra.deg, coords.dec.deg
tref = 6836.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)
params_best = np.load("example/ob140124_v2/adam_fwd_params.npy")
data = np.load("example/ob140124_v2/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
data_input = (t_data, flux_data, fluxe_data, parallax_params)

idata = az.from_netcdf("example/ob140124_v2/mcmc_full.nc")
param_names = ['t0_diff', 'log_tE', 'u0', 
               'log_q', 'log_s', 'alpha',
               'log_rho', 'piEN', 'piEE']

samples = idata.posterior["param"].values.reshape(-1, 9)
corner_labels = [r"$t_0^\prime$", r"$\log t_E$", r"$u_0$", 
                 r"$\log q$", r"$\log s$", r"$\alpha$", 
                 r"$\log \rho$", r"$\pi_{E, N}$", r"$\pi_{E, E}$"]

if(0):
    import corner
    fig = corner.corner(samples, labels=corner_labels, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
    fig.savefig("example/ob140124_v2/corner_plot.pdf", bbox_inches="tight")
    plt.close()

if(0):
    samples_reshaped = idata.posterior["param"].values
    idata_named = az.from_dict(posterior={name: samples_reshaped[:, :, i] for i, name in enumerate(param_names)})
    az.plot_trace(idata_named, var_names=param_names)
    plt.tight_layout()
    plt.savefig("example/ob140124_v2/trace_plot.pdf", bbox_inches="tight")
    plt.close()

if(1):
    medians = np.median(samples, axis=[0])
    params = medians
    magn = mag_time(t_data, medians, parallax_params)
    Fs, Fse, Fb, Fbe, chi2 = linear_chi2(magn, flux_data, fluxe_data)
    def magn_plot(flux, fluxe, fs, fb):
        magn_data = (flux - fb) / fs
        magne_data = fluxe / fs
        return magn_data, magne_data
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    time = jnp.linspace(6650, 7000, 2000)
    magn = mag_time(time, params, parallax_params)
    magn_data, magne_data = magn_plot(flux_data, fluxe_data, Fs, Fb)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(t_data, magn_data, yerr=magne_data, fmt=".", label="OGLE-I", color="gray", capsize=3)
    ax.plot(time, magn, label="Model", color="k", zorder=10)
    ax.set_xlim(6650, 7000)
    ax.set_xlabel("Time (HJD - 2450000)")
    ax.set_ylabel("Magnification")
    ax.legend(loc="lower center")

    axins1 = inset_axes(ax, width="30%", height="30%", loc="upper left", borderpad=1)
    x1, x2 = 6830, 6850 
    y1, y2 = 3, 7
    axins1.errorbar(t_data, magn_data, yerr=magne_data, fmt=".", label="OGLE-I", color="gray", capsize=3)
    axins1.plot(time, magn, color="k", zorder=10)
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    axins1.yaxis.tick_right()
    axins1.yaxis.set_label_position("right")
    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.5")

    axins2 = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=1)
    from microjax.point_source import critical_and_caustic_curves
    import matplotlib as mpl
    q = 10**params[3]
    s = 10**params[4]
    rho = 10**params[6]
    crit, cau = critical_and_caustic_curves(nlenses=2, npts=1000, s=s, q=q)
    for cc in cau:
        axins2.plot(cc.real, cc.imag, color='red', lw=0.7)
    w_points = wpoints_time(time, params, parallax_params)
    print(w_points)
    circles = [
        plt.Circle((xi,yi), radius=rho, fill=False) for xi,yi in zip(w_points.real, w_points.imag)
    ]
    c = mpl.collections.PatchCollection(
        circles, facecolor='none', edgecolor='blue', linewidth=0.7, alpha=1, zorder=2)
    #c = mpl.collections.PatchCollection(circles, match_original=True, alpha=1)
    axins2.plot(w_points.real, w_points.imag, color="b")
    axins2.add_collection(c)
    axins2.set_aspect(1)
    axins2.set(xlim=(-0.3, 0.1), ylim=(-0.075, 0.075))

    plt.savefig("example/ob140124_v2/lc.pdf", bbox_inches="tight")

exit(1)