import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.2)
from microjax.inverse_ray.lightcurve import mag_binary
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from scipy.interpolate import interp1d

"""
2L1Sp ID: 100 W149s: 24.744 q : 7.352e-05 s : 0.961 al: 4.156 rho:1.836e-03 
ts: 0.715 t0: 296.6540 tE: 16.235 u0: 0.060 Ml: 0.300 Mp: 7.343 a:  2.232 fs1: 3.95e-01

2L1Sp ID:  47 W149s: 20.73 q : 8.45e-06 s : 1.028 ts: 0.55 tE: 7.09 rho: 3.22e-03 
Ml: 1.33e-01 Mp: 3.74e-01 a: 1.13e+00 u0: 3.0e-02 al: 4.331 fs1: 7.39e-01 t0: 1406.7452 

2L1Sp ID:  95 W149s: 25.09 q : 7.28e-05 s : 0.947 ts: 0.98 tE: 25.52 rho: 1.60e-03 
Ml: 9.00e-02 Mp: 2.18e+00 a: 1.85e+00 u0: -6.3e-02 al: 2.340 fs1: 1.04e-01 t0: 516.7182 
"""
mage_table = pd.read_csv("example/data/WFIRST/full_magnitude_error.csv")
mage_func = interp1d(mage_table.mag, mage_table.err, bounds_error=False, fill_value="extrapolate")

u0 = 0.06
q = 7.352e-05 / 7.343 
s = 1.03
#s = 0.961 * 1.2
rho = 1.836e-3
t0 = 0.0
tE = 16.235
alpha = 4.156
W149s = 24.744
#fs0 = 0.8
fs0 = 0.395
mag0 = 25.0
Fs = 10**(-0.4 * (W149s - mag0))
Fb = Fs*(1.0 - fs0)/fs0
params=jnp.array([t0, u0, jnp.log10(tE), 
                  jnp.log10(q), jnp.log10(s), 
                  alpha, jnp.log10(rho)])
print(params)
print("Fs: %.3e, Fb: %.3e"%(Fs, Fb))

t = jnp.arange(-36.0, 36.0, 12/60.0/24.0)
tau = (t - t0)/tE
y1 = tau*jnp.cos(alpha) - u0*jnp.sin(alpha) 
y2 = tau*jnp.sin(alpha) + u0*jnp.cos(alpha)
w_points = jnp.array(y1 + 1j * y2, dtype=complex)

magns = mag_binary(w_points, rho, s=s, q=q, r_resolution=1000, th_resolution=1000)
Flux_model  = Fs * magns + Fb
W149_model  = -2.5 * jnp.log10(Flux_model) + mag0
W149e_model = mage_func(W149_model)
key = jax.random.PRNGKey(0)
Fluxe_obs = (jnp.log(10) / 2.5) * Flux_model * W149e_model
Flux_obs = Flux_model + jax.random.normal(key, shape=Flux_model.shape) * Fluxe_obs
W149_obs = -2.5 * jnp.log10(Flux_obs) + mag0
W149e_obs = W149e_model

mock_data = pd.DataFrame({"t": t,
                          "Flux_obs": Flux_obs,
                          "Flux_err": Fluxe_obs,
                          "W149_obs": W149_obs,
                          "W149e_obs": W149e_obs,
                          "Flux_model": Flux_model,
                          "W149_model": W149_model,
                          "Magnification_model": magns,})
mock_data.to_csv("example/synthetic_roman/mock_data.csv", index=False)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, W149_obs, ".", ms=2, label="synthetic data")
ax.plot(t, W149_model, "-", label="model")
ylim = ax.get_ylim()
ax.set_ylim(ylim[1], ylim[0])
ax.legend(loc="lower center", fontsize=12)
ax.set_title(r"$M_L=0.3M_\odot$, $M_p=1.0M_\oplus$, $a=2.3$au")

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
axins = inset_axes(ax, width="30%", height="30%", loc='upper left')
xmin, xmax = -0.75, -0.45
axins.set_xlim(xmin, xmax)
ymin = W149_obs[(t > xmin)&(t < xmax)].min()
ymax = W149_obs[(t > xmin)&(t < xmax)].max()
axins.errorbar(t, W149_obs, yerr=W149e_obs, fmt=".", capsize=2, markersize=3, label="Observed")
axins.plot(t, W149_model, "-", lw=1, zorder=10)
axins.set_ylim(ymax + 0.1*(ymax - ymin), ymin - 0.1*(ymax-ymin))
axins.set_xlim(xmin, xmax)
axins.yaxis.tick_right()
axins.yaxis.set_label_position("right")
#mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

axins2 = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=1)
from microjax.point_source import critical_and_caustic_curves
import matplotlib as mpl
crit, cau = critical_and_caustic_curves(nlenses=2, npts=1000, s=s, q=q)
for cc in cau:
    axins2.plot(cc.real, cc.imag, color='red', lw=0.7)
axins2.set_aspect(1)
axins2.plot(w_points.real, w_points.imag, color="b")
axins2.set(xlim=(-0.025, 0.1), ylim=(-0.025, 0.025))

ax.set_xlabel("Time (days)")
ax.set_ylabel("W149 (mag)")
#ax.legend(loc="upper left")
#plt.plot(t, Flux_obs, ".", label="Model")
#plt.plot(t, Flux_model, "k-", label="Observed")
#plt.xlim(-0, 3)
plt.savefig("example/synthetic_roman/0_gen_lc.pdf", bbox_inches='tight')
plt.close()


if(0):
    plt.plot(t, magns, ".")
    plt.xlim(-0, 3)
    #plt.savefig("example/synthetic_roman/0_gen_lc.pdf")
    plt.show()


if(0):
    plt.figure(figsize=(8,5))
    plt.plot(mage_table.mag, mage_table.err, ".")
    plt.yscale("log")
    plt.xlabel("W149 (mag)")
    plt.ylabel("mag error")
    plt.show()