import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)
from microjax.inverse_ray.lightcurve import mag_binary
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from scipy.interpolate import interp1d

"""
2L1Sp ID:  95 W149s: 25.09 q : 7.28e-05 s : 0.947 ts: 0.98 tE: 25.52 rho: 1.60e-03 
Ml: 9.00e-02 Mp: 2.18e+00 a: 1.85e+00 u0: -6.3e-02 al: 2.340 fs1: 1.04e-01 t0: 516.7182 
"""
mage_table = pd.read_csv("example/data/WFIRST/full_magnitude_error.csv")
mage_func = interp1d(mage_table.mag, mage_table.err, bounds_error=False, fill_value="extrapolate")

u0 = 0.063
q = 7.28e-05
s = 0.947
rho = 1.60e-03
t0 = 0.0
tE = 25.52
alpha = 2.340
W149s = 25.1
fs0 = 0.104
mag0 = 25.0
Fs = 10**(-0.4 * (W149s - mag0))
Fb = Fs*(1.0 - fs0)/fs0
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
fig, ax = plt.subplots()
ax.plot(t, Flux_obs, ".", label="Observed")
ax.plot(t, Flux_model, "-", label="Model")
axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
axins.plot(t, Flux_obs, ".", ms=2)
axins.plot(t, Flux_model, "-", lw=1)
axins.set_xlim(0, 3)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Flux (arbitrary units)")
ax.legend(loc="upper left")
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