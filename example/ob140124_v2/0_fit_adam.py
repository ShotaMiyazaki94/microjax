import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import optax
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, random
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.trajectory.parallax import compute_parallax, set_parallax, peri_vernal
from microjax.likelihood import linear_chi2
from model import mag_time

import pandas as pd
import numpy as np

params_init = jnp.array([
    6.83640951e+03 - 6836.0,  # t0_diff
    jnp.log10(1.33559958e+02),  # log_tE
    2.24211333e-01,            # u0
    jnp.log10(5.87559438e-04),  # log_q
    jnp.log10(9.16157288e-01),  # log_s
    jnp.deg2rad(1.00066409e+02), # alpha
    jnp.log10(2.44003713e-03),  # log_rho
    1.82341182e-01,            # piEN
    9.58542572e-02,            # piEE
], dtype=jnp.float64)

from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
RA, Dec = coords.ra.deg, coords.dec.deg
tref = 6836.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)

if(0):
    photo_file = pd.read_csv("example/data/ogle-2014-blg-0124/phot.dat", sep='\s+', header=None)
    photo_file.columns = ["HJD", "I", "Ie", "seeing", "sky"]
    photo_file["HJD"] = photo_file["HJD"].astype(float)
    photo_file["I"] = photo_file["I"].astype(float)
    photo_file["Ie"] = photo_file["Ie"].astype(float)

    mag0 = 18.0
    Flux = 10**(-0.4 * (photo_file["I"].values - mag0))
    Flux_err = 0.4 * np.log(10) * photo_file["Ie"].values * Flux
    Flux_err = np.abs(Flux_err)
    data = np.array([photo_file["HJD"].values, Flux, Flux_err, photo_file["seeing"].values, photo_file["sky"].values])
    np.save("example/ob140124_v2/flux", data)

    t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
    magn = mag_time(t_data, params_init, parallax_params=parallax_params)
    Fs, Fse, Fb, Fbe, chi2 = linear_chi2(magn, flux_data, fluxe_data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_data, flux_data, "o", markersize=1, label="Data")
    plt.plot(t_data, magn*Fs + Fb, "r-", label="Model")
    plt.xlabel("Time (HJD - 2450000)")
    plt.ylabel("Flux (normalized)")
    plt.xlim(6600, 7000)
    plt.title("OGLE-2014-BLG-1722 Light Curve")
    plt.legend()
    plt.savefig("example/ob140124_v2/light_curve.png", dpi=300)
    exit(1)

data = np.load("example/ob140124_v2/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
data_input = (t_data, flux_data, fluxe_data, parallax_params)

def loss_fn(params, data):
    t_data, flux_data, fluxe_data, parallax_params = data
    magn = mag_time(t_data, params, parallax_params)
    _, _, _, _, chi2 = linear_chi2(magn, flux_data, fluxe_data)
    return 0.5 * chi2 # negative log likelihood

def forward_grad(f, params, data):
    return jax.jacfwd(lambda p: f(p, data))(params)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params_init)

@jit
def update(params, opt_state, data):
    loss = loss_fn(params, data)
    grads = forward_grad(loss_fn, params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params = params_init
losses = []
for step in range(1000):
    params, opt_state, loss = update(params, opt_state, data_input)
    losses.append(loss)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.3f}")

np.save("example/ob140124_v2/adam_fwd_params", params)
plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Adam Optimization with Forward-mode Gradients")
plt.grid(True)
plt.yscale("log")
plt.savefig("example/ob140124_v2/adam_fwd_loss_trace.png", bbox_inches="tight")
plt.show()