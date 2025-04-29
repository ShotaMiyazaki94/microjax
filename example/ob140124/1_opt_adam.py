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
from microjax.likelihood import nll_ulens

data = np.load("example/ob140124/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
t_data = jnp.array(t_data)
flux_data = jnp.array(flux_data)
fluxe_data = jnp.array(fluxe_data)
from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
data_input = (t_data, flux_data, fluxe_data, coords.ra.deg, coords.dec.deg)

params_init = {
    "t0": 6.83640951e+03,
    "tE": 1.33559958e+02, 
    "u0": 2.24211333e-01,
    "log_q": jnp.log10(5.87559438e-04),
    "log_s": jnp.log10(9.16157288e-01),
    "alpha_deg": 1.00066409e+02,
    "log_rho": jnp.log10(2.44003713e-03),
    "piEE": 9.58542572e-02,
    "piEN": 1.82341182e-01,
}

#@partial(jit, static_argnames=("RA", "Dec"))
@jit
def mag_time(time, params, RA, Dec):
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    log_q, log_s, alpha_deg = params["log_q"], params["log_s"], params["alpha_deg"]
    log_rho, piEE, piEN = params["log_rho"], params["piEE"], params["piEN"]
    q = 10**log_q
    s = 10**log_s
    rho = 10**log_rho
    alpha = jnp.deg2rad(alpha_deg)

    tref = t0
    tperi, tvernal = peri_vernal(tref)
    parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)
    dtn, dum = compute_parallax(time, piEN=piEN, piEE=piEE, parallax_params=parallax_params)
    tau = (time - t0)/tE
    um  = u0 + dum
    tm  = tau + dtn

    # convert (tn, u0) -> (y1, y2)
    y1 = tm*jnp.cos(alpha) - um*jnp.sin(alpha) 
    y2 = tm*jnp.sin(alpha) + um*jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

    magn = mag_binary(
        w_points, rho=rho, s=s, q=q,
        r_resolution=500, th_resolution=500,
        cubic=True, Nlimb=500, bins_r=100, bins_th=360,
        margin_r=1.0, margin_th=1.0, MAX_FULL_CALLS=100
    )
    return magn

def loss_fn(params, data):
    t0, tE, u0 = params["t0"], params["tE"], params["u0"]
    log_q = params["log_q"]
    log_s = params["log_s"]
    alpha_deg = params["alpha_deg"]
    log_rho = params["log_rho"]
    piEE = params["piEE"]
    piEN = params["piEN"]

    model_params = {"t0": t0, "tE": tE, "u0": u0, 
                    "log_q": log_q, "log_s": log_s, "alpha_deg": alpha_deg, 
                    "log_rho": log_rho, "piEE": piEE, "piEN": piEN}
    time, flux, fluxe, RA_deg, Dec_deg = data
    mags = mag_time(time, model_params, RA_deg, Dec_deg)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    return nll_ulens(flux, M, fluxe**2, 1e9, 1e9)

def forward_grad(loss_fn, params, data):
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(params)
    def flat_loss(p):
        return loss_fn(unravel_fn(p), data)
    grad_flat = jax.jacfwd(flat_loss)(flat_params)
    grads = unravel_fn(grad_flat)
    return grads

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
for step in range(10000):
    params, opt_state, loss = update(params, opt_state, data_input)
    losses.append(loss)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.3f}")

params_np = {k: np.array(v) for k, v in params.items()}
print(params_np)
np.savez("example/ob140124/adam_fwd_params.npz", **params_np)

plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Adam Optimization with Forward-mode Gradients")
plt.grid(True)
plt.savefig("example/ob140124/adam_fwd_loss_trace.png", bbox_inches="tight")
plt.show()



if(0):
    t_plot = jnp.linspace(6650, 7000, 1000)
    #t_plot = jnp.linspace(t_data.min(), t_data.max(), 1000)
    magn = mag_time(t_plot, params_init, RA=coords.ra.deg, Dec=coords.dec.deg)
    fs0, fb0 = 8.06074085e-01, 8.62216897e-01
    plt.figure(figsize=(12, 5))
    plt.plot(t_plot, magn*fs0 + fb0, label="microJAX", color="red")
    plt.errorbar(t_data, flux_data, yerr=fluxe_data, fmt='.', color='black', label="OGLE-I")
    plt.xlim(t_plot.min(), t_plot.max())
    plt.xlabel("Time [JD - 2450000]")
    plt.ylabel("Magnification")
    plt.title("OGLE-2014-BLG-0124")
    plt.grid(True, ls=":")
    plt.tight_layout()
    plt.savefig("example/ob140124/magn.png", dpi=200)
    plt.legend()
    plt.show()