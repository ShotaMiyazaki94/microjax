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
from microjax.likelihood import nll_ulens


data_moa  = np.load("example/ogle-2003-blg-235/flux_moa.npy")
data_ogle = np.load("example/ogle-2003-blg-235/flux_ogle.npy")
t_moa, flux_moa, fluxe_moa = data_moa[0] - 2450000, data_moa[1], data_moa[2]
t_ogle, flux_ogle, fluxe_ogle = data_ogle[0] - 2450000, data_ogle[1], data_ogle[2]
t_data     = jnp.hstack([t_moa, t_ogle])
flux_data  = jnp.hstack([flux_moa, flux_ogle])
fluxe_data = jnp.hstack([fluxe_moa, fluxe_ogle])
data_input = (t_data, flux_data, fluxe_data)


params_init = {
    "t0": 2848.16048754,
    "tE": 61.61235588,
    "u0": 0.11760426,
    "log_q": -2.3609089,
    "log_s": 0.0342508,
    "alpha": 4.00180035,
    "log_rho": -3.94500971
}

@jit
def mag_binary_time(time, params):
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    q, s, alpha = params["q"], params["s"], params["alpha"]
    rho = params["rho"]

    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

    magn = mag_binary(
        w_points, rho, s=s, q=q,
        r_resolution=500, th_resolution=500,
        cubic=True, Nlimb=500, bins_r=50, bins_th=120,
        margin_r=1.0, margin_th=1.0, MAX_FULL_CALLS=50
    )
    return magn

def loss_fn(params, data):
    t0, tE, u0 = params["t0"], params["tE"], params["u0"]
    q = 10**params["log_q"]
    s = 10**params["log_s"]
    alpha = params["alpha"]
    rho = 10**params["log_rho"]

    model_params = {"t0": t0, "tE": tE, "u0": u0, "q": q, "s": s, "alpha": alpha, "rho": rho}
    time, flux, fluxe = data
    mags = mag_binary_time(time, model_params)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    return nll_ulens(flux, M, fluxe**2, 1e9, 1e9)

# Forward-mode 勾配取得
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
for step in range(1000):
    params, opt_state, loss = update(params, opt_state, data_input)
    losses.append(loss)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.3f}")

params_np = {k: np.array(v) for k, v in params.items()}
print(params_np)
np.savez("example/ogle-2003-blg-235/adam_fwd_params.npz", **params_np)

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Adam Optimization with Forward-mode Gradients")
plt.grid(True)
plt.savefig("example/ogle-2003-blg-235/adam_fwd_loss_trace.png", bbox_inches="tight")
plt.show()
