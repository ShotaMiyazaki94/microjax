import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

data = pd.read_csv("example/synthetic_roman/mock_data.csv")
t_data = jnp.array(data["t"].values)
flux_data = jnp.array(data["Flux_obs"].values)
fluxe_data = jnp.array(data["Flux_err"].values)
data_input = (t_data, flux_data, fluxe_data)

u0 = 0.063
q = 7.28e-05
s = 0.947
rho = 1.60e-03
t0 = 0.0
tE = 25.52
alpha = 2.340

params_dict = {"t0": t0, "tE": tE, "u0": u0,
               "log_q": jnp.log10(q), "log_s": jnp.log10(s),
               "alpha": alpha, "log_rho": jnp.log10(rho),}
keys = ["t0", "tE", "u0", "log_q", "log_s", "alpha", "log_rho"]
params_jnp = jnp.array([params_dict[key] for key in keys])


@jit
def mag_time(time, params):
    t0, tE, u0, log_q, log_s, alpha, log_rho = params 
    q = jnp.power(10.0, log_q)
    s = jnp.power(10.0, log_s)
    rho = jnp.power(10.0, log_rho)
    tau = (time - t0)/tE
    y1 = tau*jnp.cos(alpha) - u0*jnp.sin(alpha) 
    y2 = tau*jnp.sin(alpha) + u0*jnp.cos(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    magns = mag_binary(w_points, rho, s=s, q=q, r_resolution=500, th_resolution=500, chunk_size=1)
    return magns

@jit
def nll_fn(theta_array):
    mags = mag_time(t_data, theta_array)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    return nll_ulens(flux_data, M, fluxe_data**2, jnp.array(1e9), jnp.array(1e9))

@partial(jax.jit, static_argnums=0)
def hessian_fwd_fwd_vmap(f, x):
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    def hess_col(ei):
        def first_derivative(x):
            return jax.jvp(f, (x,), (ei,))[1]
        return jax.vmap(lambda ej: jax.jvp(first_derivative, (x,), (ej,))[1])(eye)
    return jax.vmap(hess_col)(eye) 

from jax import jacfwd
#nll_fn_fixed = lambda theta: nll_fn(theta)
jac_fn = jacfwd(nll_fn)
hessian_fn = jit(jacfwd(jacfwd(nll_fn)))
#J = jac_fn(params_jnp)
#print(J.shape)
#fisher_matrix = jnp.outer(J, J)  # Fisher Information Matrix
#fisher_matrix = J[:, None] @ J[None, :]  # Fisher Information Matrix
fisher_matrix = hessian_fn(params_jnp)
print(fisher_matrix.shape)
#hessian_fn = jacfwd(jacfwd(nll_fn))
#fisher_matrix = hessian_fn(params_jnp)
#fisher_matrix = hessian_fwd_fwd_vmap(nll_fn_fixed, params_jnp)
fisher_cov = jnp.linalg.pinv(fisher_matrix + 1e-12*jnp.eye(fisher_matrix.shape[0]))
L = jnp.linalg.cholesky(fisher_cov + 1e-18*jnp.eye(fisher_matrix.shape[0]))

print("Fisher Matrix:")
print(np.array(fisher_matrix))
print("Fisher Covariance Matrix:")
print(np.array(fisher_cov))
print("Cholesky Decomposition of Fisher Covariance Matrix:")
print(np.array(L))
save_matrix = np.array([fisher_matrix, fisher_cov, L])

param_stddev = jnp.sqrt(jnp.diag(fisher_cov))
print("Parameter Standard Deviations:")
for key, param, sigma in zip(keys, params_jnp, param_stddev):
    print(f"{key}: {param:.6f} ± {sigma:.2e}")
np.save("example/synthetic_roman/save_matrix", save_matrix)

exit(1)



from functools import partial
from jax import jacfwd
nll_fn_fixed = partial(nll_fn)
hessian_fn = jacfwd(jacfwd(nll_fn_fixed))

fisher_matrix = hessian_fn(params_jnp)
fisher_cov = jnp.linalg.inv(fisher_matrix)
param_stddev = jnp.sqrt(jnp.diag(fisher_cov))
fisher_matrix = np.array(fisher_matrix)

print("Fisher Matrix:")
print(fisher_matrix)





def loss_fn(params, data):
    t0, tE, u0 = params["t0"], params["tE"], params["u0"]
    log_q = params["log_q"]
    log_s = params["log_s"]
    alpha = params["alpha"]
    log_rho = params["log_rho"]

    model_params = {"t0": t0, "tE": tE, "u0": u0, 
                    "log_q": log_q, "log_s": log_s, 
                    "alpha": alpha, "log_rho": log_rho,}
    time, flux, fluxe = data
    mags = mag_time(time, model_params)
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
for step in range(1000):
    params, opt_state, loss = update(params, opt_state, data_input)
    losses.append(loss)
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.3f}")

params_np = {k: np.array(v) for k, v in params.items()}
print(params_np)
np.savez("example/synthetic_roman/adam_fwd_params.npz", **params_np)

plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Adam Optimization with Forward-mode Gradients")
plt.grid(True)
plt.savefig("example/synthetic_roman/adam_fwd_loss_trace.png", bbox_inches="tight")
plt.show()