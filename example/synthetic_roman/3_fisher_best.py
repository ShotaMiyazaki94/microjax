import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from model import mag_time, wpoints_time
from microjax.likelihood import linear_chi2 
import pandas as pd
import numpy as np

params_best = jnp.array([ 0., 1.21045229, 0.06,
                         -4.99946803, 0.01283722, 
                         4.156, -2.73612732])

file = pd.read_csv("example/synthetic_roman/mock_data.csv")
time_lc = jnp.array(file.t.values)
flux_lc = jnp.array(file.Flux_obs.values)
fluxe_lc = jnp.array(file.Flux_err.values)

params_injected = jnp.array([ 0., 1.21045229, 0.06,
                             -4.99946803, 0.01283722, 
                             4.156, -2.73612732])

data_input = (time_lc, flux_lc, fluxe_lc)

if(0):
    def loss_fn(params, data):
        t_data, flux_data, fluxe_data = data
        magn = mag_time(t_data, params)
        _, _, _, _, chi2 = linear_chi2(magn, flux_data, fluxe_data)
        return 0.5 * chi2 # negative log likelihood

    def forward_grad(f, params, data):
        return jax.jacfwd(lambda p: f(p, data))(params)

    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params_injected)

    @jax.jit
    def update(params, opt_state, data):
        loss = loss_fn(params, data)
        grads = forward_grad(loss_fn, params, data)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    params = params_injected
    losses = []
    for step in range(1000):
        params, opt_state, loss = update(params, opt_state, data_input)
        losses.append(loss)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.3f}")

    np.save("example/synthetic_roman/adam_fwd_params", params)
    plt.figure(figsize=(12, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Adam Optimization with Forward-mode Gradients")
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("example/synthetic_roman/adam_fwd_loss_trace.png", bbox_inches="tight")
    plt.show()

best_param = np.load("example/synthetic_roman/adam_fwd_params.npy")

if(1):
    magn = mag_time(time_lc, best_param)
    Fs, Fse, Fb, Fbe, chi2 = linear_chi2(magn, flux_lc, fluxe_lc)
    def model_flux(theta):
        magn = mag_time(time_lc, theta)
        return Fs * magn + Fb
    J = jax.jacfwd(model_flux)(best_param)
    W = (1.0 / fluxe_lc**2)[:, None]
    J = np.array(J)
    W = np.array(W)
    F = J.T @ (W * J)
    np.save("example/synthetic_roman/FM_approx_v2", F)

fisher_matrix_np = np.load("example/synthetic_roman/FM_approx_v2.npy")
print(fisher_matrix_np)
damping = 0.0
fisher_matrix_pd = fisher_matrix_np + damping * np.eye(fisher_matrix_np.shape[0])
fisher_cov = np.linalg.inv(fisher_matrix_pd)
print(fisher_cov)
print(np.sqrt(np.diag(fisher_cov)))
L = jnp.linalg.cholesky(fisher_cov)
print(L)