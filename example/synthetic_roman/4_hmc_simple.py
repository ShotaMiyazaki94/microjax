import os
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, jacfwd
jax.config.update("jax_enable_x64", True)
import pandas as pd
from model import mag_time
from microjax.likelihood import nll_ulens
import corner


params_best = np.load("example/synthetic_roman/adam_fwd_params.npy")
fisher_matrix_np = np.load("example/synthetic_roman/FM_approx_v2.npy")
damping = 0.0
fisher_matrix_pd = fisher_matrix_np + damping * np.eye(fisher_matrix_np.shape[0])
fisher_cov = np.linalg.inv(fisher_matrix_pd)
errs_FM = np.sqrt(np.diag(fisher_cov))

data = pd.read_csv("example/synthetic_roman/mock_data.csv")
t_data = jnp.array(data["t"].values)
flux_data = jnp.array(data["Flux_obs"].values)
fluxe_data = jnp.array(data["Flux_err"].values)
data_input = (t_data, flux_data, fluxe_data)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data, init_val, errs_input):
    times, flux, fluxe = data
    t0      = numpyro.sample("t0",      dist.Normal(init_val[0], errs_input[0]))
    log_tE  = numpyro.sample("log_tE",  dist.Normal(init_val[1], errs_input[1]))
    u0      = numpyro.sample("u0",      dist.Normal(init_val[2], errs_input[2]))
    log_q   = numpyro.sample("log_q",   dist.Normal(init_val[3], errs_input[3]))
    log_s   = numpyro.sample("log_s",   dist.Normal(init_val[4], errs_input[4]))
    alpha   = numpyro.sample("alpha",   dist.Normal(init_val[5], errs_input[5]))
    log_rho = numpyro.sample("log_rho", dist.Normal(init_val[6], errs_input[6]))
    params = jnp.array([t0, log_tE, u0, log_q, log_s, alpha, log_rho])
    mags = mag_time(times, params)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    sigma2_obs = fluxe ** 2
    sigma2_fs = 1e+9
    sigma2_fb = 1e+9
    nll = nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    numpyro.factor("log_likelihood", -nll)

init_strategy=numpyro.infer.init_to_median()
kernel = NUTS(model,
              init_strategy=init_strategy, 
              dense_mass=True,
              inverse_mass_matrix=fisher_cov, # useless...? 
              regularize_mass_matrix=True,
              adapt_mass_matrix=True,
              adapt_step_size=True, 
              forward_mode_differentiation=True,
              target_accept_prob=0.8,
              )

init_val = params_best
errs_input = 10.0 * errs_FM

mcmc = MCMC(kernel, num_warmup=500, num_samples=3000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data=data_input, init_val=init_val, errs_input=errs_input)
mcmc.print_summary(exclude_deterministic=False)

import arviz as az
import corner
idata = az.from_numpyro(mcmc)
idata.to_netcdf("example/synthetic_roman/mcmc_full_noL.nc")

param_labels = ["t0", "log_tE", "u0", "log_q", "log_s", "alpha", "log_rho"]
import corner
import matplotlib.pyplot as plt
hmc_sample = mcmc.get_samples()['param']
corner_params = [r"$t_0^\prime$", r"$\log t_E$", r"$u_0$", r"$\log q$", r"$\log s$", r"$\alpha$", r"$\log \rho$"]
fig = corner.corner(np.array(hmc_sample), labels=corner_params, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig.savefig("example/synthetic_roman/corner_plot_noL.png", bbox_inches="tight")
plt.close()

