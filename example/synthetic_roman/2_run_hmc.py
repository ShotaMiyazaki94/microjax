import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from model import mag_time, wpoints_time
import pandas as pd
import numpy as np
from microjax.likelihood import linear_chi2, nll_ulens

params_best = jnp.array([ 0., 1.21045229, 0.06,
                         -4.99946803, 0.01283722, 
                         4.156, -2.73612732])

file = pd.read_csv("example/synthetic_roman/mock_data.csv")
time_lc = jnp.array(file.t.values)
flux_lc = jnp.array(file.Flux_obs.values)
fluxe_lc = jnp.array(file.Flux_err.values)
data_input = (time_lc, flux_lc, fluxe_lc)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data, L, init_val):
    times, flux, fluxe = data
    param_base = numpyro.sample('param_base', dist.Uniform(-1*jnp.ones(len(init_val)), jnp.ones(len(init_val))))
    param_true = jnp.dot(L * 10, param_base) + jnp.array(init_val)
    numpyro.deterministic('param', param_true)
    mags = mag_time(times, param_true)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    sigma2_obs = fluxe ** 2
    sigma2_fs = 1e9
    sigma2_fb = 1e9
    nll = nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    numpyro.deterministic("loglike", -nll)
    numpyro.factor("log_likelihood", -nll)

fisher_matrix_np = np.load("example/synthetic_roman/FM_approx.npy")
fisher_cov = jnp.linalg.inv(fisher_matrix_np)
L = jnp.linalg.cholesky(fisher_cov)
init_strategy=numpyro.infer.init_to_value(values={'param_base':jnp.zeros(len(params_best))})
kernel = NUTS(model,
              init_strategy=init_strategy, 
              dense_mass=True, 
              regularize_mass_matrix=True,
              adapt_mass_matrix=True,
              adapt_step_size=True, 
              forward_mode_differentiation=True,
              target_accept_prob=0.8,
              )

mcmc = MCMC(kernel, num_warmup=500, num_samples=10000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data=data_input, L=L, init_val=params_best)
mcmc.print_summary(exclude_deterministic=False)

import arviz as az
import corner
idata = az.from_numpyro(mcmc)
idata.to_netcdf("example/synthetic_roman/mcmc_full.nc")

param_labels = ["t0_diff", "log_tE", "u0", "log_q", "log_s", "alpha", "log_rho"]
import corner
import matplotlib.pyplot as plt
hmc_sample = mcmc.get_samples()['param']
corner_params = [r"$t_0^\prime$", r"$\log t_E$", r"$u_0$", r"$\log q$", r"$\log s$", r"$\alpha$", r"$\log \rho$"]
fig = corner.corner(np.array(hmc_sample), labels=corner_params, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig.savefig("example/synthetic_roman/corner_plot.png", bbox_inches="tight")
plt.close()