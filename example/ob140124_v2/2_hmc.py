import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import optax
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, random, jacfwd
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.trajectory.parallax import compute_parallax, set_parallax, peri_vernal
from microjax.likelihood import linear_chi2, nll_ulens
from model import mag_time
from astropy.coordinates import SkyCoord

coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
RA, Dec = coords.ra.deg, coords.dec.deg
tref = 6836.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)
params_best = np.load("example/ob140124_v2/adam_fwd_params.npy")
data = np.load("example/ob140124_v2/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
data_input = (t_data, flux_data, fluxe_data, parallax_params)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data, L, init_val):
    times, flux, fluxe, parallax_params = data
    log_fac_err = numpyro.sample('log_fac_err', dist.Uniform(-0.5, 0.5))
    fac_err = 10**log_fac_err
    numpyro.deterministic('fac_err', fac_err)
    param_base = numpyro.sample('param_base', dist.Uniform(-1*jnp.ones(len(init_val)), jnp.ones(len(init_val))))
    param_true = jnp.dot(L * 10, param_base) + jnp.array(init_val)
    numpyro.deterministic('param', param_true)
    mags = mag_time(times, param_true, parallax_params)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    sigma2_obs = (fac_err * fluxe) ** 2
    sigma2_fs = 1e9
    sigma2_fb = 1e9
    nll = nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    numpyro.factor("log_likelihood", -nll)

fisher_matrix = np.load("example/ob140124_v2/fisher_matrix_approx.npy")
fisher_cov = jnp.linalg.inv(fisher_matrix)
L = jnp.linalg.cholesky(fisher_cov)
init_strategy=numpyro.infer.init_to_value(values={'param_base':jnp.zeros(len(params_best))})
kernel = NUTS(model,
              init_strategy=init_strategy, 
              dense_mass=True,  # ← Fisher行列は dense
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
idata.to_netcdf("example/ob140124_v2/mcmc_full.nc")
posterior = idata.posterior
print(posterior)

exit(1)


param_labels = ["t0_diff", "log_tE", "u0", "log_q", "log_s", "alpha", "log_rho", "piEN", "piEE"]
import corner
hmc_sample = mcmc.get_samples()['param']
corner_params = [r"$t_0^\prime$", r"$u_0$", r"$\log t_E$", r"$\log q$", r"$\log s$", r"$\alpha$", r"$\log \rho$", r"$\pi_{E, N}$", r"$\pi_{E, E}$"]
fig = corner.corner(np.array(hmc_sample), labels=corner_params, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig.savefig("example/ob140124_v2/corner_plot.png", bbox_inches="tight")
plt.close()