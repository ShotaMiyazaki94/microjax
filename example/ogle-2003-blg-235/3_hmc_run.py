import numpy as np
import VBBinaryLensing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax.numpy as jnp
import jax
from jax import jit, jacfwd
jax.config.update("jax_enable_x64", True)

from microjax.inverse_ray.lightcurve import mag_binary
from microjax.likelihood import nll_ulens
import corner

# --- データ読み込み ---
data_moa  = np.load("example/ogle-2003-blg-235/flux_moa.npy")
data_ogle = np.load("example/ogle-2003-blg-235/flux_ogle.npy")
fs_ogle, fb_ogle = 9.072, 2.856
t_moa, flux_moa, fluxe_moa = data_moa[0] - 2450000, data_moa[1], data_moa[2]
t_ogle, flux_ogle, fluxe_ogle = data_ogle[0] - 2450000, data_ogle[1], data_ogle[2]
t_data     = np.hstack([t_moa, t_ogle])
flux_data  = np.hstack([flux_moa, flux_ogle])
fluxe_data = np.hstack([fluxe_moa, fluxe_ogle])
data_input = (t_data, flux_data, fluxe_data)

#exit(1)
# define time function
@jit
def mag_binary_time(time, params):
    t0, tE, u0  = params[0:3]
    log_q, log_s, alpha = params[3:6]
    log_rho = params[6]
    q, s, rho = 10**log_q, 10**log_s, 10**log_rho
    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    
    Nlimb = 500
    r_resolution  = 500
    th_resolution = 500
    MAX_FULL_CALLS = 20
    cubic = True
    bins_r = 50
    bins_th = 120
    margin_r = 1.0
    margin_th= 1.0
    
    magn = mag_binary(w_points, rho, s=s, q=q, r_resolution=r_resolution, 
                      th_resolution=th_resolution,cubic=cubic, Nlimb=Nlimb, 
                      bins_r=bins_r, bins_th=bins_th, margin_r=margin_r, margin_th=margin_th, 
                      MAX_FULL_CALLS=MAX_FULL_CALLS)
    return magn

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data, L, init_val):
    times, flux, fluxe = data
    param_base = numpyro.sample('param_base', dist.Uniform(-1*jnp.ones(len(init_val)), jnp.ones(len(init_val))))
    param_true = jnp.dot(L * 10, param_base) + jnp.array(init_val)
    numpyro.deterministic('param',param_true)
    mags = mag_binary_time(times, param_true)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    sigma2_obs = fluxe ** 2
    sigma2_fs = 1e9
    sigma2_fb = 1e9
    nll = nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    numpyro.factor("log_likelihood", -nll)

params_init = np.load("example/ogle-2003-blg-235/adam_fwd_params.npz")
print(list(params_init.keys()))
params_init = jnp.array([params_init["t0"], params_init["tE"], params_init["u0"],
                         params_init["log_q"], params_init["log_s"], 
                         params_init["alpha"], params_init["log_rho"]])
print(params_init)
fisher_matrix = np.load("example/ogle-2003-blg-235/fisher_matrix_log.npy")
fisher_cov = jnp.linalg.inv(fisher_matrix)
L = jnp.linalg.cholesky(fisher_cov + 1e-9 * jnp.eye(fisher_cov.shape[0]) )
param_stddev = jnp.sqrt(jnp.diag(fisher_cov))
print("1 sigma:\n", param_stddev)
print("fisher_cov:", fisher_cov)
print("cholesky:", L)

from numpyro.infer import init_to_value
init_strategy=numpyro.infer.init_to_value(values={'param_base':jnp.zeros(len(params_init))})
kernel = NUTS(model,
              init_strategy=init_strategy, 
              forward_mode_differentiation=True,
              target_accept_prob=0.8,
              #inverse_mass_matrix=fisher_matrix,
              #dense_mass=True,
              )

mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data=data_input, L=L, init_val=params_init)
mcmc.print_summary(exclude_deterministic=False)

import pandas as pd
samples = mcmc.get_samples()
df_samples = pd.DataFrame({k: np.array(v).flatten() for k, v in samples.items()})
df_samples.to_csv("example/ogle-2003-blg-235/hmc_samples_log.csv", index=False)

import corner
hmc_sample = mcmc.get_samples()['param']
print(hmc_sample.shape)
corner_params = ["t0", "u0", "tE", "log_q", "log_s", "alpha", "log_rho"]
fig = corner.corner(np.array(hmc_sample), labels=corner_params, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig.savefig("example/ogle-2003-blg-235/hmc_corner_plot_log.png", bbox_inches="tight")
plt.close()