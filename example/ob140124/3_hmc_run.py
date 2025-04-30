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
from microjax.trajectory.parallax import compute_parallax, set_parallax, peri_vernal
from microjax.likelihood import nll_ulens
import corner

data = np.load("example/ob140124/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
t_data = jnp.array(t_data)
flux_data = jnp.array(flux_data)
fluxe_data = jnp.array(fluxe_data)
from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
data_input = (t_data, flux_data, fluxe_data, coords.ra.deg, coords.dec.deg)

param_adam = np.load("example/ob140124/adam_fwd_params.npz")
param_dict = {key: param_adam[key] for key in param_adam.files}
keys = ["t0", "tE", "u0", "log_q", "log_s", "alpha_deg", "log_rho", "piEE", "piEN"]
params_adam_np = np.array([param_dict[key] for key in keys])

@jit
def mag_time(time, params, RA, Dec):
    t0, tE, u0, log_q, log_s, alpha_deg, log_rho, piEE, piEN = params 
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
    y1 = tm*jnp.cos(alpha) - um*jnp.sin(alpha) 
    y2 = tm*jnp.sin(alpha) + um*jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    
    Nlimb = 500
    r_resolution  = 500
    th_resolution = 500
    MAX_FULL_CALLS = 100
    cubic = True
    bins_r = 100
    bins_th = 360
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
    times, flux, fluxe, RA, Dec = data
    param_base = numpyro.sample('param_base', dist.Uniform(-1*jnp.ones(len(init_val)), jnp.ones(len(init_val))))
    param_true = jnp.dot(L * 10, param_base) + jnp.array(init_val)
    numpyro.deterministic('param',param_true)
    mags = mag_time(times, param_true, RA, Dec)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    sigma2_obs = fluxe ** 2
    sigma2_fs = 1e9
    sigma2_fb = 1e9
    nll = nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    numpyro.factor("log_likelihood", -nll)

fisher_matrix = np.load("example/ob140124/fisher_matrix.npy")
fisher_cov = jnp.linalg.inv(fisher_matrix)
L = jnp.linalg.cholesky(fisher_cov + 1e-9 * jnp.eye(fisher_cov.shape[0]) )
param_stddev = jnp.sqrt(jnp.diag(fisher_cov))
print("1 sigma:\n", param_stddev)
print("fisher_cov:", fisher_cov)
print("cholesky:", L)

init_strategy=numpyro.infer.init_to_value(values={'param_base':jnp.zeros(len(params_adam_np))})
kernel = NUTS(model,
              init_strategy=init_strategy, 
              forward_mode_differentiation=True,
              target_accept_prob=0.8,
              #inverse_mass_matrix=fisher_matrix,
              #dense_mass=True,
              )

mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data=data_input, L=L, init_val=params_adam_np)
mcmc.print_summary(exclude_deterministic=False)

import pandas as pd
samples = mcmc.get_samples()
df_samples = pd.DataFrame({k: np.array(v).flatten() for k, v in samples.items()})
df_samples.to_csv("example/ob140124/hmc_samples_log.csv", index=False)

import corner
hmc_sample = mcmc.get_samples()['param']
print(hmc_sample.shape)
#corner_params = ["t0", "u0", "tE", "log_q", "log_s", "alpha", "log_rho", "piEE", "piEN"]
corner_params = [r"$t_0$", r"$u_0$", r"$t_E$", r"$\log q$", r"$\log s$", r"$\alpha$ (deg)", r"$\log \rho$", r"$\pi_{EE}$", r"$\pi_{EN}$"]
fig = corner.corner(np.array(hmc_sample), labels=corner_params, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig.savefig("example/ob140124/hmc_corner_plot_log.png", bbox_inches="tight")
plt.close()