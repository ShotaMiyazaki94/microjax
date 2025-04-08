import numpy as np
import VBBinaryLensing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax.numpy as jnp
import jax
from jax import lax, vmap, jit
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.inverse_ray.cond_extended import test_full
from microjax.multipole import _mag_hexadecapole
from microjax.point_source import critical_and_caustic_curves
from microjax.likelihood import nll_ulens

data_moa  = np.load("example/ogle-2003-blg-235/flux_moa.npy")
data_ogle = np.load("example/ogle-2003-blg-235/flux_ogle.npy")
fs_ogle, fb_ogle = 9.072, 2.856
t_moa, flux_moa, fluxe_moa = data_moa[0] - 2450000, data_moa[1], data_moa[2]
t_ogle, flux_ogle, fluxe_ogle = data_ogle[0] - 2450000, data_ogle[1], data_ogle[2]
t_data     = np.hstack([t_moa, t_ogle])
flux_data  = np.hstack([flux_moa, flux_ogle])
fluxe_data = np.hstack([fluxe_moa, fluxe_ogle])
data_input = (t_data, flux_data, fluxe_data)

@jit
def test_full_binary(time, params):
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    q, s, alpha = params["q"], params["s"], params["alpha"]
    rho = params["rho"]
    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    
    a = 0.5 * s
    e1 = q / (1.0 + q)
    nlenses = 2
    x_cm = a * (1 - q) / (1 + q)
    _params = {"a": a, "e1": e1}
    w_points_shifted = w_points - x_cm
    
    test = test_full(w_points_shifted, rho, nlenses=nlenses, **_params)
    num_full  = jnp.sum(~test)
    return num_full

t0_init    = 2848.040574007477
tE_init    = 61.554606247591636
u0_init    = 0.13243845397676268
q_init     = np.exp(-5.550309923228882)
s_init     = np.exp(0.11379367519592068)
alpha_init = 0.7696166272645534 + jnp.pi
rho_init   = np.exp(-6.939076774112927)
fs_init    = 9.007546647372939
par0 = {'t0': t0_init, 'u0': u0_init, 'tE': tE_init, 
        'rho': rho_init, 'alpha': alpha_init, 's': s_init, 'q': q_init}
for key, value in par0.items():
    print(f"{key}: {value:.5e}")

print("Nfull=", test_full_binary(time=t_data, params=par0))

@jit
def mag_binary_time(time, params):
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    q, s, alpha = params["q"], params["s"], params["alpha"]
    rho = params["rho"]
    
    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    
    Nlimb = 500
    r_resolution  = 200
    th_resolution = 200
    MAX_FULL_CALLS = 100
    
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

def model(data):
    times, flux, fluxe = data
    t0    = numpyro.sample("t0",    dist.Uniform(2840.0, 2855.0))
    tE    = numpyro.sample("tE",    dist.Uniform(0, 100.0))
    u0    = numpyro.sample("u0",    dist.Uniform(0.0, 1.0))
    ln_q  = numpyro.sample("ln_q",  dist.Uniform(-6, -2))
    ln_s  = numpyro.sample("ln_s",  dist.Uniform(-0.5, 0.5))
    alpha = numpyro.sample("alpha", dist.Uniform(0.0, 2 * jnp.pi))
    ln_rho   = numpyro.sample("ln_rho",   dist.Uniform(-5, -2))

    q, s, rho = jnp.exp(ln_q), jnp.exp(ln_s), jnp.exp(ln_rho)
    numpyro.deterministic("q", q)
    numpyro.deterministic("s", s)
    numpyro.deterministic("rho", rho)


    params = {
        "t0": t0, "tE": tE, "u0": u0,
        "q": q, "s": s, "alpha": alpha,
        "rho": rho
    }

    mags = mag_binary_time(times, params)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    sigma2_obs = fluxe ** 2
    sigma2_fs = 1e6
    sigma2_fb = 1e6
    nll = nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    numpyro.factor("log_likelihood", -nll)

from numpyro.infer import init_to_value
t0_init    = 2848.040574007477
tE_init    = 61.554606247591636
u0_init    = 0.13243845397676268
q_init     = np.exp(-5.550309923228882)
s_init     = np.exp(0.11379367519592068)
alpha_init = 0.7696166272645534 + jnp.pi
rho_init   = np.exp(-6.939076774112927)
fs_init    = 9.127274361202149
fb_init    = 2.7986759799613043
init_params = {'t0': t0_init, 'u0': u0_init, 'tE': tE_init, 
               'rho': rho_init, 'alpha': alpha_init, 's': s_init, 'q': q_init}

kernel = NUTS(model, init_strategy=init_to_value(values=init_params),step_size=1e-2, 
              forward_mode_differentiation=True,target_accept_prob=0.8)

mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data=data_input)

import pandas as pd
samples = mcmc.get_samples()
df_samples = pd.DataFrame({k: np.array(v).flatten() for k, v in samples.items()})

df_samples.to_csv("example/ogle-2003-blg-235/hmc_samples.csv", index=False)