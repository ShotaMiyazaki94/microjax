import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)
from astropy.coordinates import SkyCoord
import astropy.units as u

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.extended_source import mag_limb_dark
from microjax.trajectory.parallax import peri_vernal, set_parallax, compute_parallax
from microjax.likelihood import linear_chi2, nll_ulens
from microjax.point_source import mag_point_source
from triple_fit import triple_fit

#params = np.load("example/OGLE-2014-BLG-1722/params_final.npy")
params_init = jnp.array([6.90022772e+03 - 6900.0, jnp.log10(2.32698878e+01), -1.34886439e-01,
                         -3.34134233e+00, -1.25602528e-01, -2.20540555e-01,
                         -3.21033816e+00, 4.71741660e-02, -2.46430115e+00, 
                         4.23139645e-01, 5.50070356e-02, jnp.log10(0.001)], dtype=jnp.float64)

labels = ["t0_diff", "log_tE", "u0", "log_q", "log_s", "alpha", "log_q3", "log_s2", "psi", "piEN", "piEE", "log_rho"]
for i, label in enumerate(labels):
    print(f"{label}: {params_init[i]:.6f}")
coords = "17:55:00.57 -31:28:08.6"
c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg),)
RA = c.ra.deg
Dec = c.dec.deg
tref = 6900.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)

data_moa = np.loadtxt("example/OGLE-2014-BLG-1722/data/Ian2.dat.flux.norm")
data_ogle = np.loadtxt("example/OGLE-2014-BLG-1722/data/OGLE-2014-BLG-1722.dat.flux.norm")

t_moa = data_moa[:, 0]
t_ogle = data_ogle[:, 0]

x1, x2 = 6880, 6893 
x3, x4 = 6898, 6902 

print(f"Initial params: {params_init}")

@partial(jit, static_argnames=("u1", "chunk_size", "MAX_FULL_CALLS"))
def mag_time(time, params, parallax_params, u1=0.0, chunk_size=50, MAX_FULL_CALLS=500):
    t0_diff, log_tE, u0, log_q, log_s, alpha, log_q3, log_s2, psi, piEN, piEE, log_rho = params
    t0 = t0_diff + 6900.0
    tE = 10**log_tE
    q  = 10**log_q
    s  = 10**log_s
    q3 = 10**log_q3
    s2 = 10**log_s2
    rho = 10**log_rho
    dtn, dum = compute_parallax(time, piEN=piEN, piEE=piEE, parallax_params=parallax_params)
    tau = (time - t0) / tE
    um = u0 + dum
    tm = tau + dtn

    y1 = tm * jnp.cos(alpha) - um * jnp.sin(alpha)
    y2 = tm * jnp.sin(alpha) + um * jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=jnp.complex128)

    x1, x2 = 6880, 6893 
    x3, x4 = 6898, 6902 
    full_time = ((time >= x1) & (time <= x2)) | ((time >= x3) & (time <= x4))
    _params = {"q": q, "s": s, "q3": q3, "r3": s2, "psi": psi}
    magn = triple_fit(w_points, rho, full_time, u1=u1, chunk_size=chunk_size, MAX_FULL_CALLS=MAX_FULL_CALLS,
                      r_resolution=1000, th_resolution=1000, Nlimb=500, **_params)
    return magn

def nll_fn(params, data_moa, data_ogle, parallax_params):
    t_moa, flux_moa, fluxe_moa = data_moa[:, 0], data_moa[:, 1], data_moa[:, 2]
    t_ogle, flux_ogle, fluxe_ogle = data_ogle[:, 0], data_ogle[:, 1], data_ogle[:, 2]

    u1_moa = 0.5895
    chunk_moa = 50
    MAX_FULL_moa = 450
    mags_moa = mag_time(t_moa, params, parallax_params, 
                        u1=u1_moa, chunk_size=chunk_moa, MAX_FULL_CALLS=MAX_FULL_moa)
    M_moa = jnp.stack([mags_moa - 1.0, jnp.ones_like(mags_moa)], axis=1)
    nll_moa = nll_ulens(flux_moa, M_moa, fluxe_moa**2, jnp.array(1e9), jnp.array(1e9))

    u1_ogle = 0.5470
    chunk_ogle = 20
    MAX_FULL_ogle = 20
    mags_ogle = mag_time(t_ogle, params, parallax_params, 
                         u1=u1_ogle, chunk_size=chunk_ogle, MAX_FULL_CALLS=MAX_FULL_ogle)
    M_ogle = jnp.stack([mags_ogle - 1.0, jnp.ones_like(mags_ogle)], axis=1)
    nll_ogle = nll_ulens(flux_ogle, M_ogle, fluxe_ogle**2, jnp.array(1e9), jnp.array(1e9))
    return nll_moa + nll_ogle

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data_moa, data_ogle, parallax_params):
    inits = jnp.array([ 0.22772, 1.36679429, -0.13488644, 
                       -3.34134233, -0.12560253, -0.22054055, 
                       -3.21033816, 0.04717417, -2.46430115,  
                       0.42313965, 0.05500704, -3.0], dtype=jnp.float64)
    t0_diff_d = numpyro.sample('t0_diff_d',   dist.Uniform(-0.1,  0.1))
    log_tE_d  = numpyro.sample('log_tE_d',  dist.Uniform(-0.25,  0.25))
    u0_diff   = numpyro.sample('u0_diff',   dist.Uniform(-0.01, 0.01))
    log_q_d   = numpyro.sample('log_q_d',   dist.Uniform(-0.25,  0.25))
    log_s_d   = numpyro.sample('log_s_d',   dist.Uniform(-0.25,  0.25))
    alpha_d   = numpyro.sample('alpha_d',   dist.Uniform(-0.01, 0.01))
    log_q3_d  = numpyro.sample('log_q3_d',  dist.Uniform(-0.25,  0.25))
    log_s2_d  = numpyro.sample('log_s2_d',  dist.Uniform(-0.25,  0.25))
    psi_d     = numpyro.sample('psi_d',     dist.Uniform(-0.01, 0.01))
    piEN      = numpyro.sample('piEN',   dist.Uniform(-0.5, 0.5))
    piEE      = numpyro.sample('piEE',   dist.Uniform(-0.5, 0.5))
    log_rho   = numpyro.sample('log_rho', dist.Uniform(-4.0, -2.0))

    t0_diff = inits[0] + t0_diff_d
    log_tE  = inits[1] + log_tE_d
    u0      = inits[2] + u0_diff
    log_q   = inits[3] + log_q_d
    log_s   = inits[4] + log_s_d
    alpha   = inits[5] + alpha_d
    log_q3  = inits[6] + log_q3_d
    log_s2  = inits[7] + log_s2_d
    psi     = inits[8] + psi_d
    numpyro.deterministic("t0_diff", t0_diff)
    numpyro.deterministic("log_tE",  log_tE)
    numpyro.deterministic("u0",      u0)
    numpyro.deterministic("log_q",   log_q)
    numpyro.deterministic("log_s",   log_s)
    numpyro.deterministic("alpha",   alpha)
    numpyro.deterministic("log_q3",  log_q3)
    numpyro.deterministic("log_s2",  log_s2)
    numpyro.deterministic("psi",     psi)

    params = jnp.array([t0_diff, log_tE, u0, 
                        log_q, log_s, alpha, 
                        log_q3, log_s2, psi, 
                        piEN, piEE, log_rho], dtype=jnp.float64)

    nll = nll_fn(params, data_moa, data_ogle, parallax_params)
    numpyro.factor("log_likelihood", -nll)

# Define the model
init_strategy=numpyro.infer.init_to_median()
kernel = NUTS(model, init_strategy=init_strategy, 
              forward_mode_differentiation=True,
              target_accept_prob=0.8,
              )
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data_moa=data_moa, data_ogle=data_ogle, parallax_params=parallax_params)
mcmc.print_summary(exclude_deterministic=False)

import arviz as az
idata = az.from_numpyro(mcmc)
idata.to_netcdf("example/OGLE-2014-BLG-1722/mcmc_full.nc")

exit(1)


exit(1)

from functools import partial
from jax import jacfwd
nll_fn_fixed = partial(nll_fn, data_moa=data_moa, data_ogle=data_ogle, parallax_params=parallax_params)
hessian_fn = jit(jacfwd(jacfwd(nll_fn_fixed)))
fisher_matrix = hessian_fn(params_init)
np.save("example/OGLE-2014-BLG-1722/fisher_matrix.npy", np.array(fisher_matrix))
fisher_cov = jnp.linalg.pinv(fisher_matrix)
param_stddev = jnp.sqrt(jnp.diag(fisher_cov))
keys = ["t0_diff", "log_tE", "u0", "log_q", "log_s", "alpha", "log_q3", "log_s2", "psi", "piEN", "piEE", "log_rho"]
for key, param, sigma in zip(keys, params_init, param_stddev):
    print(f"{key}: {param:.6f} ± {sigma:.6f}")

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data_moa, data_ogle, parallax_params):
    inits = jnp.array([ 0.22772, 1.36679429, -0.13488644, 
                       -3.34134233, -0.12560253, -0.22054055, 
                       -3.21033816, 0.04717417, -2.46430115,  
                       0.42313965, 0.05500704, -3.0], dtype=jnp.float64)
    t0_diff_d = numpyro.sample('t0_diff_d',   dist.Uniform(-0.1,  0.1))
    log_tE_d  = numpyro.sample('log_tE_d',  dist.Uniform(-0.25,  0.25))
    u0_diff   = numpyro.sample('u0_diff',   dist.Uniform(-0.01, 0.01))
    log_q_d   = numpyro.sample('log_q_d',   dist.Uniform(-0.25,  0.25))
    log_s_d   = numpyro.sample('log_s_d',   dist.Uniform(-0.25,  0.25))
    alpha_d   = numpyro.sample('alpha_d',   dist.Uniform(-0.01, 0.01))
    log_q3_d  = numpyro.sample('log_q3_d',  dist.Uniform(-0.25,  0.25))
    log_s2_d  = numpyro.sample('log_s2_d',  dist.Uniform(-0.25,  0.25))
    psi_d     = numpyro.sample('psi_d',     dist.Uniform(-0.01, 0.01))
    piEN      = numpyro.sample('piEN',   dist.Uniform(-0.5, 0.5))
    piEE      = numpyro.sample('piEE',   dist.Uniform(-0.5, 0.5))
    log_rho   = numpyro.sample('log_rho', dist.Uniform(-4.0, -2.0))

    t0_diff = inits[0] + t0_diff_d
    log_tE  = inits[1] + log_tE_d
    u0      = inits[2] + u0_diff
    log_q   = inits[3] + log_q_d
    log_s   = inits[4] + log_s_d
    alpha   = inits[5] + alpha_d
    log_q3  = inits[6] + log_q3_d
    log_s2  = inits[7] + log_s2_d
    psi     = inits[8] + psi_d
    numpyro.deterministic("t0_diff", t0_diff)
    numpyro.deterministic("log_tE",  log_tE)
    numpyro.deterministic("u0",      u0)
    numpyro.deterministic("log_q",   log_q)
    numpyro.deterministic("log_s",   log_s)
    numpyro.deterministic("alpha",   alpha)
    numpyro.deterministic("log_q3",  log_q3)
    numpyro.deterministic("log_s2",  log_s2)
    numpyro.deterministic("psi",     psi)

    params = jnp.array([t0_diff, log_tE, u0, 
                        log_q, log_s, alpha, 
                        log_q3, log_s2, psi, 
                        piEN, piEE, log_rho], dtype=jnp.float64)

    nll = nll_fn(params, data_moa, data_ogle, parallax_params)
    numpyro.factor("log_likelihood", -nll)

# Define the model
init_strategy=numpyro.infer.init_to_median()
kernel = NUTS(model, init_strategy=init_strategy, 
              forward_mode_differentiation=True,
              target_accept_prob=0.8,
              )
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data_moa=data_moa, data_ogle=data_ogle, parallax_params=parallax_params)
mcmc.print_summary(exclude_deterministic=False)

import arviz as az
idata = az.from_numpyro(mcmc)
idata.to_netcdf("example/OGLE-2014-BLG-1722/mcmc_full.nc")

exit(1)
import blackjax
import tqdm

def logprob_fn(theta, data_moa, data_ogle, parallax_params):
    return -nll_fn(theta, data_moa, data_ogle, parallax_params)

logprob = lambda th: logprob_fn(th, data_moa, data_ogle, parallax_params)

rng_key  = jax.random.PRNGKey(42)
rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
init_pos = params_init

nuts_factory = blackjax.nuts
adapt = blackjax.window_adaptation(nuts_factory, logprob, target_acceptance_rate=0.8)
(state, adaptation_state) = adapt.run(warmup_key, init_pos, num_steps=1000)
step_size = adaptation_state["step_size"]
inv_mass  = adaptation_state["inverse_mass_matrix"]
print(f"Adapted step_size = {step_size:.3g}")
print(f"Diagonal mass matrix shape = {inv_mass.shape}")

kernel = nuts_factory(logprob, step_size, inv_mass)

@jax.jit
def sample_chain(rng_key, state):
    state, _ = kernel.step(rng_key, state)
    return state

n_samples = 2000
sample_keys = jax.random.split(sample_key, n_samples)
states = []

for k in tqdm.tqdm(sample_keys, desc="Sampling"):
    state = sample_chain(k, state)
    states.append(state.position)

samples = jnp.stack(states)
df = pd.DataFrame(samples, columns=labels + ["log_rho"])
print(df.describe(percentiles=[0.16, 0.5, 0.84]))
df.to_csv("example/OGLE-2014-BLG-1722/mcmc_samples.csv", index=False)


from functools import partial
from jax import jacfwd
nll_fn_fixed = partial(nll_fn, data_moa=data_moa, data_ogle=data_ogle, parallax_params=parallax_params)

hessian_fn = jacfwd(jacfwd(nll_fn_fixed))

fisher_matrix = hessian_fn(params_init)
fisher_cov = jnp.linalg.inv(fisher_matrix)
param_stddev = jnp.sqrt(jnp.diag(fisher_cov))
keys = ["t0", "tE", "u0", "log_q", "log_s", "alpha", "log_q3", "log_s2", "psi", "piEN", "piEE", "log_rho"]
for key, param, sigma in zip(keys, params_init, param_stddev):
    print(f"{key}: {param:.6f} ± {sigma:.6f}")

fisher_matrix = np.array(fisher_matrix)
np.save("example/OGLE-2014-BLG-1722/fisher_matrix", fisher_matrix)