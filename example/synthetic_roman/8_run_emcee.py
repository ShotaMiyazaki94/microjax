import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from model import mag_time, wpoints_time
import pandas as pd
import numpy as np
from microjax.likelihood import linear_chi2, nll_ulens
import emcee, arviz as az

file = pd.read_csv("example/synthetic_roman/mock_data.csv")
time_lc  = jnp.array(file.t.values)
flux_lc  = jnp.array(file.Flux_obs.values)
fluxe_lc = jnp.array(file.Flux_err.values)
data = (time_lc, flux_lc, fluxe_lc)

params_best = np.load("example/synthetic_roman/adam_fwd_params.npy")
mu     = params_best                                # shape (7,)
Fisher = np.load("example/synthetic_roman/FM_approx_v2.npy")
Fisher_cov = jnp.linalg.inv(Fisher)
sigma = 30 * jnp.sqrt(jnp.diag(Fisher_cov)) 

@jax.jit
def _loglike(param, time_lc, flux_lc, fluxe_lc):
    mags = mag_time(time_lc, param)
    M    = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    nll = nll_ulens(flux_lc, M,fluxe_lc**2, 1e9, 1e9)
    return -nll

def log_prob_z(z, mu, sigma, data):
    lp_z = -0.5 * jnp.dot(z, z)       # 標準正規 prior
    if not jnp.isfinite(lp_z):
        return -jnp.inf
    param_true = mu + sigma * z
    ll = _loglike(param_true, *data).block_until_ready()
    ll = jnp.asarray(ll)
    return ll + lp_z

ndim      = len(mu)
nwalkers  = 40
rng       = np.random.default_rng(42)
z0        = rng.normal(size=(nwalkers, ndim)) * 0.1

sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                log_prob_z,
                                args=(mu, sigma, data))

nsteps     = 45000
nburn_frac = 0.2
sampler.run_mcmc(z0, nsteps, progress=True)

samples_z = sampler.get_chain(discard=0, flat=False)
samples_param = mu + sigma * samples_z  # same shape
#np.save("example/synthetic_roman/samples_z.npy",      samples_z)
#np.save("example/synthetic_roman/samples_param.npy",  samples_param)
try:
    tau = sampler.get_autocorr_time(discard=int(nburn_frac * nsteps))
    ess = samples_param.size / tau.sum()       
    print("τ (per dim) =", tau)
    print("ESS (total) ≈", ess)
except emcee.autocorr.AutocorrError as e:
    print("AutocorrError:", e)
    print("ESS skip...")
idata = az.from_dict(
    posterior={f"param_{i}": samples_param[:, :, i]        # (chain, draw)
               for i in range(samples_param.shape[2])}
)
path_nc = "example/synthetic_roman/emcee_full.nc"
idata.to_netcdf(path_nc)
print(f"Save: {path_nc}")