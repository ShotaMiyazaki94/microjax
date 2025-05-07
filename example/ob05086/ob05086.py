import numpy as np
import matplotlib.pyplot as plt
import corner
import os
import pandas as pd
import MulensModel as mm
import seaborn as sns
plt.rcParams["figure.figsize"] = (18,6)
from matplotlib import rc

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import config
config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import grad, jit, vmap, random

file_name = os.path.join(mm.DATA_PATH, "photometry_files", "OB05086", "starBLG234.6.I.218982.dat")
data = mm.MulensData(file_name=file_name, add_2450000=False)
event_name = 'OGLE-2005-BLG-086'

t, y, yerr = data.time, data.flux, data.err_flux
ybase, ystd = np.percentile(y, 10), np.std(y)
t_dense = jnp.linspace(t[0], t[-1], 1000) # just to make plots nicer

def magnification(t, t0, te, u0):
    u2 = u0**2 + ((t-t0)/te)**2
    u = jnp.sqrt(u2)
    mag = (u2 + 2) / (u*jnp.sqrt(u2 + 4))
    return mag

def model(t, y, yerr):
    t0 = numpyro.sample("t0", dist.Uniform(t[0], t[-1]))
    u0 = numpyro.sample("u0", dist.Uniform(0, 10))
    lnte = numpyro.sample("lnte", dist.Uniform(0, 10))
    te = jnp.exp(lnte)
    numpyro.deterministic("tE", te)
    mag = magnification(t, t0, te, u0)
    mag_dense = magnification(t_dense, t0, te, u0)
    numpyro.deterministic("mag", mag)

    fsum = numpyro.sample("fsum", dist.TruncatedNormal(loc=ybase, scale=ystd, low=ybase-ystd))
    fratio = numpyro.sample("fratio", dist.Uniform(0, 2))
    fs = fsum / (1.+fratio)
    fb = fs * fratio

    fmodel = fb + fs * mag
    numpyro.deterministic("model", fmodel)
    numpyro.deterministic("model_dense", fb + fs * mag_dense)
    numpyro.deterministic("fb", fb)
    numpyro.deterministic("fs", fs)

    k_norm = numpyro.sample("k_norm", dist.Uniform(0.0, 5.0))
    res = y - fmodel
    sigma = k_norm * yerr
    loglike = -0.5 * jnp.sum( (res/(sigma))**2) - jnp.sum(jnp.log(sigma)) #- 0.5 * jnp.log(2*jnp.pi) * len(t)
    numpyro.factor("loglike", loglike)

def make_corner(samples, keys, labs=None):
    params = pd.DataFrame(data=dict(zip(keys, [samples[k] for k in keys])))
    if labs is None:
        labs = [k.replace("_", "") for k in keys]
    return corner.corner(params, labels=labs, show_titles="%.2f")

kernel = numpyro.infer.NUTS(model)
n_sample = 5000
mcmc = numpyro.infer.MCMC(kernel, num_warmup=n_sample, num_samples=n_sample)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, t, y, yerr)
mcmc.print_summary()

samples = mcmc.get_samples()
models = samples['model_dense']
model_mean = jnp.mean(models, axis=0)
model_std = jnp.std(models, axis=0)
res = y - jnp.mean(samples['model'], axis=0)

fig, ax = plt.subplots(2, 1, figsize=(14,8), sharex=True, gridspec_kw={'height_ratios': [2,1]})
ax[0].set_ylabel('flux')
ax[0].plot(t, y, '.', label='data')
ax[0].plot(t_dense, model_mean, '-', color='C1', label='1PL1PS model')
ax[0].fill_between(t_dense, model_mean-model_std, model_mean+model_std, alpha=0.2, color='C1')
ax[0].legend(loc='upper right')
ax[1].errorbar(t, res, yerr=yerr, fmt='o', lw=0, markersize=3)
ax[1].set_xlim(t[0], t[-1])
ax[1].axhline(y=0, color='C1', lw=1)
ax[1].set_xlabel("time")
ax[1].set_ylabel('residual')
plt.tight_layout(pad=0.02)
plt.savefig("example/templete/ob05086_lc.pdf")
plt.close()

fig = make_corner(samples, ['t0', 'u0', 'tE', 'fs', 'fsum'], labs=['$t_0$', '$u_0$', '$t_E$', '$f_s$', '$f_\mathrm{sum}$'])
plt.savefig("example/templete/ob05086_corner.pdf")
plt.close()