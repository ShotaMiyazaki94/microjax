import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax
import jax.numpy as jnp
from jax import jit, random
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.likelihood import nll_ulens

data_moa  = np.load("example/ogle-2003-blg-235/flux_moa.npy")
data_ogle = np.load("example/ogle-2003-blg-235/flux_ogle.npy")
#data_moa  = np.load("flux_moa.npy")
#data_ogle = np.load("flux_ogle.npy")
t_moa, flux_moa, fluxe_moa = data_moa[0] - 2450000, data_moa[1], data_moa[2]
t_ogle, flux_ogle, fluxe_ogle = data_ogle[0] - 2450000, data_ogle[1], data_ogle[2]
t_data     = jnp.hstack([t_moa, t_ogle])
flux_data  = jnp.hstack([flux_moa, flux_ogle])
fluxe_data = jnp.hstack([fluxe_moa, fluxe_ogle])
data_input = (t_data, flux_data, fluxe_data)

@jit
def mag_binary_time(time, params):
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    q, s, alpha = params["q"], params["s"], params["alpha"]
    rho = params["rho"]

    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

    magn = mag_binary(
        w_points, rho, s=s, q=q,
        r_resolution=200, th_resolution=200,
        cubic=True, Nlimb=500, bins_r=50, bins_th=120,
        margin_r=1.0, margin_th=1.0, MAX_FULL_CALLS=20
    )
    return magn

import numpyro
import numpyro.distributions as dist
def model_prob(data):
    times, flux, fluxe = data
    t0        = numpyro.sample("t0",         dist.Normal(2848.04, 100.0))
    tE        = numpyro.sample("tE",         dist.Normal(61.55, 100.0))
    u0        = numpyro.sample("u0",         dist.Uniform(0.0, 1.0))
    alpha     = numpyro.sample("alpha",      dist.Uniform(0.0, 2 * jnp.pi))
    log_q     = numpyro.sample("log_q",      dist.Uniform(-5, -1))
    log_s     = numpyro.sample("log_s",      dist.Uniform(-0.5, 0.5))
    log_rho   = numpyro.sample("log_rho",    dist.Uniform(-5, -1))
    q, s, rho = 10**(log_q), 10**(log_s), 10**(log_rho)
    numpyro.deterministic("q", q)
    numpyro.deterministic("s", s)
    numpyro.deterministic("rho", rho)

    params = {"t0": t0, "tE": tE, "u0": u0, "q": q, "s": s, "alpha": alpha, "rho": rho}
    mags = mag_binary_time(times, params)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    nll = nll_ulens(flux, M, fluxe**2, 1e9, 1e9)
    numpyro.factor("log_likelihood", -nll)

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoBNAFNormal, AutoLowRankMultivariateNormal
from numpyro.infer.initialization import init_to_value
import numpyro.optim as optim
params_init = {
    "t0": 2848.16048754,
    "tE": 61.61235588,
    "u0": 0.11760426,
    "log_q": -2.3609089,
    "log_s": 0.0342508,
    "alpha": 4.00180035,
    "log_rho": -3.94500971
}

#guide = AutoBNAFNormal(model_prob, init_loc_fn=init_to_value(values=params_init))
#guide = AutoLowRankMultivariateNormal(model_prob, init_loc_fn=init_to_value(values=params_init))
guide = AutoMultivariateNormal(model_prob, init_loc_fn=init_to_value(values=params_init))
optimizer = optim.Adam(1e-3)
svi = SVI(model_prob, guide, optimizer, loss=Trace_ELBO())

num_steps = 5000
rng_key = random.PRNGKey(0)
svi_result = svi.run(rng_key, num_steps, data=data_input, forward_mode_differentiation=True)

params_np = {k: np.array(v) for k, v in svi_result.params.items()}
print(params_np.keys())
np.savez("example/ogle-2003-blg-235/svi_params.npz", **params_np)

from numpyro.infer import Predictive
predictive = Predictive(guide, params=svi_result.params, num_samples=5000)
posterior_samples = predictive(random.PRNGKey(1), data=data_input)
#print_summary(posterior_samples)
posterior_samples.keys()

from numpyro.diagnostics import print_summary
import arviz as az
import pandas as pd

labels = ["t0", "u0", "tE", "log_q", "log_s", "alpha", "log_rho"]
df_posterior = pd.DataFrame({k: posterior_samples[k] for k in labels})
df_posterior.to_csv("example/ogle-2003-blg-235/svi_samples.csv", index=False)

import corner
corner_params = ["t0", "u0", "tE", "log_q", "log_s", "alpha", "log_rho"]
samples = df_posterior[corner_params].copy()
fig = corner.corner(samples, labels=corner_params, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 12})
fig.suptitle("Posterior Distributions (SVI)", fontsize=14)
fig.savefig("example/ogle-2003-blg-235/svi_corner_plot.png", bbox_inches="tight")
plt.show()

losses = svi_result.losses
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("ELBO loss")
plt.title("SVI Loss over Steps")
plt.grid(True)
plt.savefig("example/ogle-2003-blg-235/svi_loss_trace.png", bbox_inches="tight")
plt.show()

