import os
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"
#os.environ["XLA_PYTHON_CLIENT_TRACE_ALLOCATOR"] = "1"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import jax
from jax import jit, jacfwd
jax.config.update("jax_enable_x64", True)
import pandas as pd

from microjax.inverse_ray.lightcurve import mag_binary
from microjax.likelihood import nll_ulens
import corner

data = pd.read_csv("example/synthetic_roman/mock_data.csv")
t_data = jnp.array(data["t"].values)
flux_data = jnp.array(data["Flux_obs"].values)
fluxe_data = jnp.array(data["Flux_err"].values)
data_input = (t_data, flux_data, fluxe_data)

@jit
def mag_time(time, params):
    t0, tE, u0, log_q, log_s, alpha, log_rho = params
    q = 10**log_q
    s = 10**log_s
    rho = 10**log_rho
    tau = (time - t0)/tE
    y1 = tau*jnp.cos(alpha) - u0*jnp.sin(alpha) 
    y2 = tau*jnp.sin(alpha) + u0*jnp.cos(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    magns = mag_binary(w_points, rho, s=s, q=q, r_resolution=1000, th_resolution=1000, chunk_size=50)
    return magns

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data, init_val):
    times, flux, fluxe = data
    t0 = numpyro.sample("t0", dist.Normal(init_val[0], 0.1))
    tE = numpyro.sample("tE", dist.Normal(init_val[1], 1.0))
    u0 = numpyro.sample("u0", dist.Normal(init_val[2], 0.1))
    log_q = numpyro.sample("log_q", dist.Normal(init_val[3], 0.1))
    log_s = numpyro.sample("log_s", dist.Normal(init_val[4], 0.1))
    alpha = numpyro.sample("alpha", dist.Normal(init_val[5], 0.1))
    log_rho = numpyro.sample("log_rho", dist.Normal(init_val[6], 0.1))
    params = jnp.array([t0, tE, u0, log_q, log_s, alpha, log_rho])
    mags = mag_time(times, params)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    sigma2_obs = fluxe ** 2
    sigma2_fs = 1e9
    sigma2_fb = 1e9
    nll = nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    numpyro.factor("log_likelihood", -nll)

init_val = jnp.array([0.0, 25.52, 0.063, 
                      jnp.log10(7.28e-05), jnp.log10(0.947), 
                      2.34, jnp.log10(1.60e-03)])
init_strategy=numpyro.infer.init_to_value(values={"t0": init_val[0],
                                                  "tE": init_val[1],
                                                  "u0": init_val[2],
                                                  "log_q": init_val[3],
                                                  "log_s": init_val[4],
                                                  "alpha": init_val[5],
                                                  "log_rho": init_val[6]})
kernel = NUTS(model, init_strategy=init_strategy,
              forward_mode_differentiation=True,
              target_accept_prob=0.75)

mcmc = MCMC(kernel, num_warmup=500, num_samples=3000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data=data_input, init_val=init_val)
mcmc.print_summary(exclude_deterministic=False)

samples = mcmc.get_samples()
print("Acceptance rate:", mcmc.get_acceptance_rate())
corner.corner(samples, labels=["t0", "tE", "u0", "log_q", "log_s", "alpha", "log_rho"])
import matplotlib.pyplot as plt
plt.savefig("example/synthetic_roman/corner_plot.png", bbox_inches='tight')
plt.close()
# Save the samples to a CSV file
samples_df = pd.DataFrame(samples)
samples_df.to_csv("example/synthetic_roman/mcmc_samples.csv", index=False)
# Save the MCMC results
mcmc.save("example/synthetic_roman/mcmc_results.pkl")
# Save the model parameters
params_df = pd.DataFrame({
    "t0": samples["t0"],
    "tE": samples["tE"],
    "u0": samples["u0"],
    "q": 10**samples["log_q"],
    "s": 10**samples["log_s"],
    "alpha": samples["alpha"],
    "rho": 10**samples["log_rho"]
})
params_df.to_csv("example/synthetic_roman/model_params.csv", index=False)


    
    