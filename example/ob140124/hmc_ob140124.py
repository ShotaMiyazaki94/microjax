import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MulensModel as mm
plt.rcParams["figure.figsize"] = (12,5)
import seaborn as sns
sns.set_theme(style='ticks', font_scale=1.2,)
from matplotlib import rc
#rc('text', usetex=True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import config 
config.update('jax_enable_x64', True) 
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from microjax.inverse_ray.lightcurve import mag_lc
from microjax.inverse_ray.extended_source import mag_uniform
from microjax.trajectory import dtn_dum_parallax, _get_info_parallax

data = pd.read_csv("example/data/ogle-2014-blg-0124/phot.dat", 
                   delim_whitespace=True,header=None, names=["HJD", "mag", "mage", "seeing", "sky"])
data["HJD"] -= 2450000
data = data[data.HJD>6600]
#data = data[(data.HJD>2.4566e+6)&(data.mage<0.4)]
mag0 = 18.0
data["flux"] = 10**(-0.4*(data.mag - mag0))
data["fluxe"] = data.flux * 0.4 * np.log(10) * data.mage

# initial guess
_t_0, _u_0, _t_E = 6.83640951e+03, 2.24211333e-01, 1.33559958e+02 
_s, _q, _alpha = 9.16157288e-01, 5.87559438e-04, 1.00066409e+02
_rho, _pi_E_N, _pi_E_E = 2.44003713e-03, 1.82341182e-01,9.58542572e-02
_fs, _fb = 8.06074085e-01, 8.62216897e-01
_f_sum = _fs + _fb
#theta_0 = jnp.array([_t_0, _u_0, _t_E, _s, _q, _alpha, _rho, _pi_E_N, _pi_E_E, _fs, _fb])

def model(t, y, yerr):
    t0 = numpyro.sample("t0", dist.Uniform(_t_0 - 1.0, _t_0 + 1.0))
    u0 = numpyro.sample("u0", dist.Uniform(_u_0 - 0.5, _u_0 + 0.5))
    lntE = numpyro.sample("lntE", dist.Uniform(0, 10))
    tE = jnp.exp(lntE)
    numpyro.deterministic("tE", tE)
    lns = numpyro.sample("lns", dist.Uniform(jnp.log(_s - 0.1), jnp.log(_s + 0.1)))
    lnq = numpyro.sample("lnq", dist.Uniform(jnp.log(_q / 10.0), jnp.log(10.0 * _q))) 
    s = jnp.exp(lns)
    q = jnp.exp(lnq)
    numpyro.deterministic("q", q)
    numpyro.deterministic("s", s) 
    alpha = numpyro.sample("alpha", dist.Uniform(_alpha*0.99, 1.01*_alpha))
    alpha_rad = jnp.radians(alpha)
    lnrho = numpyro.sample("lnrho", dist.Uniform(jnp.log(_rho / 5.0), jnp.log(5.0 * _rho)))
    rho = jnp.exp(lnrho)
    numpyro.deterministic("rho", rho)
    k_norm = numpyro.sample("k_norm", dist.Uniform(0.5, 3.0))

    tau = (t - t0)/tE 
    um  = u0 
    y1 = -um*jnp.sin(alpha_rad) + tau*jnp.cos(alpha_rad)
    y2 = um*jnp.cos(alpha_rad) + tau*jnp.sin(alpha_rad)
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    def chunked_vmap(func, data, chunk_size):
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            results.append(jit(vmap(func))(chunk))
        return jnp.concatenate(results)
    #mag_f = lambda w: mag_uniform(w, rho, q=q, s=s, r_resolution=500, th_resolution=500)
    #chunk_size = 100
    #mags = chunked_vmap(mag_f, w_points, chunk_size)
    #mags = vmap(mag_func)(w_points)
    mags = mag_lc(w_points, rho, s=s, q=q, nlenses=2, cubic=True, r_resolution=1000, th_resolution=1000)
    fs = numpyro.sample("fs", dist.Uniform(_fs - 1.0, _fs + 1.0))
    f_sum = numpyro.sample("f_sum", dist.Uniform(_f_sum - 1.0, _f_sum + 1.0))
    fb = f_sum - fs
    f_model = mags * fs + fb
    numpyro.deterministic("f_model", f_model)
    numpyro.deterministic("fb", fb)

    residuals = y - f_model
    N = len(y)
    sigma = yerr * k_norm 
    loglike = -0.5 * jnp.sum((residuals/sigma)**2) - jnp.sum(jnp.log(sigma)) # - 0.5 * N * jnp.log(2*jnp.pi) 
    #loglike = -0.5 * jnp.sum((residuals/sigma)**2) - 0.5 * N * jnp.log(2*jnp.pi) - jnp.sum(jnp.log(sigma)) 
    numpyro.factor("loglike", loglike)

init_params = {'t0': _t_0, 'lntE': jnp.log(_t_E), 'u0': _u_0, 
               'lns': jnp.log(_s), 'lnq': jnp.log(_q), 
               'alpha': _alpha, 'lnrho': jnp.log(_rho),
               'k_norm': 1.0, 'fs': _fs, 'f_sum': _f_sum}
kernel = numpyro.infer.NUTS(model, forward_mode_differentiation=True, init_strategy=numpyro.infer.init_to_value(values=init_params))
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, data.HJD.values, data.flux.values, data.fluxe.values)
mcmc.print_summary()
sample = pd.DataFrame(mcmc.get_samples(group_by_chain=False))
sample.to_csv("example/ob140124/output.csv", index=False)
