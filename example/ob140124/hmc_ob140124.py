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
from microjax.trajectory import dtn_dum_parallax, _get_info_parallax

data = pd.read_csv("example/data/ogle-2014-blg-0124/phot.dat", delim_whitespace=True,header=None, names=["HJD", "mag", "mage", "seeing", "sky"])
data = data[(data.HJD>2.4563e+6)&(data.mage<0.4)]
#data = data[(data.HJD>2.4566e+6)&(data.mage<0.4)]
data["HJD"] -= 2450000
mag0 = 18.0
data["flux"] = 10**(-0.4*(data.mag - mag0))
data["fluxe"] = data.flux * 0.4 * np.log(10) * data.mage

# initial guess
_t_0, _u_0, _t_E = 6.83640951e+03, 2.24211333e-01, 1.33559958e+02 
_s, _q, _alpha = 9.16157288e-01, 5.87559438e-04, 1.00066409e+02
_rho, _pi_E_N, _pi_E_E = 2.44003713e-03, 1.82341182e-01,9.58542572e-02
_fs, _fb = 8.06074085e-01, 8.62216897e-01
_f_sum = _fs + _fb
theta_0 = jnp.array([_t_0, _u_0, _t_E, _s, _q, _alpha, _rho, _pi_E_N, _pi_E_E, _fs, _fb])

from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
coords_deg = (coords.ra.deg, coords.dec.deg)
tref = 6836.0
info_parallax = _get_info_parallax(RA=coords_deg[0], Dec=coords_deg[1], tref=tref)
t_peri, qne0, vne0, xpos, ypos, north, east = info_parallax
#dtn_dum = lambda t, _pi_E_N, _pi_E_E: dtn_dum_parallax(t, _pi_E_N, _pi_E_E, t_peri, qne0, vne0, xpos, ypos, north, east)

def model(t, y, yerr): #, t_peri, qne0, vne0, xpos, ypos, north, east):
    t0 = numpyro.sample("t0", dist.Uniform(_t_0 - 1.0, _t_0 + 1.0))
    u0 = numpyro.sample("u0", dist.Uniform(_u_0 - 0.5, _u_0 + 0.5))
    lntE = numpyro.sample("lntE", dist.Uniform(0, 10))
    tE = jnp.exp(lntE)
    numpyro.deterministic("tE", tE)
    lns = numpyro.sample("lns", dist.Uniform(jnp.log(_s - 0.1), jnp.log(_s + 0.1)))
    lnq = numpyro.sample("lnq", dist.Uniform(jnp.log(_q / 10.0), jnp.log(10.0 * _q))) 
    s = jnp.exp(lns)
    q = jnp.exp(lnq)
    alpha = numpyro.sample("alpha", dist.Uniform(-180, 180))
    alpha_rad = jnp.radians(alpha)
    lnrho = numpyro.sample("rho", dist.Uniform(jnp.log(_rho / 5.0), jnp.log(5.0 * _rho)))
    rho = jnp.exp(lnrho)
    #pi_E_N = numpyro.sample("pi_E_N", dist.Uniform(-1, 1))
    #pi_E_E = numpyro.sample("pi_E_E", dist.Uniform(-1, 1))
    #dtn, dum = dtn_dum_parallax(t, pi_E_N, pi_E_E, t_peri, qne0, vne0, xpos, ypos, north, east)
    k_norm = numpyro.sample("k_norm", dist.Uniform(0.0, 5.0))

    tau = (t - t0)/tE 
    um  = u0 
    #tau = (t - t0)/tE + dtn
    #um  = u0 + dum 
    y1 = -um*jnp.sin(alpha_rad) + tau*jnp.cos(alpha_rad)
    y2 = um*jnp.cos(alpha_rad) + tau*jnp.sin(alpha_rad)
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

    mags = mag_lc(w_points, rho, s=s, q=q, nlenses=2, cubic=True, r_resolution=500, th_resolution=500)
    fs = numpyro.sample("fs", dist.Uniform(_fs - 1.0, _fs + 1.0))
    f_sum = numpyro.sample("f_sum", dist.Uniform(_f_sum - 1.0, _f_sum + 1.0))
    fb = f_sum - fs
    f_model = mags * fs + fb
    numpyro.deterministic("f_model", f_model)
    numpyro.deterministic("fb", fb)

    residuals = y - f_model
    N = len(y)
    loglike = -0.5 * jnp.sum(residuals**2 / yerr**2) - 0.5 * N * jnp.log(2*jnp.pi) - jnp.sum(jnp.log(yerr * k_norm)) 
    numpyro.factor("loglike", loglike)

kernel = numpyro.infer.NUTS(model, forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
rng_key = random.PRNGKey(0)
#mcmc.run(rng_key, data.HJD.values, data.flux.values, data.fluxe.values, t_peri, qne0, vne0, xpos, ypos, north, east)
mcmc.run(rng_key, data.HJD.values, data.flux.values, data.fluxe.values)

mcmc.print_summary()

if(0):
    #params_mm = {'t_0': _t_0, 'u_0': _u_0, 't_E': _t_E, 's': _s, 'q': _q, 'alpha': _alpha, 'rho': _rho}
    params_mm = {'t_0': _t_0, 'u_0': _u_0, 't_E': _t_E, 's': _s, 'q': _q, 'alpha': _alpha, 'rho': _rho, 'pi_E_N': _pi_E_N, 'pi_E_E': _pi_E_E}
    model = mm.Model(parameters=params_mm, coords=coords)
    model.set_default_magnification_method("VBBL")
    time_plot = jnp.linspace(6350,7000,2000)
    amp_mm = model.get_magnification(time_plot)
    plt.plot(time_plot, amp_mm*_fs + _fb, "-", lw=2, color="red", label="MulensModel")
    plt.errorbar(data.HJD, data.flux, yerr=data.fluxe, fmt='.', color='black', label="OGLE-I")
    plt.title("OGLE-2014-BLG-0124")
    plt.xlabel("HJD - 2,450,000")
    plt.ylabel("flux")
    plt.xlim(6650,7000)
    plt.legend()
    plt.savefig("example/ob140124/ob140124_data.pdf")
    plt.show()

