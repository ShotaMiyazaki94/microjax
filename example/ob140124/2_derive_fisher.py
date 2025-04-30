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
from microjax.trajectory.parallax import compute_parallax, set_parallax, peri_vernal
import corner

data = np.load("example/ob140124/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
t_data = jnp.array(t_data)
flux_data = jnp.array(flux_data)
fluxe_data = jnp.array(fluxe_data)
from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
data_input = (t_data, flux_data, fluxe_data, coords.ra.deg, coords.dec.deg)

param_keys = ["t0", "tE", "u0", "log_q", "log_s", "alpha_deg", "log_rho"]

param_adam = np.load("example/ob140124/adam_fwd_params.npz")
param_dict = {key: param_adam[key] for key in param_adam.files}
for key in param_adam.files:
    print(f"{key}:", param_adam[key])

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

    # convert (tn, u0) -> (y1, y2)
    y1 = tm*jnp.cos(alpha) - um*jnp.sin(alpha) 
    y2 = tm*jnp.sin(alpha) + um*jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

    magn = mag_binary(
        w_points, rho=rho, s=s, q=q,
        r_resolution=500, th_resolution=500,
        cubic=True, Nlimb=500, bins_r=100, bins_th=360,
        margin_r=1.0, margin_th=1.0, MAX_FULL_CALLS=100, chunk_size=1,
    )
    return magn

times, fluxs, fluxes, RA, Dec = data_input

@jit
def nll_fn(theta_array, RA, Dec):
    mags = mag_time(times, theta_array, RA, Dec)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    return nll_ulens(fluxs, M, fluxes**2, jnp.array(1e9), jnp.array(1e9))

from jax import make_jaxpr
from functools import partial
nll_fn_fixed = partial(nll_fn, RA=RA, Dec=Dec)
hessian_fn = jacfwd(jacfwd(nll_fn_fixed))

fisher_matrix = hessian_fn(params_adam_np)
fisher_cov = jnp.linalg.inv(fisher_matrix)
param_stddev = jnp.sqrt(jnp.diag(fisher_cov))

for key, param, sigma in zip(keys, params_adam_np, param_stddev):
    print(f"{key}: {param:.6f} ± {sigma:.6f}")

fisher_matrix = np.array(fisher_matrix)
np.save("example/ob140124/fisher_matrix", fisher_matrix)

if(1):
    mean = np.array(params_adam_np)
    cov = np.array(fisher_cov)
    samples = np.random.multivariate_normal(mean, cov, size=5000)

    labels = [r"$t_0$", r"$t_E$", r"$u_0$", r"$\log{q}$", r"$\log{s}$", r"$\alpha$", r"$\log{\rho}$", r"$\pi_{E,E}$", r"$\pi_{E,N}$"]
    fig = corner.corner(samples, labels=labels, truths=mean, show_titles=True, title_fmt=".4f")
    fig.savefig("example/ob140124/hessian_corner_plot.png", bbox_inches="tight")
    plt.show()