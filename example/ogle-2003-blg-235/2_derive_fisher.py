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
import corner

# --- データ読み込み ---
data_moa  = np.load("example/ogle-2003-blg-235/flux_moa.npy")
data_ogle = np.load("example/ogle-2003-blg-235/flux_ogle.npy")
fs_ogle, fb_ogle = 9.072, 2.856
t_moa, flux_moa, fluxe_moa = data_moa[0] - 2450000, data_moa[1], data_moa[2]
t_ogle, flux_ogle, fluxe_ogle = data_ogle[0] - 2450000, data_ogle[1], data_ogle[2]
t_data     = np.hstack([t_moa, t_ogle])
flux_data  = np.hstack([flux_moa, flux_ogle])
fluxe_data = np.hstack([fluxe_moa, fluxe_ogle])
data_input = (t_data, flux_data, fluxe_data)

# --- パラメータ名と順序を定義 ---
param_keys = ["t0", "tE", "u0", "q", "s", "alpha", "rho"]

# --- dict <-> array 変換関数 ---
def dict_to_array(params_dict):
    return jnp.array([params_dict[k] for k in param_keys])

def array_to_dict(params_array):
    return {k: v for k, v in zip(param_keys, params_array)}

# {'alpha': array(3.94610881), 'log_q': array(-2.34634473), 'log_rho': array(-3.99074251), 'log_s': array(-0.05999554), 't0': array(2848.06685328), 'tE': array(61.54353175), 'u0': array(0.13737328)}
params_init_dict = {
    "t0": 2848.16048754,
    "tE": 61.61235588,
    "u0": 0.11760426,
    "q": 10**-2.3609089,
    "s": 10**0.0342508,
    "alpha": 4.00180035,
    "rho": 10**-3.94500971
}
params_init = dict_to_array(params_init_dict)

# --- モデル関数（JAX JIT） ---
@jit
def mag_binary_time(time, theta):
    params = array_to_dict(theta)
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    q, s, alpha = params["q"], params["s"], params["alpha"]
    rho = params["rho"]
    
    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    
    magn = mag_binary(
        w_points, rho, s=s, q=q,
        r_resolution=500, th_resolution=500,
        cubic=True, Nlimb=500,
        bins_r=50, bins_th=120,
        margin_r=1.0, margin_th=1.0,
        MAX_FULL_CALLS=50, chunk_size=10)
    return magn

# --- NLL関数（パラメータはarray） ---
times, fluxs, fluxes = data_input

@jit
def nll_fn(theta_array):
    mags = mag_binary_time(times, theta_array)
    M = jnp.stack([mags - 1.0, jnp.ones_like(mags)], axis=1)
    return nll_ulens(fluxs, M, fluxes**2, jnp.array(1e9), jnp.array(1e9))

# --- Forward-mode Hessian（jacfwd x2） ---
from jax import make_jaxpr
hessian_fn = jacfwd(jacfwd(nll_fn))
#jaxpr = make_jaxpr(hessian_fn)(params_init)
#print(jaxpr)
fisher_matrix = hessian_fn(params_init)
fisher_cov = jnp.linalg.inv(fisher_matrix)
param_stddev = jnp.sqrt(jnp.diag(fisher_cov))
print("1 sigma:\n", param_stddev)

fisher_matrix = np.array(fisher_matrix)
np.save("example/ogle-2003-blg-235/fisher_matrix", fisher_matrix)

if(0):
    mean = np.array(params_init)
    cov = np.array(fisher_cov)
    samples = np.random.multivariate_normal(mean, cov*10, size=2000)

    labels = [r"$t_0$", r"$t_E$", r"$u_0$", r"$q$", r"$s$", r"$\alpha$", r"$\rho$"]
    fig = corner.corner(samples, labels=labels, truths=mean, show_titles=True, title_fmt=".4f")
    fig.savefig("example/ogle-2003-blg-235/hessian_corner_plot.png", bbox_inches="tight")
    plt.show()
