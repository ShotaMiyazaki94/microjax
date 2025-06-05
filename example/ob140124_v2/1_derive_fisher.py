import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import optax
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, random, jacfwd
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.trajectory.parallax import compute_parallax, set_parallax, peri_vernal
from microjax.likelihood import linear_chi2
from model import mag_time
from astropy.coordinates import SkyCoord

coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
RA, Dec = coords.ra.deg, coords.dec.deg
tref = 6836.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)
params_best = np.load("example/ob140124_v2/adam_fwd_params.npy")
data = np.load("example/ob140124_v2/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
data_input = (t_data, flux_data, fluxe_data, parallax_params)

if(0):
    magn = mag_time(t_data, params_best, parallax_params=parallax_params)
    Fs, Fse, Fb, Fbe, chi2 = linear_chi2(magn, flux_data, fluxe_data)
    def model_flux(theta):
        magn = mag_time(t_data, theta, parallax_params=parallax_params)
        return Fs * magn + Fb
    
    J = jax.jacfwd(model_flux)(params_best)
    W = (1.0 / fluxe_data**2)[:, None]
    F = J.T @ (W * J)
    #eig = np.linalg.eigvalsh(np.asarray(F))
    np.save("example/ob140124_v2/fisher_matrix_approx", F)

if(0):
    @jit
    def nll_fn(params, t_data, flux_data, fluxe_data, parallax_params):
        magn = mag_time(t_data, params, parallax_params=parallax_params, chunk_size=10)
        Fs, Fse, Fb, Fbe, chi2 = linear_chi2(magn, flux_data, fluxe_data)
        return 0.5 * chi2
    nll_fixed = partial(nll_fn, t_data=t_data, flux_data=flux_data, fluxe_data=fluxe_data, parallax_params=parallax_params)
    hessian_fn = jit(jacfwd(jacfwd(nll_fixed)))
    params_best_jax = jnp.array(params_best)
    fisher_matrix = hessian_fn(params_best_jax)
    fisher_matrix_np = np.asarray(fisher_matrix)
    np.save("example/ob140124_v2/fisher_matrix", fisher_matrix_np)

#fisher_matrix_np = np.load("example/ob140124_v2/fisher_matrix.npy")
fisher_matrix_np = np.load("example/ob140124_v2/fisher_matrix_approx.npy")
print(fisher_matrix_np)
damping = 0.0
fisher_matrix_pd = fisher_matrix_np + damping * np.eye(fisher_matrix_np.shape[0])
fisher_cov = np.linalg.inv(fisher_matrix_pd)
print(fisher_cov)
print(np.sqrt(np.diag(fisher_cov)))
#eigvals = np.linalg.eigvalsh(fisher_matrix_pd)
#print("Fisher matrix eigenvalues:", eigvals)



if(0):
    magn = mag_time(t_data, params_best, parallax_params=parallax_params)
    Fs, Fse, Fb, Fbe, chi2 = linear_chi2(magn, flux_data, fluxe_data)
    plt.figure(figsize=(12, 6))
    plt.plot(t_data, flux_data, "o", markersize=1, label="Data")
    plt.plot(t_data, magn*Fs + Fb, "r-", label="Model")
    plt.xlabel("Time (HJD - 2450000)")
    plt.ylabel("Flux (normalized)")
    plt.xlim(6600, 7000)
    plt.title("OGLE-2014-BLG-1722 Light Curve")
    plt.legend()
    plt.savefig("example/ob140124_v2/light_curve.png", dpi=300)