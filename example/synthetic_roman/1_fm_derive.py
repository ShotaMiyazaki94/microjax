import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from model import mag_time, wpoints_time
from microjax.likelihood import linear_chi2 
import pandas as pd
import numpy as np

params_best = jnp.array([ 0., 1.21045229, 0.06,
                         -4.99946803, 0.01283722, 
                         4.156, -2.73612732])

file = pd.read_csv("example/synthetic_roman/mock_data.csv")
time_lc = jnp.array(file.t.values)
flux_lc = jnp.array(file.Flux_obs.values)
fluxe_lc = jnp.array(file.Flux_err.values)

if(0):
    magn = mag_time(time_lc, params_best)
    Fs, Fse, Fb, Fbe, chi2 = linear_chi2(magn, flux_lc, fluxe_lc)
    def model_flux(theta):
        magn = mag_time(time_lc, theta)
        return Fs * magn + Fb
    J = jax.jacfwd(model_flux)(params_best)
    W = (1.0 / fluxe_lc**2)[:, None]
    J = np.array(J)
    W = np.array(W)
    F = J.T @ (W * J)
    np.save("example/synthetic_roman/FM_approx", F)

fisher_matrix_np = np.load("example/synthetic_roman/FM_approx.npy")
print(fisher_matrix_np)
damping = 0.0
fisher_matrix_pd = fisher_matrix_np + damping * np.eye(fisher_matrix_np.shape[0])
fisher_cov = np.linalg.inv(fisher_matrix_pd)
print(fisher_cov)
print(np.sqrt(np.diag(fisher_cov)))
L = jnp.linalg.cholesky(fisher_cov)
print(L)