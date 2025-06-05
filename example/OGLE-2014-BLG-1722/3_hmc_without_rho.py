import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)
from astropy.coordinates import SkyCoord
import astropy.units as u

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.extended_source import mag_limb_dark
from microjax.trajectory.parallax import peri_vernal, set_parallax, compute_parallax
from microjax.likelihood import linear_chi2, nll_ulens
from microjax.point_source import mag_point_source
from triple_fit import triple_fit

#params = np.load("example/OGLE-2014-BLG-1722/params_final.npy")
params_init = jnp.array([6.90022772e+03 - 6900.0, jnp.log10(2.32698878e+01), -1.34886439e-01,
                         -3.34134233e+00, -1.25602528e-01, -2.20540555e-01,
                         -3.21033816e+00, 4.71741660e-02, -2.46430115e+00, 
                         4.23139645e-01, 5.50070356e-02], dtype=jnp.float64)

labels = ["t0_diff", "log_tE", "u0", "log_q", "log_s", "alpha", "log_q3", "log_s2", "psi", "piEN", "piEE"]
for i, label in enumerate(labels):
    print(f"{label}: {params_init[i]:.6f}")
coords = "17:55:00.57 -31:28:08.6"
c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg),)
RA = c.ra.deg
Dec = c.dec.deg
tref = 6900.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)

data_moa = np.loadtxt("example/OGLE-2014-BLG-1722/data/Ian2.dat.flux.norm")
data_ogle = np.loadtxt("example/OGLE-2014-BLG-1722/data/OGLE-2014-BLG-1722.dat.flux.norm")

t_moa = data_moa[:, 0]
t_ogle = data_ogle[:, 0]

x1, x2 = 6880, 6893 
x3, x4 = 6898, 6902 

print(f"Initial params: {params_init}")

@jax.jit
def mag_time(time, params, parallax_params):
    t0_diff, log_tE, u0, log_q, log_s, alpha, log_q3, log_s2, psi, piEN, piEE = params
    t0 = t0_diff + 6900.0
    tE = 10**log_tE
    q  = 10**log_q
    s  = 10**log_s
    q3 = 10**log_q3
    s2 = 10**log_s2
    dtn, dum = compute_parallax(time, piEN=piEN, piEE=piEE, parallax_params=parallax_params)
    tau = (time - t0) / tE
    um = u0 + dum
    tm = tau + dtn

    y1 = tm * jnp.cos(alpha) - um * jnp.sin(alpha)
    y2 = tm * jnp.sin(alpha) + um * jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=jnp.complex128)

    _params = {"q": q, "s": s, "q3": q3, "r3": s2, "psi": psi}
    magn = mag_point_source(w_points, nlenses=3, **_params)
    return magn

def nll_fn(params, data_moa, data_ogle, parallax_params):
    t_moa, flux_moa, fluxe_moa = data_moa[:, 0], data_moa[:, 1], data_moa[:, 2]
    t_ogle, flux_ogle, fluxe_ogle = data_ogle[:, 0], data_ogle[:, 1], data_ogle[:, 2]

    mags_moa = mag_time(t_moa, params, parallax_params) 
    M_moa = jnp.stack([mags_moa - 1.0, jnp.ones_like(mags_moa)], axis=1)
    nll_moa = nll_ulens(flux_moa, M_moa, fluxe_moa**2, jnp.array(1e9), jnp.array(1e9))

    mags_ogle = mag_time(t_ogle, params, parallax_params) 
    M_ogle = jnp.stack([mags_ogle - 1.0, jnp.ones_like(mags_ogle)], axis=1)
    nll_ogle = nll_ulens(flux_ogle, M_ogle, fluxe_ogle**2, jnp.array(1e9), jnp.array(1e9))
    return nll_moa + nll_ogle

#from functools import partial
#nll_fn_fixed = partial(nll_fn, data_moa=data_moa, data_ogle=data_ogle, parallax_params=parallax_params)
#hessian_fn = jax.jit(jax.hessian(nll_fn_fixed))
#epsilon = 1e-4
#fisher_matrix = hessian_fn(params_init)
#fisher_matrix_reg = fisher_matrix + epsilon * jnp.eye(fisher_matrix.shape[0])
#fisher_cov = jnp.linalg.inv(fisher_matrix_reg)
#inv_mass_matrix = jnp.array(fisher_matrix_reg)
#np.save("example/OGLE-2014-BLG-1722/fisher_matrix_norho", np.array(fisher_matrix))

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(data_moa, data_ogle, parallax_params):
    inits = jnp.array([ 0.22772, 1.36679429, -0.13488644, 
                       -3.34134233, -0.12560253, -0.22054055, 
                       -3.21033816, 0.04717417, -2.46430115,  
                       0.42313965, 0.05500704], dtype=jnp.float64)
    t0_diff_d = numpyro.sample('t0_diff_d',   dist.Uniform(-0.1,  0.1))
    log_tE_d  = numpyro.sample('log_tE_d',  dist.Uniform(-0.25,  0.25))
    u0_diff   = numpyro.sample('u0_diff',   dist.Uniform(-0.01, 0.01))
    log_q_d   = numpyro.sample('log_q_d',   dist.Uniform(-0.5,  0.5))
    log_s_d   = numpyro.sample('log_s_d',   dist.Uniform(-0.5,  0.5))
    alpha_d   = numpyro.sample('alpha_d',   dist.Uniform(-0.1, 0.1))
    log_q3_d  = numpyro.sample('log_q3_d',  dist.Uniform(-0.5,  0.5))
    log_s2_d  = numpyro.sample('log_s2_d',  dist.Uniform(-0.5,  0.5))
    psi_d     = numpyro.sample('psi_d',     dist.Uniform(-0.1, 0.1))
    piEN      = numpyro.sample('piEN',   dist.Uniform(-1.0, 1.0))
    piEE      = numpyro.sample('piEE',   dist.Uniform(-1.0, 1.0))

    t0_diff = inits[0] + t0_diff_d
    log_tE  = inits[1] + log_tE_d
    u0      = inits[2] + u0_diff
    log_q   = inits[3] + log_q_d
    log_s   = inits[4] + log_s_d
    alpha   = inits[5] + alpha_d
    log_q3  = inits[6] + log_q3_d
    log_s2  = inits[7] + log_s2_d
    psi     = inits[8] + psi_d
    numpyro.deterministic("t0_diff", t0_diff)
    numpyro.deterministic("log_tE",  log_tE)
    numpyro.deterministic("u0",      u0)
    numpyro.deterministic("log_q",   log_q)
    numpyro.deterministic("log_s",   log_s)
    numpyro.deterministic("alpha",   alpha)
    numpyro.deterministic("log_q3",  log_q3)
    numpyro.deterministic("log_s2",  log_s2)
    numpyro.deterministic("psi",     psi)

    params = jnp.array([t0_diff, log_tE, u0, 
                        log_q, log_s, alpha, 
                        log_q3, log_s2, psi, 
                        piEN, piEE], dtype=jnp.float64)

    nll = nll_fn(params, data_moa, data_ogle, parallax_params)
    numpyro.factor("log_likelihood", -nll)

#fisher_matrix = np.load("example/OGLE-2014-BLG-1722/fisher_matrix_norho.npy")
#damping = 1e-3  # Damping factor for numerical stability
#fisher_matrix_pd = fisher_matrix + damping * np.eye(fisher_matrix.shape[0])
#fisher_cov = np.linalg.pinv(fisher_matrix_pd)
#diag_variances = jnp.abs(jnp.diag(fisher_cov))  # 共分散行列の対角（各パラメータの分散）
#print("Parameter variances:")
#print(diag_variances)
#inv_mass_matrix = 1.0 / diag_variances

inv_mass_matrix = jnp.array([
    1.0 / 0.0001**2,  # t0_diff
    1.0 / 0.01**2,    # log_tE
    1.0 / 0.001**2,   # u0
    1.0 / 0.01**2,    # log_q
    1.0 / 0.01**2,    # log_s
    1.0 / 0.01**2,    # alpha
    1.0 / 0.01**2,    # log_q3
    1.0 / 0.01**2,    # log_s2
    1.0 / 0.01**2,    # psi
    1.0 / 0.1**2,     # piEN
    1.0 / 0.1**2,     # piEE
    ], dtype=jnp.float64)


init_strategy=numpyro.infer.init_to_median()
kernel = NUTS(model, init_strategy=init_strategy,
              dense_mass=False,  # ← Fisher行列は dense
              inverse_mass_matrix=inv_mass_matrix,
              regularize_mass_matrix=True,
              adapt_mass_matrix=True,
              adapt_step_size=True,  
              target_accept_prob=0.9,
              )
mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, num_chains=1, progress_bar=True)
mcmc.run(jax.random.PRNGKey(0), data_moa=data_moa, data_ogle=data_ogle, parallax_params=parallax_params)
mcmc.print_summary(exclude_deterministic=False)

import arviz as az
import corner
idata = az.from_numpyro(mcmc)
idata.to_netcdf("example/OGLE-2014-BLG-1722/mcmc_full.nc")

idata = az.from_netcdf("example/OGLE-2014-BLG-1722/mcmc_full.nc")
param_names = ['t0_diff', 'u0', 'log_tE', 'log_q', 'log_s', 'alpha',
               'log_q3', 'log_s2', 'psi', 'piEE', 'piEN']

summary = az.summary(idata, var_names=param_names, round_to=4)
print(summary) 
az.plot_trace(idata, var_names=param_names, figsize=(12, len(param_names)*2.5))
plt.tight_layout()
plt.savefig("example/OGLE-2014-BLG-1722/trace_plot.png", dpi=300)
plt.close()

az.plot_autocorr(idata, var_names=param_names)
plt.savefig("example/OGLE-2014-BLG-1722/autocorr_plot.png", dpi=300)
plt.close()

posterior_samples = idata.posterior
samples = np.concatenate(
    [posterior_samples[param].values.reshape(-1, 1) for param in param_names],
    axis=1
)

labels = [
    r"$t_0^\prime$",        # プライム記号は右上に
    r"$u_0$",               # サブスクリプト付き
    r"$\log t_E$",          # 対数は roman に
    r"$\log q$",            
    r"$\log s$",            
    r"$\alpha$",            # ギリシャ文字
    r"$\log q_3$",          
    r"$\log s_2$",          
    r"$\psi$",              
    r"$\pi_{E,E}$",         # 下付き添字にカンマ（E方向）
    r"$\pi_{E,N}$",         # N方向
]
figure = corner.corner(samples, labels=labels, show_titles=True,
                       title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 14})

figure.savefig("example/OGLE-2014-BLG-1722/corner_plot.png", dpi=300)