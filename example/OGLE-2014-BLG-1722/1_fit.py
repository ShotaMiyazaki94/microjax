import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_triple
from scipy.interpolate import interp1d
from microjax.trajectory.parallax import peri_vernal, set_parallax, compute_parallax
from microjax.point_source import critical_and_caustic_curves
from microjax.likelihood import linear_chi2, nll_ulens
from microjax.point_source import mag_point_source


from astropy.coordinates import SkyCoord
import astropy.units as u

coords = "17:55:00.57 -31:28:08.6"
c = SkyCoord(coords, frame="icrs", unit=(u.hourangle, u.deg),)
RA = c.ra.deg
Dec = c.dec.deg
data_moa = np.loadtxt("example/OGLE-2014-BLG-1722/data/Ian2.dat.flux.norm")
data_ogle = np.loadtxt("example/OGLE-2014-BLG-1722/data/OGLE-2014-BLG-1722.dat.flux.norm")
data_moa = data_moa[data_moa[:, 0] > 6250]
data_ogle = data_ogle[data_ogle[:, 0] > 6250]

inits = jnp.array([6900.224, 23.819, -0.131, # t0, tE, u0
                   jnp.log10(4.468e-4), jnp.log10(0.754), -0.228, # q, s, alpha,
                   jnp.log10(6.388e-4), jnp.log10(0.851), -2.196, # q3, s2, psi
                   0.199, 0.092 # piEN, piEE
                   ])
tref = 6900.0
tperi, tvernal = peri_vernal(tref)
parallax_params = set_parallax(tref, tperi, tvernal, RA, Dec)

@jax.jit
def mag_time(time, params, parallax_params):
    t0, tE, u0, log_q, log_s, alpha, log_q3, log_s2, psi, piEN, piEE = params
    q = 10**log_q
    s = 10**log_s
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
    return magn, w_points

def forward_grad(loss_fn, params, data):
    grad = jax.jacfwd(loss_fn)(params, data)
    return grad

def loss_fn(params, data):
    t0, tE, u0, log_q, log_s, alpha, log_q3, log_s2, psi, piEN, piEE = params
    data_moa, data_ogle, parallax_params = data
    #time, flux, fluxe, parallax_params = data
    model_params = jnp.array([t0, tE, u0, log_q, log_s, alpha, log_q3, log_s2, psi, piEN, piEE])
    # MOA data
    t_moa, flux_moa, fluxe_moa = data_moa[:, 0], data_moa[:, 1], data_moa[:, 2]
    magn_moa, _ = mag_time(t_moa, model_params, parallax_params)
    M = jnp.stack([magn_moa - 1.0, jnp.ones_like(magn_moa)], axis=1)
    nll_moa = nll_ulens(flux_moa, M, fluxe_moa**2, 1e9, 1e9)
    # OGLE data
    t_ogle, flux_ogle, fluxe_ogle = data_ogle[:, 0], data_ogle[:, 1], data_ogle[:, 2]
    magn_ogle, _ = mag_time(t_ogle, model_params, parallax_params)
    M = jnp.stack([magn_ogle - 1.0, jnp.ones_like(magn_ogle)], axis=1)
    nll_ogle = nll_ulens(flux_ogle, M, fluxe_ogle**2, 1e9, 1e9)
    return nll_moa + nll_ogle

import optax
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(inits)

@jax.jit
def update(params, opt_state, data):
    loss = loss_fn(params, data)
    grads = forward_grad(loss_fn, params, data)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params = inits
data_input = (data_moa, data_ogle, parallax_params)
losses = []
for step in range(5000):
    params, opt_state, loss = update(params, opt_state, data_input)
    losses.append(loss)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss:.3f}")

np.save("example/OGLE-2014-BLG-1722/params_final.npy", np.array(params))       # パラメータの最終値
np.save("example/OGLE-2014-BLG-1722/loss_curve.npy", np.array(losses))         # 損失関数の履歴

