import numpy as np
import VBBinaryLensing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax.numpy as jnp
import jax
from jax import lax, vmap, jit
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.inverse_ray.cond_extended import test_full
from microjax.multipole import _mag_hexadecapole
from microjax.point_source import critical_and_caustic_curves

data_moa  = np.load("example/ogle-2003-blg-235/flux_moa.npy")
data_ogle = np.load("example/ogle-2003-blg-235/flux_ogle.npy")
fs_ogle, fb_ogle = 9.072, 2.856
t_moa, flux_moa, fluxe_moa = data_moa[0] - 2450000, data_moa[1], data_moa[2]
t_ogle, flux_ogle, fluxe_ogle = data_ogle[0] - 2450000, data_ogle[1], data_ogle[2]
t_data     = np.hstack([t_moa, t_ogle])
flux_data  = np.hstack([flux_moa, flux_ogle])
fluxe_data = np.hstack([fluxe_moa, fluxe_ogle])
data_input = (t_data, flux_data, fluxe_data)

params_init = {
    "t0": 2848.16048754,
    "tE": 61.61235588,
    "u0": 0.11760426,
    "q": 10**-2.3609089,
    "s": 10**0.0342508,
    "alpha": 4.00180035,
    "rho": 10**-3.94500971
}


@jit
def mag_binary_time(time, params):
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    q, s, alpha = params["q"], params["s"], params["alpha"]
    rho = params["rho"]
    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    
    Nlimb = 500
    r_resolution  = 500
    th_resolution = 500
    MAX_FULL_CALLS = 100
    
    cubic = True
    bins_r = 50
    bins_th = 120
    margin_r = 1.0
    margin_th= 1.0
    
    magn = mag_binary(w_points, rho, s=s, q=q, r_resolution=r_resolution, 
                      th_resolution=th_resolution,cubic=cubic, Nlimb=Nlimb, 
                      bins_r=bins_r, bins_th=bins_th, margin_r=margin_r, margin_th=margin_th, 
                      MAX_FULL_CALLS=MAX_FULL_CALLS)
    return magn


@jit
def test_full_binary(time, params):
    t0, tE, u0  = params["t0"], params["tE"], params["u0"]
    q, s, alpha = params["q"], params["s"], params["alpha"]
    rho = params["rho"]
    tau = (time - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    
    a = 0.5 * s
    e1 = q / (1.0 + q)
    nlenses = 2
    x_cm = a * (1 - q) / (1 + q)
    _params = {"a": a, "e1": e1}
    w_points_shifted = w_points - x_cm
    
    test = test_full(w_points_shifted, rho, nlenses=nlenses, **_params)
    num_full  = jnp.sum(~test)
    return num_full

for key, value in params_init.items():
    print(f"{key}: {value:.6e}")

print("N_full=", test_full_binary(time=t_data, params=params_init))