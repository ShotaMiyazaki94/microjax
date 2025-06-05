import optax
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, random
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.trajectory.parallax import compute_parallax, set_parallax, peri_vernal

import pandas as pd
import numpy as np

@partial(jit, static_argnums=(3,))
def mag_time(time, params, parallax_params, chunk_size=100):
    t0_diff, log_tE, u0, log_q, log_s, alpha, log_rho, piEN, piEE = params
    t0 = t0_diff + 6836.0
    tE = 10**log_tE
    q  = 10**log_q
    s  = 10**log_s
    rho = 10**log_rho
    _params ={"q": q, "s": s}
    dtn, dum = compute_parallax(time, piEN=piEN, piEE=piEE, parallax_params=parallax_params)
    tau = (time - t0)/tE
    um  = u0 + dum
    tm  = tau + dtn

    y1 = tm*jnp.cos(alpha) - um*jnp.sin(alpha) 
    y2 = tm*jnp.sin(alpha) + um*jnp.cos(alpha)
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

    magn = mag_binary(w_points, rho=rho, **_params, chunk_size=chunk_size,
                      r_resolution=500, th_resolution=500,
                      Nlimb=500, bins_r=100, bins_th=360,
                      margin_r=1.0, margin_th=1.0, MAX_FULL_CALLS=100)
    return magn