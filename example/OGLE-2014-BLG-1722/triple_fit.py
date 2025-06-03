from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap 

from microjax.inverse_ray.extended_source import mag_limb_dark
from microjax.point_source import mag_point_source, _images_point_source
from microjax.multipole import _mag_hexadecapole
from microjax.utils import *

@partial(jax.jit, static_argnames=("r_resolution", "th_resolution", "u1", "delta_c",
                                   "bins_r", "bins_th", "margin_r", "margin_th", 
                                   "Nlimb", "MAX_FULL_CALLS", "chunk_size"))
def triple_fit(w_points, rho, full_label, r_resolution=500, th_resolution=500, u1=0.0, delta_c=0.01,
               Nlimb=500, bins_r=50, bins_th=120, margin_r=1.0, margin_th=1.0, MAX_FULL_CALLS=500, chunk_size=50, **params):
    nlenses = 3
    s, q, q3 = params["s"], params["q"], params["q3"]
    r3 = params["r3"]
    a = 0.5 * s
    e1 = q / (1.0 + q + q3) 
    e2 = 1.0/(1.0 + q + q3)
    _params = {**params, "a": a, "e1": e1, "e2": e2, "r3": r3}
    x_cm = a * (1.0 - q) / (1.0 + q)
    #w_points_shifted = w_points - x_cm
    mag_point = mag_point_source(w_points, nlenses=nlenses, **_params)
    
    def _mag_full(w):
            return mag_limb_dark(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution= th_resolution,
                                 u1 = u1, delta_c = delta_c, bins_r = bins_r, bins_th = bins_th, margin_r = margin_r,
                                 margin_th = margin_th, Nlimb = Nlimb, **_params)
    
    def chunked_vmap(func, data, chunk_size):
        N = data.shape[0]
        pad_len = (-N) % chunk_size
        chunks = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:])
        return lax.map(lambda c: vmap(func)(c), chunks).reshape(-1, *data.shape[2:])[:N]

    idx_sorted = jnp.argsort(~full_label)
    idx_full = idx_sorted[:MAX_FULL_CALLS]
    mag_extended = chunked_vmap(_mag_full, w_points[idx_full], chunk_size)
    mags = mag_point.at[idx_full].set(mag_extended)
    return mags