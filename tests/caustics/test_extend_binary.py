import unittest
import numpy as np
import jax.numpy as jnp
from microjax.caustics.extended_source import mag_extended_source
from microjax.point_source import _images_point_source
import jax
from jax import lax, jit
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from functools import partial

q = 0.5
s = 1.0
rho = 0.1
theta = jnp.linspace(-np.pi, np.pi, 100 - 1, endpoint=False)
theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-8)
w_sec = rho * jnp.exp(1j * theta) 
print("w_sec shape:",w_sec.shape)

a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"a": a, "e1": e1, "q": q, "s": s}
z, z_mask = _images_point_source(w_sec, nlenses=2, a=a, e1=e1)
print("z, z_mask",z.shape, z_mask.shape)

w = jnp.array([0.0 + 0.0j]) 
mag = mag_extended_source(w, rho, nlenses=2, s=s, q=q, limb_darkening=True, u1=0.5)
print(mag)

@partial(jit, static_argnames=("npts_limb", "npts_ld", "limb_darkening"))
def mag_binary(w_points, rho, s, q, u1=0., npts_limb=300, npts_ld=100, limb_darkening=False):
    def body_fn(_, w):
        mag = mag_extended_source(
            w,
            rho,
            nlenses=2,
            npts_limb=npts_limb,
            limb_darkening=limb_darkening,
            npts_ld=npts_ld,
            u1=u1,
            s=s,
            q=q,
        )
        return 0, mag

    _, mags = lax.scan(body_fn, 0, w_points)
    return mags

w = jnp.array([0.0 + 0.0j, 0.1+0.1j]) 
mags = mag_binary(w, rho, s=s, q=q)
print(mags)