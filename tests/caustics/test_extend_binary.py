import unittest
import numpy as np
import jax.numpy as jnp
from microjax.caustics.extended_source import mag_extended_source
from microjax.point_source_new import _images_point_source
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

q = 0.5
s = 1.0
rho = 1e-4
theta = jnp.linspace(-np.pi, np.pi, 100 - 1, endpoint=False)
theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-8)
w_sec = rho * jnp.exp(1j * theta) 
print("w_sec shape:",w_sec.shape)

a  = 0.5 * s
e1 = q / (1 + q) 
_params = {"a": a, "e1": e1, "q": q, "s": s}
z, z_mask = _images_point_source(w_sec, nlenses=2, **_params)
print("z, z_mask",z.shape, z_mask.shape)

w = jnp.array([0.0 + 0.0j]) 
mag = mag_extended_source(w, rho, nlenses=2, **_params, limb_darkening=True, u1=0.5,)
print(mag)