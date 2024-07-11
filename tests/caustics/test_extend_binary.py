import unittest
import numpy as np
import jax.numpy as jnp
from microjax.caustics.extended_source_binary import mag_extended_source_binary
from microjax.point_source import _images_point_source_binary_sequential
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

q = 0.5
s = 1.0
rho = 0.31
theta = jnp.linspace(-np.pi, np.pi, 100 - 1, endpoint=False)
theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-8)
w_sec = rho * jnp.exp(1j * theta) 
print("w_sec shape:",w_sec.shape)

a  = 0.5 * s
e1 = q / (1 + q) 
z, z_mask = _images_point_source_binary_sequential(w_sec, a, e1)
print("z, z_mask",z.shape, z_mask.shape)

w = jnp.array([0.0 + 0.0j]) 
mag = jax.jit(mag_extended_source_binary)(w, s, q, rho)
#print(mag)