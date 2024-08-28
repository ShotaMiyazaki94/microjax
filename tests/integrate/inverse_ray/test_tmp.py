import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.inverse_ray.inverse_ray import image_area_all

w_center = jnp.complex128(0.425 + 0.0j)
#w_center = jnp.complex128(-0.1 - 0.1j)
q  = 0.5
s  = 1.0
a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}
rho = 1e-2
NBIN = 20

import time
start_time = time.time()
N_limb = 10000
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
jax.device_put(image_limb).block_until_ready()
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")


for N in jnp.logspace(2,5,10):
    start_time = time.time()
    N_limb = jnp.int32(N)
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
    w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
    image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
    image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
    jax.device_put(image_limb).block_until_ready()
    end_time = time.time()
    print(f"%d Execution time: {end_time - start_time} seconds"%N_limb) 
