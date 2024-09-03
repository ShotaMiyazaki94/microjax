from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.poly_solver import poly_roots
from microjax.utils import match_points
from microjax.coeffs import _poly_coeffs_binary, _poly_coeffs_triple 
from microjax.coeffs import _poly_coeffs_critical_triple, _poly_coeffs_critical_binary
from microjax.point_source import lens_eq, lens_eq_det_jac,_images_point_source
import time
jax.config.update("jax_enable_x64", True)

w_center = jnp.complex128(-0.14 - 0.1j)
s  = 1.0
q  = 0.1
a  = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1, "q": q, "s": s}
rho = 1e-3

Nlimbs = jnp.int_(10**jnp.arange(1,4.5,0.5))
num_measurements = 100
mean_times = []
std_times = []
for Nlimb in Nlimbs:
    times = []
    for _ in range(num_measurements):
        start = time.time()
        w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
        w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
        image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
        image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
        image_limb.block_until_ready()
        #jax.device_get(source_)
        end = time.time()
        times.append(end - start)
    average_time = np.mean(times)
    std_time = np.std(times)
    print(f"{Nlimb:.1e}:Average Time: {average_time:.5f}, Std Dev: {std_time:.5f}")
    mean_times.append(average_time)
    std_times.append(std_time)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(Nlimbs, mean_times)
plt.xscale("log")
plt.yscale("log")
#plt.ylim(5e-4, 10)
plt.grid(ls="--")
plt.title("Speed of _images_point_source")
plt.xlabel("Number of Source Limb Points")
plt.ylabel("Computation Time (seconds)")
plt.savefig('tests/integrate/brute_force/speed_image_solve.png')
plt.show()