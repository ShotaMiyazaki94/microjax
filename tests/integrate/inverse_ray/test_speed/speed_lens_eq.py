from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.poly_solver import poly_roots
from microjax.utils import match_points
from microjax.coeffs import _poly_coeffs_binary, _poly_coeffs_triple 
from microjax.coeffs import _poly_coeffs_critical_triple, _poly_coeffs_critical_binary
from microjax.point_source import lens_eq, lens_eq_det_jac
#jax.config.update("jax_enable_x64", True)

resolution = jnp.logspace(-1.5, -4, 50)

s  = 1.0
q  = 0.1
a  = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1, "q": q, "s": s} 

import time
num_measurements = 10
mean_times = []
std_times = []

for r in resolution:
    x_ = jnp.arange(-1, 1, r)
    y_ = jnp.arange(-1, 1, r)
    x_grid, y_grid = jnp.meshgrid(x_, y_)
    z_mesh = x_grid.ravel() + 1j * y_grid.ravel()

    times = []
    for _ in range(num_measurements):
        start = time.time()
        source_ = lens_eq(z_mesh, **_params)
        source_.block_until_ready()
        #jax.device_get(source_)
        end = time.time()
        times.append(end - start)
    
    average_time = np.mean(times)
    std_time = np.std(times)
    print(f"Resolution: {r:.2e}, Grid Points: {4.0/r**2:.1e}, Average Time: {average_time:.5f}, Std Dev: {std_time:.5f}")
    mean_times.append(average_time)
    std_times.append(std_time)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(4.0/resolution**2, mean_times)
plt.xscale("log")
plt.yscale("log")
plt.ylim(5e-4, 10)
plt.grid(ls="--")
plt.title("Speed of lens_eq")
plt.xlabel("Number of Grid Points")
plt.ylabel("Computation Time (seconds)")
plt.savefig('tests/integrate/inverse_ray/test_speed/speed_lens_eq.pdf', bbox_inches="tight")
#plt.savefig('tests/integrate/brute_force/speed_lens_eq.pdf')
plt.show()
