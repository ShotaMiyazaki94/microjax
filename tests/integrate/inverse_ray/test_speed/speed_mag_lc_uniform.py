import jax
from jax import config 
config.update('jax_enable_x64', True) 
import pandas as pd
import numpy as np
import time
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from microjax.inverse_ray.lightcurve import mag_lc_uniform
#from microjax.inverse_ray.extended_source import mag_uniform

# parameters
q = 0.1
s = 1.0
alpha = jnp.deg2rad(30) 
tE = 10 
t0 = 0.0 
u0 = 0.1 
rho = 0.1
# conditions
Nlimb = 500
r_resolution  = 500
th_resolution = 500
cubic = True
bins_r = 50
bins_th = 120
margin_r = 0.5
margin_th= 0.5

Npoints = jnp.logspace(1, 3.5, 30, endpoint=True)
times=[]
for num_points in Npoints:
    # source trajectory
    num_points = jnp.array(num_points, dtype=int)
    t  =  jnp.linspace(-0.8*tE, 0.8*tE, num_points)
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    test_params = {"q": q, "s": s} 
    _ = mag_lc_uniform(w_points, rho, nlenses=2, q=q, s=s, cubic=cubic,
                       r_resolution=r_resolution, th_resolution=th_resolution)
    start = time.time()
    _ = mag_lc_uniform(w_points, rho, nlenses=2, q=q, s=s, cubic=cubic,
                       r_resolution=r_resolution, th_resolution=th_resolution).block_until_ready()
    end = time.time()
    calc_time = end - start 
    print(f"{num_points:3d}: Average Time: {calc_time:.5f}")
    times.append(calc_time)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(Npoints, times)
plt.xscale("log")
plt.yscale("log")
plt.grid(ls="--")
plt.xlabel("Number of Data Points")
plt.ylabel("Computation Time (seconds)")
plt.title("Speed of mag_lc_uniform")
plt.savefig('tests/integrate/inverse_ray/test_speed/speed_mag_lc_uniform.pdf', bbox_inches="tight")
#plt.title("Speed of determine_grid_regions")
#plt.savefig('tests/integrate/inverse_ray/figs/speed_determine_grid_regions.pdf', bbox_inches="tight")
plt.show()