import jax
from jax import config 
config.update('jax_enable_x64', True) 
import pandas as pd
import numpy as np
import time
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from microjax.inverse_ray.extended_source import mag_uniform

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
chunk_size = 1000  

@jax.jit
def mag_mj(w):
    return mag_uniform(w, rho, s=s, q=q, Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th,
                        r_resolution=r_resolution, th_resolution=th_resolution, 
                        margin_r = margin_r, margin_th=margin_th, cubic=cubic)
def chunked_vmap(func, data, chunk_size):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        results.append(jax.vmap(func)(chunk))
    return jnp.concatenate(results)

Npoints = jnp.logspace(0, 3, 50, endpoint=True)
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
    _ = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()
    start = time.time()
    magnifications = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()
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
plt.title("Speed of vmap mag_uniform")
plt.savefig('tests/integrate/inverse_ray/test_speed/speed_extended_vmap.pdf', bbox_inches="tight")
#plt.title("Speed of determine_grid_regions")
#plt.savefig('tests/integrate/inverse_ray/figs/speed_determine_grid_regions.pdf', bbox_inches="tight")
plt.show()