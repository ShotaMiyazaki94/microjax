import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.point_source import _images_point_source
from microjax.inverse_ray.merge_area import calc_source_limb, determine_grid_regions
import time
jax.config.update("jax_enable_x64", True)

w_center = jnp.complex128(-0.14 - 0.1j)
s = 1.0
q = 0.1
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1, "q": q, "s": s}
rho = 1e-3

Nlimbs = jnp.int_(10**jnp.arange(2.0,5.55,0.05))
num_measurements = 10
mean_times = []
std_times = []

for Nlimb in Nlimbs:
    Nlimb = int(Nlimb)
    times = []
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    image_limb.block_until_ready()
    #r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r=0.1, offset_th=0.1)
    #r_scan.block_until_ready() 
    for _ in range(num_measurements):
        start = time.time()
        image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
        image_limb.block_until_ready()
        #r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r=0.1, offset_th=0.1)
        #r_scan.block_until_ready()
        end = time.time()
        times.append(end - start)
    average_time = np.mean(times)
    std_time = np.std(times)
    print(f"{Nlimb:.1e}: Average Time: {average_time:.5f}, Std Dev: {std_time:.5f}")
    mean_times.append(average_time)
    std_times.append(std_time)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(Nlimbs[1:], mean_times[1:])
plt.xscale("log")
plt.yscale("log")
plt.grid(ls="--")
plt.xlabel("Number of Source Limb Points")
plt.ylabel("Computation Time (seconds)")
plt.title("Speed of calc_source_limb")
plt.savefig('tests/integrate/inverse_ray/test_speed/speed_calc_source_limb.pdf', bbox_inches="tight")
#plt.title("Speed of determine_grid_regions")
#plt.savefig('tests/integrate/inverse_ray/figs/speed_determine_grid_regions.pdf', bbox_inches="tight")
plt.show()
