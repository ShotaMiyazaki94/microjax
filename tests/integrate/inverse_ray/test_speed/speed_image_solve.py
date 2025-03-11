from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.point_source import _images_point_source
import time
jax.config.update("jax_enable_x64", True)

# Constants and parameters
w_center = jnp.complex128(-0.14 - 0.1j)
s = 1.0
q = 0.1
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1, "q": q, "s": s}
rho = 1e-3

Nlimbs = jnp.int_(10**jnp.arange(1.5,5.5,0.125))
num_measurements = 100
mean_times = []
std_times = []

@jit
def _images_point_source_jit(w_limb_shift, a, e1):
    return _images_point_source(w_limb_shift, a=a, e1=e1)

for Nlimb in Nlimbs:
    times = []
    for _ in range(num_measurements):
        start = time.time()

        # Precompute `w_limb` efficiently with jnp.complex128
        w_limb = w_center + rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, Nlimb))
        w_limb_shift = w_limb - 0.5 * s * (1 - q) / (1 + q)

        # JIT-compiled function call
        image, mask = _images_point_source_jit(w_limb_shift, a=a, e1=e1)

        # Computation on the image
        image_limb = image + 0.5 * s * (1 - q) / (1 + q)

        # Block until ready to ensure all computations are complete
        image_limb.block_until_ready()

        end = time.time()
        times.append(end - start)
    
    average_time = np.mean(times)
    std_time = np.std(times)
    print(f"{Nlimb:.1e}: Average Time: {average_time:.5f}, Std Dev: {std_time:.5f}")
    mean_times.append(average_time)
    std_times.append(std_time)

# Plotting
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(Nlimbs[1:], mean_times[1:])
plt.xscale("log")
plt.yscale("log")
plt.grid(ls="--")
plt.title("Speed of _images_point_source")
plt.xlabel("Number of Source Limb Points")
plt.ylabel("Computation Time (seconds)")
plt.savefig('tests/integrate/inverse_ray/test_speed/speed_image_solve.pdf', bbox_inches="tight")
plt.show()
