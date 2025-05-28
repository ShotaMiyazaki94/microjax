import jax, jax.numpy as jnp
from jax import jit
jax.config.update("jax_enable_x64", True)

from microjax.inverse_ray.lightcurve import mag_binary

s, q = 1.1, 0.5
alpha, tE, t0, u0, rho = jnp.deg2rad(60.), 10.0, 0.0, 0.0, 0.03
t = jnp.linspace(-22, 12, 2000)

r_resolution, th_resolution = 500, 500
Nlimb, MAX_FULL_CALLS, cubic = 500, 500, True

#@jit
def get_mag(params):
    s, q, rho, alpha, u0, t0, tE = params
    tau = (t - t0) / tE
    y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
    y2 =  u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)

    w_points = (y1 + 1j * y2).astype(jnp.complex128)  # dtype を明示
    mag = lambda w: mag_binary(
        w, rho, q=q, s=s, chunk_size=100,
        r_resolution=r_resolution, th_resolution=th_resolution,
        Nlimb=Nlimb, MAX_FULL_CALLS=MAX_FULL_CALLS,
    )
    return w_points, mag(w_points)

params = jnp.array([s, q, rho, alpha, u0, t0, tE])

_, A = get_mag(params)
A.block_until_ready()          

jax.profiler.save_device_memory_profile("memory_profile.pb", backend="gpu")  
print("device-memory profile saved")

# 実測ベンチマーク
import time, jax.profiler
start = time.time()
_, A = get_mag(params)
A.block_until_ready()
print(f"mag finish: {time.time() - start:.3f} s")