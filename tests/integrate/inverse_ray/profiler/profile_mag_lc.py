import time
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
from microjax.inverse_ray.extended_source import mag_uniform
from microjax.inverse_ray.lightcurve import mag_lc_uniform
import MulensModel as mm

q = 0.5
s = 0.9
alpha = jnp.deg2rad(30)
tE = 10
t0 = 0.0
u0 = 0.1
rho = 0.08

num_points = 2000
t = jnp.linspace(-0.8 * tE, 0.8 * tE, num_points)
tau = (t - t0) / tE
y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
test_params = {"q": q, "s": s}

r_resolution = 500
th_resolution = 500
cubic = True

def chunked_vmap(func, data, chunk_size):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        results.append(jax.vmap(func)(chunk))
    return jnp.concatenate(results)

chunk_size = 2000
#mag_mj = lambda w: mag_uniform(w, rho, s=s, q=q, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic)
@jax.jit
def mag_mj(w):
    return mag_uniform(w, rho, s=s, q=q, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic)

#_ = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()

def mag_vbbl(w0, rho, u1=0., accuracy=1e-05):
    a = 0.5 * s
    e1 = 1.0 / (1.0 + q)
    e2 = 1.0 - e1
    bl = mm.BinaryLens(e1, e2, 2 * a)
    return bl.vbbl_magnification(w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)

mag_lc_uniform(w_points, rho, q=q, s=s, r_resolution=r_resolution, th_resolution=th_resolution)
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    magnifications = mag_lc_uniform(w_points, rho, q=q, s=s, r_resolution=r_resolution, th_resolution=th_resolution)
    #magnifications = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()