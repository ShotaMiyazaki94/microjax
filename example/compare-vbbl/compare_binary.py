import time
import jax
import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
jax.config.update("jax_enable_x64", True)

q = 0.05
s = 1.0
alpha = jnp.deg2rad(10.0)
tE = 30.0
t0 = 0.0
u0 = 0.0
rho = 0.06
nlenses = 2
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1}
x_cm = a * (1.0 - q) / (1.0 + q)
num_points = 2000
t = jnp.linspace(-tE, tE, num_points)
tau = (t - t0) / tE
y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
y2 =  u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
w_points = (y1 + 1j * y2).astype(jnp.complex128)

chunk_size = 500 
Nlimb = 500
r_resolution = 500
th_resolution = 1000
bins_r = 50
bins_th = 120
margin_r = 1.0
margin_th = 1.0

from microjax.point_source import mag_point_source, critical_and_caustic_curves
from microjax.multipole import _mag_hexadecapole
from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import _images_point_source
import MulensModel as mm

def chunked_vmap(func, data, chunk_size: int):
    outs = []
    for i in range(0, len(data), chunk_size):
        outs.append(jax.vmap(func)(data[i:i + chunk_size]))
    return jnp.concatenate(outs)

@jit
def mag_mj(w):
    return mag_uniform(
        w, rho,
        s=s, q=q,
        Nlimb=Nlimb,
        bins_r=bins_r, bins_th=bins_th,
        r_resolution=r_resolution, th_resolution=th_resolution,
        margin_r=margin_r, margin_th=margin_th,
    )

@jit
def scan_mag_mj(w_points):
    def body(carry, w):
        return carry, mag_mj(w)
    _, out = lax.scan(body, None, w_points)
    return out

def mag_vbbl_(w0, rho, u1=0.0, accuracy=1e-4):
    a = 0.5 * s
    e1 = 1.0 / (1.0 + q)
    e2 = 1.0 - e1
    bl = mm.BinaryLens(e1, e2, 2 * a)
    return bl.vbbl_magnification(w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)

mag_vbbl  = lambda w0: jnp.array([mag_vbbl_(w, rho) for w in w0])

# ---- Warmup (JIT compile) ----
_ = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()
_ = mag_point_source(w_points, s=s, q=q).block_until_ready()
z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
_, _ = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) 

print("number of data points: %d"%(num_points))
start = time.time()
mags_poi = mag_point_source(w_points, s=s, q=q)
mags_poi.block_until_ready()
end = time.time()
print("computation time: %.3f sec (%.3f ms per points) for point-source in microjax"%(end-start, 1000*(end - start)/num_points))

start = time.time()
z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
mu_multi.block_until_ready()
end = time.time()
print("computation time: %.3f sec (%.3f ms per points) for hexadecapole in microjax"%(end-start, 1000*(end - start)/num_points))

start = time.time()
mag_VB = mag_vbbl(w_points)
mag_VB.block_until_ready() 
end = time.time()
print("computation time: %.3f sec (%.3f ms per points) with VBBinaryLensing"%(end - start,1000*(end - start)/num_points))

start = time.time()
mag_jax = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()
end = time.time()
print("computation time: %.3f sec (%.3f ms per points) with microjax, %d chunk_size, %d rbin, %d thbin"
      %(end-start, 1000*(end - start)/num_points, chunk_size, r_resolution, th_resolution))
if(0):
    print("start computation with lax.scan")
    start = time.time()
    mag_scan = scan_mag_mj(w_points).block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for lax.scan in microjax"%(end-start, 1000*(end - start)/num_points))