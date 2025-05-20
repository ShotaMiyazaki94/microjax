import os
os.environ["OMP_DISPLAY_ENV"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax, vmap, jit

from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import critical_and_caustic_curves

import time
import pandas as pd
import VBBinaryLensing
VBBL = VBBinaryLensing.VBBinaryLensing()
VBBL.a1 = 0.0
VBBL.RelTol = 1e-4

s, q = 1.0, 0.1
r_resolution = 1000
th_resolution = 1000
Nlimb = 500
margin_r = 1.0
margin_th = 1.0
bins_r = 50
bins_th = 120
cubic = True

@jit
def mag_mj(w):
    return mag_uniform(w, rho, s=s, q=q,
                        r_resolution=r_resolution, th_resolution=th_resolution,
                        Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th,
                        margin_r=margin_r, margin_th=margin_th, cubic=cubic)
def chunked_vmap(func, data, chunk_size):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        results.append(jax.vmap(func)(chunk))
    return jnp.concatenate(results)

results = []
rho_list = [1e-01, 1e-02, 1e-03]
npts_list = [10, 30, 100, 300]
for i, rho in enumerate(rho_list):
    for j, npts in enumerate(npts_list):
        #print(f"rho: {rho}, npts: {npts}")
        npts = int(npts) 
        _, caustic_curves = critical_and_caustic_curves(npts=npts, nlenses=2, s=s, q=q)
        caustic_curves = caustic_curves.reshape(-1)
        
        key = random.PRNGKey(0)
        key, subkey1, subkey2 = random.split(key, num=3)
        phi = random.uniform(subkey1, caustic_curves.shape, minval=-jnp.pi, maxval=jnp.pi)
        r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=1.0*rho)
        w_test = caustic_curves + r * jnp.exp(1j * phi)

        _ = chunked_vmap(mag_mj, w_test, chunk_size=1000).block_until_ready()  # Warm-up
        times_mj = []
        for _ in range(10):
            start = time.time()
            _ = chunked_vmap(mag_mj, w_test, chunk_size=1000).block_until_ready()
            times_mj.append(time.time() - start)

        y1 = w_test.real
        y2 = w_test.imag
        times_vbb = []
        for _ in range(10):
            start = time.time()
            _ = jnp.array([VBBL.BinaryMag2(s, q, float(x), float(y), rho) for x, y in zip(y1, y2)])
            times_vbb.append(time.time() - start)

        print(f"rho: {rho}, npts: {npts}, time_mj: {jnp.mean(jnp.array(times_mj))}, time_vbb: {jnp.mean(jnp.array(times_vbb))}")
        results.append({
            "rho": float(rho),
            "npts": int(npts),
            "time_mj_mean": jnp.mean(jnp.array(times_mj)),
            "time_mj_std": jnp.std(jnp.array(times_mj)),
            "time_vbb_mean": jnp.mean(jnp.array(times_vbb)),
            "time_vbb_std": jnp.std(jnp.array(times_vbb)),
        })

df_results = pd.DataFrame(results)
df_results.to_csv("paper/data/speed_comparison_uniform.csv", index=False)
print(df_results)
