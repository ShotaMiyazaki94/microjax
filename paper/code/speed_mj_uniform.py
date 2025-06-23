import os
import time
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import critical_and_caustic_curves
jax.config.update("jax_enable_x64", True)

data = np.load("paper/data/shared_test_points.npz")
s, q = 1.0, 0.1
r_resolution = 1000
th_resolution = 1000
Nlimb = 500
margin_r = 1.0
margin_th = 1.0
bins_r = 50
bins_th = 120
cubic = True
rho_list = [1e-01, 1e-02, 1e-03]
npts_list = [10, 30, 100, 300]

@jit
def mag_mj(w, rho):
    return mag_uniform(w, rho, s=s, q=q,
                       r_resolution=r_resolution, th_resolution=th_resolution,
                       Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th,
                       margin_r=margin_r, margin_th=margin_th, cubic=cubic)

def chunked_vmap(func, data, rho, chunk_size=1000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        results.append(vmap(lambda w: func(w, rho))(chunk))
    return jnp.concatenate(results)

results = []
for rho in rho_list:
    for npts in npts_list:
        keyname = f"rho{rho:.0e}_npts{npts}"
        w_test = jnp.array(data[keyname])
        #w_test = generate_test_points(rho, npts, key)
        _ = chunked_vmap(mag_mj, w_test, rho).block_until_ready()
        times = []
        for _ in range(10):
            start = time.time()
            _ = chunked_vmap(mag_mj, w_test, rho).block_until_ready()
            times.append(time.time() - start)
        results.append({
            "rho": float(rho),
            "npts": int(npts),
            "time_mj_mean": jnp.mean(jnp.array(times)),
            "time_mj_std": jnp.std(jnp.array(times)),
        })

df = pd.DataFrame(results)
df.to_csv("paper/data/speed_uniform_microjax.csv", index=False)
print(df)
