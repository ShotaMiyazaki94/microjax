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

import VBBinaryLensing
VBBL = VBBinaryLensing.VBBinaryLensing()
VBBL.a1 = 0.0
VBBL.RelTol = 1e-4

results = []
for rho in rho_list:
    for npts in npts_list:
        keyname = f"rho{rho:.0e}_npts{npts}"
        w_test = jnp.array(data[keyname])
        y1 = w_test.real
        y2 = w_test.imag
        times = []
        for _ in range(10):
            start = time.time()
            _ = jnp.array([VBBL.BinaryMag2(s, q, float(x), float(y), rho) for x, y in zip(y1, y2)])
            times.append(time.time() - start)
        results.append({
            "rho": float(rho),
            "npts": int(npts),
            "time_vbb_mean": jnp.mean(jnp.array(times)),
            "time_vbb_std": jnp.std(jnp.array(times)),
        })

df = pd.DataFrame(results)
df.to_csv("paper/data/speed_uniform_vbb.csv", index=False)
print(df)

VBBL.a1 = 0.5

results = []
for rho in rho_list:
    for npts in npts_list:
        keyname = f"rho{rho:.0e}_npts{npts}"
        w_test = jnp.array(data[keyname])
        y1 = w_test.real
        y2 = w_test.imag
        times = []
        for _ in range(10):
            start = time.time()
            _ = jnp.array([VBBL.BinaryMag2(s, q, float(x), float(y), rho) for x, y in zip(y1, y2)])
            times.append(time.time() - start)
        results.append({
            "rho": float(rho),
            "npts": int(npts),
            "time_vbb_mean": jnp.mean(jnp.array(times)),
            "time_vbb_std": jnp.std(jnp.array(times)),
        })

df = pd.DataFrame(results)
df.to_csv("paper/data/speed_LD_vbbl.csv", index=False)
print(df)

