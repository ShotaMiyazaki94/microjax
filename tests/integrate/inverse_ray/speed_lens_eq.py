from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.poly_solver import poly_roots
from microjax.utils import match_points
from microjax.coeffs import _poly_coeffs_binary, _poly_coeffs_triple 
from microjax.coeffs import _poly_coeffs_critical_triple, _poly_coeffs_critical_binary
from microjax.point_source import lens_eq, lens_eq_det_jac

x_ = jnp.arange(-1, 1, 1e-3)
y_ = jnp.arange(-1, 1, 1e-3)

x_grid, y_grid = jnp.meshgrid(x_, y_)

s  = 1.0
q  = 0.1
a  = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1, "q": q, "s": s} 
z_mesh = x_grid.ravel() + 1j * y_grid.ravel()

jit_lens_eq = jit(lens_eq)
# 初回の実行（JITコンパイルを行うため）
_ = jit_lens_eq(z_mesh, **_params)
jax.device_get(_)

# 実際の計測
import time
num_measurements = 10
times = []

for _ in range(num_measurements):
    start = time.time()
    source_ = jit_lens_eq(z_mesh, **_params)
    jax.device_get(source_)
    end = time.time()
    times.append(end - start)

# 平均時間を計算
average_time = np.mean(times)
std_time = np.std(times)
print("Average execution time: %.1e +- %.1e"%(average_time, std_time))