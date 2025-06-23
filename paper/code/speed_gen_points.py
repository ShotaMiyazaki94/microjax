import jax
import jax.numpy as jnp
from jax import random
from microjax.point_source import critical_and_caustic_curves
jax.config.update("jax_enable_x64", True)
import numpy as np

# 固定パラメータ
s, q = 1.0, 0.1
rho_list = [1e-01, 1e-02, 1e-03]
npts_list = [10, 30, 100, 300]
key = random.PRNGKey(42)

def generate_test_points(rho, npts, key):
    _, caustic_curves = critical_and_caustic_curves(npts=npts, nlenses=2, s=s, q=q)
    caustic_curves = caustic_curves.reshape(-1)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-jnp.pi, maxval=jnp.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=rho)
    return caustic_curves + r * jnp.exp(1j * phi), key

# 全パターンの点を生成
all_points = {}
for rho in rho_list:
    for npts in npts_list:
        w, key = generate_test_points(rho, npts, key)
        all_points[f"rho{rho:.0e}_npts{npts}"] = np.array(w)

# ファイル保存
np.savez("paper/data/shared_test_points.npz", **all_points)
print("Saved shared test points to 'paper/data/shared_test_points.npz'")
