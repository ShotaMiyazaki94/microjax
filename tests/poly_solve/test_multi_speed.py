import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.poly_solver import poly_roots_EA, poly_roots_EA_multi
import timeit
import time

test_deg = 11
length = int(1e+5)
key = jax.random.PRNGKey(0)
#coeffs_multi = jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) + 1j * jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) 
coeffs_multi = jnp.array([np.random.uniform(-1, 1, test_deg) + 1j * np.random.uniform(-1, 1, test_deg) for i in range(length)])

print(coeffs_multi.shape)
start = time.time()
roots = poly_roots_EA_multi(coeffs_multi)
end = time.time()
print(roots.shape)
print("%.3f"%(end - start), "sec")