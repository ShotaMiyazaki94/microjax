import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.poly_solver import poly_roots_EA, poly_roots_EA_multi
import timeit
import time

test_deg = 10
length = int(1e+6)
coeff_multi = jnp.array([np.random.uniform(-1, 1, test_deg) + 1j * np.random.uniform(-1, 1, test_deg) for i in range(length)])

print(coeff_multi.shape)
start = time.time()
roots = poly_roots_EA_multi(coeff_multi)
end = time.time()
print(roots.shape)
print(end - start)