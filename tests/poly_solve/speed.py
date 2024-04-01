import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.poly_solver import poly_roots_EA
import timeit

def test_solver(solver):
    coeffs = np.random.uniform(-1, 1, deg) + 1j * np.random.uniform(-1, 1, deg)
    return solver(coeffs).block_until_ready()


nrun=10000
deg=6
print("deg: %d"%(deg))
time_jax = timeit.timeit(lambda: test_solver(poly_roots_EA), number=nrun)
time_jnp = timeit.timeit(lambda: test_solver(jnp.roots), number=nrun)
print("JAX Erhlich-Aberth:", time_jax)
print("JAX Numpy roots   :", time_jnp)