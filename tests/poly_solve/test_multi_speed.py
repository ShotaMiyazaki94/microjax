import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.poly_solver import poly_roots_EA, poly_roots_EA_multi
import timeit
import time

def generate_coeffs(key, num_coeffs, num_eqs, dtype=jnp.complex128):
    real_part = jax.random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    imag_part = jax.random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    return real_part + 1j * imag_part

test_deg = 6
lengths = np.array([1e+5, 1e+6, 1e+7],dtype=int)

for l in lengths:
    key = jax.random.PRNGKey(0)
    coeffs_multi = generate_coeffs(key, test_deg, int(l))
    #coeffs_multi = jnp.array([np.random.uniform(-1, 1, test_deg) + 1j * np.random.uniform(-1, 1, test_deg) for i in range(l)])
    start = time.time()
    roots = poly_roots_EA_multi(coeffs_multi)
    end = time.time()
    print("%d degree, %.1e equations: %.3f"%(test_deg, l ,end - start), "sec")

test_deg = 11
for l in lengths:
    key = jax.random.PRNGKey(0)
    coeffs_multi = generate_coeffs(key, test_deg, int(l))
    #coeffs_multi = jnp.array([np.random.uniform(-1, 1, test_deg) + 1j * np.random.uniform(-1, 1, test_deg) for i in range(l)])
    start = time.time()
    roots = poly_roots_EA_multi(coeffs_multi)
    end = time.time()
    print("%d degree, %.1e equations: %.3f"%(test_deg, l ,end - start), "sec")

#coeffs_multi = jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) + 1j * jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) 
