import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.poly_solver import poly_roots_EA_multi
import timeit
import time
import matplotlib.pyplot as plt

def generate_coeffs(key, num_coeffs, num_eqs, dtype=jnp.complex128):
    real_part = jax.random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    imag_part = jax.random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    return jnp.array(real_part + 1j * imag_part, dtype=dtype)

def measure_time(degree, lengths):
    key = jax.random.PRNGKey(0)
    times = []
    for l in lengths:
        key, subkey = jax.random.split(key)
        coeffs_multi = generate_coeffs(subkey, degree, int(l))
        start = time.time()
        roots = poly_roots_EA_multi(coeffs_multi)
        jax.device_get(roots)
        end = time.time()
        print(f"{degree} degree, {l:.1e} equations: {end - start:.3f} sec")
        times.append(end - start)
    return times

lengths = 10**np.arange(3, 7.25, 0.25)

time6 = measure_time(6, lengths)
time11 = measure_time(11, lengths)

plt.figure()
plt.title("test_multi_speed.py")
plt.plot(lengths, time6)
plt.plot(lengths, time11)
plt.loglog()
plt.legend(["5 deg (binary-lens)","10 deg (triple-lens)"])
plt.grid(ls=":")
plt.xlabel("number of equations")
plt.ylabel("time (sec.)")
plt.savefig("tests/poly_solve/test_multi_speed.png", dpi=200, bbox_inches="tight")
plt.close()



"""
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
    return jnp.array(real_part + 1j * imag_part, dtype=dtype)

key = jax.random.PRNGKey(0)
test_deg = 6
lengths = 10**np.arange(4,7.5,0.5)
time6 = []
for l in lengths:
    coeffs_multi = generate_coeffs(key, test_deg, int(l))
    #coeffs_multi = jnp.array([np.random.uniform(-1, 1, test_deg) + 1j * np.random.uniform(-1, 1, test_deg) for i in range(l)])
    start = time.time()
    roots = poly_roots_EA_multi(coeffs_multi)
    jax.device_get(roots)
    end = time.time()
    print("%d degree, %.1e equations: %.3f"%(test_deg, l ,end - start), "sec")
    time6.append(end - start)

test_deg = 11
time11 = []
for l in lengths:
    coeffs_multi = generate_coeffs(key, test_deg, int(l))
    #coeffs_multi = jnp.array([np.random.uniform(-1, 1, test_deg) + 1j * np.random.uniform(-1, 1, test_deg) for i in range(l)])
    start = time.time()
    roots = poly_roots_EA_multi(coeffs_multi)
    jax.device_get(roots)
    end = time.time()
    print("%d degree, %.1e equations: %.3f"%(test_deg, l ,end - start), "sec")
    time11.append(end - start)
#coeffs_multi = jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) + 1j * jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) 

import matplotlib.pyplot as plt
plt.figure()
plt.plot(lengths, time6)
plt.plot(lengths, time11)
plt.loglog()
plt.legend(["6 deg","11 deg"])
plt.grid(ls=":")
plt.xlabel("number of equations")
plt.ylabel("time (sec.)")
plt.savefig("tests/figs/speed.png",dpi=200,bbox_inches="tight")
plt.close()
"""