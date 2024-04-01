import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
from microjax.poly_solver import poly_roots_EA
import timeit
import time

def test(func,iteration=5000,test_deg=6):
    time_np = []
    time_EA = []
    sol_ac_EA=[]
    for i in range(iteration):
        coeffs = np.random.uniform(-1, 1, test_deg) + 1j * np.random.uniform(-1, 1, test_deg)
        # numpy.roots
        start = time.time()
        roots_np = jnp.roots(coeffs)
        end = time.time()
        time_np.append(end - start)
        roots_np = jnp.sort_complex(roots_np)
        # Ehrlich-Aberth-Laguerre
        start = time.time()
        roots_EA = func(coeffs)
        end = time.time()
        time_EA.append(end - start)
        roots_EA = jnp.sort_complex(roots_EA)
        # calc diffs
        diff_roots = jnp.abs(roots_np - roots_EA)
        #diff_roots = jnp.abs(roots_np) - jnp.abs(roots_EA)
        diff_sum   = jnp.sum(diff_roots)
        sol_ac_EA.append(diff_sum)
    time_np = jnp.array(time_np)
    time_EA = jnp.array(time_EA)
    sol_ac_EA = jnp.array(sol_ac_EA)
    sol_ac_EA = jnp.nan_to_num(sol_ac_EA,nan=1)
    return time_np, time_EA, sol_ac_EA

degs = 6
r  = test(func=poly_roots_EA,test_deg=degs)
degs = 11
r2 = test(func=poly_roots_EA,test_deg=degs)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="times",style="ticks",font_scale=1.2)

fig,ax=plt.subplots(1,2,figsize=(10,4))
fig.subplots_adjust(wspace=0.3)
bins=np.arange(-1.2,0,0.02)
ax[0].plot(r[1]/r[0],".",  label="5th order  (mean:%.2f)"%np.mean(r[1]/r[0]))
ax[0].plot(r2[1]/r2[0],".",label="10th order (mean:%.2f)"%np.mean(r2[1]/r2[0]))
#ax[0].set_ylim(0,1.1)
ax[0].legend()
#ax[0].set_yscale("log")
ax[0].set_title("relative computation speed to $\\tt{numpy.roots}$")
ax[0].set_xlabel("iteration")

ax[1].plot(r[2], label="5th order  (mean:%.1e)"%np.mean(r[2]))
ax[1].plot(r2[2],label="10th order (mean:%.1e)"%np.mean(r2[2]))
ax[1].set_yscale("log")
ax[1].set_xlabel("iteration")
ax[1].set_title("relative solution accuracy to $\\tt{numpy.roots}$")
ax[1].legend()
plt.show()

"""
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
"""