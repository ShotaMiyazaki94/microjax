import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from microjax.fastlens.mag_fft_jax import magnification_disk, magnification_limb1,magnification_limb2
jax.config.update("jax_enable_x64", True)
from microjax.fastlens.mag_fft import magnification_disk as magnification_disk_org 
from microjax.fastlens.mag_fft import magnification_limb as magnification_limb_org 
from jax import jit, grad, vmap
import timeit

mag_disk  = magnification_disk()
mag_disk_o= magnification_disk_org()

mag_limb1  = magnification_limb1()
mag_limb1_o= magnification_limb_org(1)

mag = jit(mag_disk.A)

def sum_A(u,rho):
    return jnp.sum(mag(u,rho))

mag_grad = jit(vmap(grad(sum_A, argnums=(0,1))))

u_grid = jnp.logspace(-4,1,50)
r_grid = jnp.logspace(-5,0,50)
u_grid, r_grid = jnp.meshgrid(u_grid,r_grid)
u_grid = u_grid.ravel()
r_grid = r_grid.ravel()
result = mag_grad(u_grid,r_grid)
result0 = jnp.nan_to_num(result[0],nan=-1.0)
result1 = jnp.nan_to_num(result[1],nan=-1.0)

from matplotlib.colors import LogNorm

fig,ax = plt.subplots(1,2,figsize=(14,5))
sc = ax[0].scatter(u_grid,r_grid,c=result0,norm=LogNorm(),ec="k")
ax[0].loglog()
plt.colorbar(sc,ax=ax[0])
sc1 = ax[1].scatter(u_grid,r_grid,c=result1,norm=LogNorm(),ec="k")
ax[1].loglog()
plt.colorbar(sc1,ax=ax[1])
ax[0].set_xlabel("u")
ax[0].set_ylabel("rho")
ax[1].set_xlabel("u")
ax[1].set_ylabel("rho")
#plt.savefig("./a.png",dpi=200)
plt.show()