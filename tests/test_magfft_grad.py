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
jax.config.update("jax_debug_nans", True) 

mag_disk  = magnification_disk(rho_switch=1e-5)
#mag_disk_o= magnification_disk_org()

mag_limb1  = magnification_limb1(rho_switch=1e-5)
#mag_limb1_o= magnification_limb_org(1)

mag = mag_limb1.A
#mag = mag_disk.A

def mag_scaler(u,rho):
    mag_ = mag(u,rho)
    return mag_[0] 

mag_grad = jit(vmap(grad(mag_scaler,argnums=(0,1))))
tmp = jit(grad(mag_scaler,argnums=(0,1)))
#print(mag(1e-3,1e-3))
#print(tmp(1e-1,1e-5))

u_grid = jnp.logspace(-6,0,50)
r_grid = jnp.logspace(-5,1,50)
u_grid, r_grid = jnp.meshgrid(u_grid,r_grid)
u_grid = u_grid.ravel()
r_grid = r_grid.ravel()
result = mag_grad(u_grid,r_grid)
result0 = jnp.array(result[0])
result1 = jnp.array(result[1])

from matplotlib.colors import LogNorm

fig,ax = plt.subplots(1,2,figsize=(14,5))
#sc = ax[0].scatter(u_grid,r_grid,c=result0,ec="k")
sc = ax[0].scatter(u_grid,r_grid,c=np.abs(result0),norm=LogNorm(),ec="k")
ax[0].loglog()
cb0 = plt.colorbar(sc,ax=ax[0])
cb0.set_label("|dA/du|")
#sc1 = ax[1].scatter(u_grid,r_grid,c=result1,ec="k")
sc1 = ax[1].scatter(u_grid,r_grid,c=np.abs(result1),norm=LogNorm(),ec="k")
ax[1].loglog()
cb1 = plt.colorbar(sc1,ax=ax[1])
cb1.set_label("|dA/drho|")
ax[0].set_xlabel("u")
ax[0].set_ylabel("rho")
ax[1].set_xlabel("u")
ax[1].set_ylabel("rho")
#plt.savefig("./a.png",dpi=200)
plt.show()

mosaic="""
AAAAAA
AAAAAA
AAAAAA
BBBBBB
CCCCCC
"""

u   = jnp.linspace(1e-5,5,1000)
rho = jnp.ones(1000) * 1

mag_plot = vmap(mag)(u,rho)
mag_grad_plot = mag_grad(u,rho)

fig,ax = plt.subplot_mosaic(figsize=(6,5),mosaic=mosaic)
fig.subplots_adjust(hspace=0)
ax["A"].plot(u,jnp.ravel(mag_plot))
ax["B"].plot(u,mag_grad_plot[0].ravel(),"-")
ax["C"].plot(u,mag_grad_plot[1].ravel(),"-")
ax["A"].set_ylabel("magnification")
ax["B"].set_ylabel("dA/du")
ax["C"].set_ylabel("dA/drho")
ax["C"].set_xlabel("u")
plt.show()