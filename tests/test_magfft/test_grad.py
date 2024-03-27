import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from microjax.fastlens.mag_fft_jax import magnification_disk, magnification_limb1, mag_limb1 
jax.config.update("jax_enable_x64", True)
from microjax.fastlens.mag_fft import magnification_disk as magnification_disk_org 
from microjax.fastlens.mag_fft import magnification_limb as magnification_limb_org 
from jax import jit, grad, vmap
import timeit
jax.config.update("jax_debug_nans", True) 
import seaborn as sns
sns.set_theme(font="serif",font_scale=1.,style="ticks",)

mag_disk = magnification_disk(rho_switch=1e-5)
mag_limb = mag_limb1(rho_switch=1e-5, a1=0.5)

magd = mag_disk.A
magl = mag_limb.A

def magl_scaler(u,rho):
    mag_ = magl(u,rho)
    return mag_[0]

def magd_scaler(u,rho):
    mag_ = magd(u,rho)
    return mag_[0]

magl_grad = jit(vmap(grad(magl_scaler,argnums=(0,1))))
magd_grad = jit(vmap(grad(magd_scaler,argnums=(0,1))))

rho_value = 1.0
u   = jnp.linspace(1e-3,3,1000)
rho = jnp.ones(1000) * rho_value
magd_plot = vmap(magd)(u,rho) 
magl_plot = vmap(magl)(u,rho) 

magd_plot_grad = magd_grad(u,rho)
magl_plot_grad = magl_grad(u,rho)

mosaic="""
AAAAA
AAAAA
AAAAA
BBBBB
CCCCC
"""

fig,ax = plt.subplot_mosaic(figsize=(7,5),mosaic=mosaic)
fig.subplots_adjust(hspace=0)
ax["A"].plot(u,jnp.ravel(magd_plot),label="disk")
ax["A"].plot(u,jnp.ravel(magl_plot),label="limb, a1=0.5")
ax["A"].legend()
ax["A"].set_ylabel("magnification")
ax["B"].plot(u,magd_plot_grad[0].ravel(),"-")
ax["B"].plot(u,magl_plot_grad[0].ravel(),"-")
ax["C"].plot(u,magd_plot_grad[1].ravel(),"-")
ax["C"].plot(u,magl_plot_grad[1].ravel(),"-")
ax["B"].set_ylabel("dA/du")
ax["C"].set_ylabel("dA/drho")
ax["C"].set_xlabel("u")
plt.show()

u_grid = jnp.logspace(-5,0,50)
r_grid = jnp.logspace(-5,1,50)
u_grid, r_grid = jnp.meshgrid(u_grid,r_grid)
u_grid = u_grid.ravel()
r_grid = r_grid.ravel()
result = magd_grad(u_grid,r_grid)
result0 = jnp.array(result[0])
result1 = jnp.array(result[1])

from matplotlib.colors import LogNorm

fig,ax = plt.subplots(1,2,figsize=(14,5))
sc = ax[0].scatter(u_grid,r_grid,c=np.abs(result0),norm=LogNorm(),ec="k")
ax[0].loglog()
cb0 = plt.colorbar(sc,ax=ax[0])
cb0.set_label("|dA/du|")
sc1 = ax[1].scatter(u_grid,r_grid,c=np.abs(result1),norm=LogNorm(),ec="k")
ax[1].loglog()
cb1 = plt.colorbar(sc1,ax=ax[1])
cb1.set_label("|dA/drho|")
ax[0].set_xlabel("u")
ax[0].set_ylabel("rho")
ax[1].set_xlabel("u")
ax[1].set_ylabel("rho")
plt.show()


exit(0)

mosaic="""
AAAAAA
AAAAAA
AAAAAA
BBBBBB
CCCCCC
"""

rho_value = 1.0
u   = jnp.linspace(-3,3,1000)
rho = jnp.ones(1000) * rho_value
mag_plot1 = vmap(mag)(u,rho)
mag_grad_plot1 = mag_grad(u,rho)

mag_plot1_c = vmap(mag_limb1_c.A)(u,rho) 
mag_plot1_c2 = vmap(mag_limb1_c2.A)(u,rho) 

rho_value = 0.75
u   = jnp.linspace(-3,3,1000)
rho = jnp.ones(1000) * rho_value
mag_plot75 = vmap(mag)(u,rho)
mag_grad_plot75 = mag_grad(u,rho)


rho_value = 0.5
u   = jnp.linspace(-3,3,1000)
rho = jnp.ones(1000) * rho_value
mag_plot05 = vmap(mag)(u,rho)
mag_grad_plot05 = mag_grad(u,rho)

plt.plot(jnp.ravel(mag_plot1) - jnp.ravel(mag_plot1_c))


fig,ax = plt.subplot_mosaic(figsize=(7,5),mosaic=mosaic)
fig.subplots_adjust(hspace=0)
ax["A"].plot(u,jnp.ravel(mag_plot1),label="rho=1.0")
ax["A"].plot(u,jnp.ravel(mag_plot1_c),label="rho=1.0, custom a1=0.5")
ax["A"].plot(u,jnp.ravel(mag_plot1_c2),label="rho=1.0, custom a1=0.3")
ax["B"].plot(u,mag_grad_plot1[0].ravel(),"-")
ax["C"].plot(u,mag_grad_plot1[1].ravel(),"-")
ax["A"].plot(u,jnp.ravel(mag_plot75),label="rho=0.75")
ax["B"].plot(u,mag_grad_plot75[0].ravel(),"-")
ax["C"].plot(u,mag_grad_plot75[1].ravel(),"-")
ax["A"].plot(u,jnp.ravel(mag_plot05),label="rho=0.5")
ax["B"].plot(u,mag_grad_plot05[0].ravel(),"-")
ax["C"].plot(u,mag_grad_plot05[1].ravel(),"-")
ax["A"].set_ylabel("magnification")
ax["B"].set_ylabel("dA/du")
ax["C"].set_ylabel("dA/drho")
ax["C"].set_xlabel("u")
ax["A"].legend()
plt.show()

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