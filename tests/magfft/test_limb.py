import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from microjax.fastlens.mag_fft_jax import mag_limb1, magnification_disk as mag_disk
jax.config.update("jax_enable_x64", True)
from jax import jit, grad, vmap
import timeit
jax.config.update("jax_debug_nans", True)
import VBBinaryLensing
import seaborn as sns
sns.set_theme(font="serif",font_scale=1.,style="ticks",)

VBBL = VBBinaryLensing.VBBinaryLensing()
VBBL.LoadESPLTable("tests/ESPL.tbl")

mag_disk = mag_disk()

mag_limb09  = mag_limb1(rho_switch=1e-5,a1=0.9)
mag_limb05  = mag_limb1(rho_switch=1e-5,a1=0.5)
mag_limb00  = mag_limb1(rho_switch=1e-5,a1=0.0)

rho_value = 1.5
u   = jnp.linspace(1e-3,5,1000)
rho = jnp.ones(1000) * rho_value
mag_plot   = vmap(mag_disk.A)(u,rho)
mag_plot09 = vmap(mag_limb09.A)(u,rho)
mag_plot05 = vmap(mag_limb05.A)(u,rho)
mag_plot00 = vmap(mag_limb00.A)(u,rho)
mag_VB = np.array([VBBL.ESPLMag(u_,r_) for u_, r_ in zip(u, rho)])

mosaic = """
AAAA
AAAA
BBBB
"""
fig,ax = plt.subplot_mosaic(mosaic=mosaic,figsize=(6,5))
fig.subplots_adjust(hspace=0)
ax["A"].plot(u, mag_plot,  label="disk")
ax["A"].plot(u, mag_VB,  label="disk (VBBinaryLensing)")
ax["A"].plot(u, mag_plot00,label="limb, a1=0.0",ls="--")
ax["A"].plot(u, mag_plot05,label="limb, a1=0.5",ls="-.")
ax["A"].plot(u, mag_plot09,label="limb, a1=0.9",ls=":")
ax["A"].legend()
ax["B"].plot(u, (mag_plot.ravel() - mag_plot00.ravel()))
ax["A"].set_ylabel("magnification")
ax["B"].set_ylabel("disk diff")
ax["B"].set_xlabel("u")
plt.tight_layout()
plt.show()