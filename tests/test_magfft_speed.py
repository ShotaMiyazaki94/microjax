import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from microjax.fastlens.mag_fft_jax import magnification_disk, magnification_limb1,magnification_limb2
jax.config.update("jax_enable_x64", True)
from microjax.fastlens.mag_fft import magnification_disk as magnification_disk_org 
from microjax.fastlens.mag_fft import magnification_limb as magnification_limb_org 
from jax import jit
import timeit

mag_disk  = magnification_disk()
mag_disk_o= magnification_disk_org()

mag_limb1  = magnification_limb1()
mag_limb1_o= magnification_limb_org(1)

u_rho = np.logspace(-2,2,1000)
rho = np.array([1e-4,1e-3,0.01,0.1,1,10])

mag = jit(mag_disk.A)

for r in rho:
    u = u_rho * r
    result0 = timeit.timeit(lambda: mag(u,r),number=100)  
    result1 = timeit.timeit(lambda: mag_disk_o.A(u,r),number=100)  
    print("------------------------")
    print("rho    :", r)
    print("JAX (s):", result0/100.0)
    print("JAX/ORG:", result0/result1)

exit(0)
fig, ax = plt.subplots(2,6, figsize=(24, 8),sharex=True,sharey=True)

for i,r in enumerate(rho):
    u = u_rho * r
    u_j = jnp.array(u)
    mag_ = mag_disk_o.A(u,r) 
    mag  = mag_disk.A(u_j,jnp.array(r))
    ax[0,i].plot(u_rho,jnp.abs(mag-mag_))
    ax[0,i].set_xlabel("u/rho")
    ax[0,i].set_title("rho=%.1e"%(r))
    ax[0,i].grid(ls="--")

for i,r in enumerate(rho):
    u = u_rho * r
    u_j = jnp.array(u)
    mag_ = mag_limb1_o.A(u,r) 
    mag  = mag_limb1.A(u_j,jnp.array(r))
    ax[1,i].plot(u_rho,jnp.abs(mag-mag_))
    ax[1,i].set_xlabel("u/rho")
    ax[1,i].set_title("rho=%.1e"%(r))
    ax[1,i].grid(ls="--")
ax[0,0].set_yscale("log")
ax[0,0].set_xscale("log")
ax[0,0].set_xlim(1e-2,100)
ax[0,0].set_ylabel("mag difference\n(uniform disk)")
ax[1,0].set_ylabel("mag difference\n(linear limb-darkening)")
plt.show()

