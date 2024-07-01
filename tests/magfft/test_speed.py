import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from microjax.fastlens.mag_fft_jax import magnification_disk, mag_limb1, magnification_limb1, magnification_limb2
from microjax.fastlens.mag_fft import magnification_disk as magnification_disk_org 
from microjax.fastlens.mag_fft import magnification_limb as magnification_limb_org 
from jax import jit
import timeit

jax.config.update("jax_enable_x64", True)

# JAX versions
mag_disk = magnification_disk()
mag_limb1 = mag_limb1()
mag_limb2 = magnification_limb2()

# Original versions
mag_disk_o = magnification_disk_org()
mag_limb1_o = magnification_limb_org(1)
mag_limb2_o = magnification_limb_org(2)

u_rho = jnp.logspace(-1, 1, 10000)
rho = jnp.array([1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10])

# JIT compiled functions
mag = jit(mag_disk.A)
mag_d = jit(mag_limb1.A)
mag_d2 = jit(mag_limb2.A)

def time_function(func, *args):
    func(*args).block_until_ready()  # Ensure synchronization
    return timeit.timeit(lambda: func(*args).block_until_ready(), number=100) / 100.0

def print_results(rho, jax_time, org_time, label):
    print(f"rho    : {rho}")
    print(f"{label}")
    print(f"JAX (s): {jax_time:.2e}")
    print(f"ORG (s): {org_time:.2e}")
    print(f"JAX/ORG: {jax_time/org_time:.2f}")

for r in rho:
    u = u_rho * r

    # Disk magnification
    result0 = time_function(mag, u, r)
    result1 = time_function(mag_disk_o.A, u, r)
    print("------------------------")
    print_results(r, result0, result1, "disk")

    # Limb linear magnification
    result0 = time_function(mag_d, u, r)
    result1 = time_function(mag_limb1_o.A, u, r)
    print_results(r, result0, result1, "limb linear")

    # Limb quadratic magnification
    result0 = time_function(mag_d2, u, r)
    result1 = time_function(mag_limb2_o.A, u, r)
    print_results(r, result0, result1, "limb quad")

exit(0)







import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from microjax.fastlens.mag_fft_jax import magnification_disk, mag_limb1, magnification_limb1,magnification_limb2
jax.config.update("jax_enable_x64", True)
from microjax.fastlens.mag_fft import magnification_disk as magnification_disk_org 
from microjax.fastlens.mag_fft import magnification_limb as magnification_limb_org 
from jax import jit
import timeit

mag_disk  = magnification_disk()
mag_disk_o= magnification_disk_org()

mag_limb1  = mag_limb1()
mag_limb1_o= magnification_limb_org(1)
mag_limb2_o= magnification_limb_org(2)

mag_limb2  = magnification_limb2()

u_rho = np.logspace(-1,1,10000)
rho = np.array([1e-5,1e-4,1e-3,0.01,0.1,1,10])

mag = jit(mag_disk.A)
mag_d = jit(mag_limb1.A)
mag_d2 = jit(mag_limb2.A)

for r in rho:
    u = u_rho * r
    result0 = timeit.timeit(lambda: mag(u,r),number=100)  
    result1 = timeit.timeit(lambda: mag_disk_o.A(u,r),number=100) 
    print("------------------------")
    print("rho    :", r)
    print("disk")
    print("JAX (s): %.2e"%(result0/100.0))
    print("ORG (s): %.2e"%(result1/100.0))
    print("JAX/ORG: %.2f"%(result0/result1))
    result0 = timeit.timeit(lambda: mag_d(u,r),number=100)  
    result1 = timeit.timeit(lambda: mag_limb1_o.A(u,r),number=100)  
    print("limb linear")
    print("JAX (s): %.2e"%(result0/100.0))
    print("ORG (s): %.2e"%(result1/100.0))
    print("JAX/ORG: %.2f"%(result0/result1))
    result0 = timeit.timeit(lambda: mag_d2(u,r),number=100)  
    result1 = timeit.timeit(lambda: mag_limb2_o.A(u,r),number=100)  
    print("limb quad")
    print("JAX (s): %.2e"%(result0/100.0))
    print("ORG (s): %.2e"%(result1/100.0))
    print("JAX/ORG: %.2f"%(result0/result1))



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
plt.savefig("tests/magfft/test_speed.png",dpi=200,bbox_inches="tight")
plt.close()

