import jax.numpy as jnp
import jax
import numpy as np
from microjax.fastlens.mag_fft_jax import magnification_disk, mag_limb1, magnification_limb1, magnification_limb2
jax.config.update("jax_enable_x64", True)
from microjax.fastlens.mag_fft import magnification_disk as magnification_disk_org 
from microjax.fastlens.mag_fft import magnification_limb as magnification_limb_org 
from jax import jit
import timeit

# JAX関数の事前コンパイル
mag_disk_jax = jit(magnification_disk().A)
mag_disk_o = magnification_disk_org().A

mag_limb1_jax = jit(mag_limb1().A)
mag_limb1_o = magnification_limb_org(1).A

mag_limb2_jax = jit(magnification_limb2().A)
mag_limb2_o = magnification_limb_org(2).A

u_rho = np.logspace(-1, 1, 10000)
rho = np.array([1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10])

def measure_time(f, args):
    return timeit.timeit('f(*args)', globals={'f': f, 'args': args}, number=100)

def warmup(f, args):
    f(*args)

for r in rho:
    u = u_rho * r
    args = (u, r)
    warmup(mag_disk_jax, args)
    warmup(mag_disk_o, args)
    warmup(mag_limb1_jax, args)
    warmup(mag_limb1_o, args)
    warmup(mag_limb2_jax, args)
    warmup(mag_limb2_o, args)

for r in rho:
    u = u_rho * r
    args_jax = (u, r)
    print("------------------------")
    print("rho    :", r)
    print("disk")
    result0 = measure_time(mag_disk_jax, args_jax)
    result1 = measure_time(mag_disk_o, args_jax)
    print("JAX | ORG (s): %.2e | %.2e" % (result0 / 100.0, result1/100))
    print("JAX/ORG: %.2f" % (result0 / result1))
    
    print("limb linear")
    result0 = measure_time(mag_limb1_jax, args_jax)
    result1 = measure_time(mag_limb1_o, args_jax)
    print("JAX | ORG (s): %.2e | %.2e" % (result0 / 100.0, result1/100))
    print("JAX/ORG: %.2f" % (result0 / result1))
    
    print("limb quad")
    result0 = measure_time(mag_limb2_jax, args_jax)
    result1 = measure_time(mag_limb2_o, args_jax)
    print("JAX | ORG (s): %.2e | %.2e" % (result0 / 100.0, result1/100))
    print("JAX/ORG: %.2f" % (result0 / result1))
