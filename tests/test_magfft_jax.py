import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from microjax.fastlens.mag_fft_jax import magnification_disk, magnification_limb1
from microjax.fastlens.special import ellipk, ellipe
jax.config.update("jax_enable_x64", True)

mag_disk = magnification_disk()
mag_limb1= magnification_limb1()
#mag_limb2= magnification_limb(2)

rho = 2
print(mag_limb1.A0(rho))
print(ellipk(-rho**2/4))
print(ellipe(-rho**2/4))
u = jnp.linspace(0.0, 5.0 ,1000)
a_disk = mag_disk.A(u, rho)
a_limb1 = mag_limb1.A(u, rho)
#a_limb2 = mag_limb2.A(u, rho)
#a0 = mag_limb1.A0(rho)
#print(a0)
exit(0);
plt.figure()
plt.xlabel(r'$u$')
plt.ylabel(r'$A(u)$')
#plt.yscale('log')
plt.plot(u, a_disk,label="uniform")
plt.plot(u, a_limb1,label="limb 1st order")
#plt.plot(u, a_limb2,label="limb 2nd order")
plt.legend()
plt.show()

k = mag_disk.k
apk = mag_disk.apk
plt.figure()
plt.xlabel(r'$k$')
plt.ylabel(r'$s(k)$')
plt.loglog(k, apk*k**2*mag_disk.sk(k, rho) )
plt.loglog(k, apk*k**2*mag_limb1.sk(k, rho) )
#plt.loglog(k, apk*k**2*mag_limb2.sk(k, rho) )
plt.show()

