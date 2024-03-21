import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from microjax.fastlens.mag_fft_jax import magnification_disk, magnification_limb1,magnification_limb2
from microjax.fastlens.special import ellipk, ellipe
jax.config.update("jax_enable_x64", True)
from microjax.fastlens.mag_fft import magnification_disk as magnification_disk_org 
from microjax.fastlens.mag_fft import magnification_limb as magnification_limb_org 

mag_disk = magnification_disk()
mag_limb1= magnification_limb1()
mag_limb2= magnification_limb2()

mag_disk_org = magnification_disk_org()
mag_limb1_org = magnification_limb_org(1)
mag_limb2_org = magnification_limb_org(2)

rho = 2.5
u = jnp.linspace(-5.0, 5.0 ,1000)
a_disk = mag_disk.A(u, rho)
a_limb1 = mag_limb1.A(u, rho)
a_limb2 = mag_limb2.A(u, rho)

a_disk_org = mag_disk_org.A(u,rho)
a_limb1_org = mag_limb1_org.A(u,rho)
a_limb2_org = mag_limb2_org.A(u,rho)

plt.figure()
plt.xlabel(r'$u$')
plt.ylabel(r'$A(u)$')
#plt.yscale('log')
plt.plot(u, a_disk,label="uniform (jax)")
plt.plot(u, a_disk_org,"--",label="uniform (org)")
plt.plot(u, a_limb1,label="limb 1st order (jax)")
plt.plot(u, a_limb1_org,"--",label="limb 1st orde (org)")
plt.plot(u, a_limb2,label="limb 2nd order (jax)")
plt.plot(u, a_limb2_org,"--",label="limb 2nd order (org)")
plt.legend()
plt.show()


plt.figure()
plt.xlabel(r'$u$')
plt.ylabel(r'$A(u)$')
#plt.yscale('log')
plt.plot(u, a_disk-a_disk_org,label="uniform (diff)")
plt.plot(u, a_limb1-a_limb1_org,label="limb 1st order (diff)")
plt.plot(u, a_limb2-a_limb2_org,label="limb 2nd order (diff)")
plt.legend()
plt.show()


k = mag_disk.k
apk = mag_disk.apk
plt.figure()
plt.xlabel(r'$k$')
plt.ylabel(r'$s(k)$')
plt.loglog(k, apk*k**2*mag_disk.sk(k, rho) )
plt.loglog(k, apk*k**2*mag_limb1.sk(k, rho) )
plt.loglog(k, apk*k**2*mag_limb2.sk(k, rho) )
plt.show()

