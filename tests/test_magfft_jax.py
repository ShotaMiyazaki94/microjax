import numpy as np
import matplotlib.pyplot as plt
from microjax.fastlens.mag_fft_jax import magnification_disk, magnification_limb

mag_disk = magnification_disk()
#mag_limb1= magnification_limb(1)
#mag_limb2= magnification_limb(2)

rho = 2
u = np.linspace(0.0, 5.0 ,1000)
a_disk = mag_disk.A(u, rho)
#a_limb1 = mag_limb1.A(u, rho)
#a_limb2 = mag_limb2.A(u, rho)

plt.figure()
plt.xlabel(r'$u$')
plt.ylabel(r'$A(u)$')
#plt.yscale('log')
plt.plot(u, a_disk,label="uniform")
#plt.plot(u, a_limb1,label="limb 1st order")
#plt.plot(u, a_limb2,label="limb 2nd order")
plt.legend()
plt.show()

k = mag_disk.k
apk = mag_disk.apk
plt.figure()
plt.xlabel(r'$k$')
plt.ylabel(r'$s(k)$')
plt.loglog(k, apk*k**2*mag_disk.sk(k, rho) )
#plt.loglog(k, apk*k**2*mag_limb1.sk(k, rho) )
#plt.loglog(k, apk*k**2*mag_limb2.sk(k, rho) )
plt.show()

