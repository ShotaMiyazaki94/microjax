import numpy as np

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib.pyplot as plt

from microjax.point_source import critical_and_caustic_curves, mag_point_source
import MulensModel as mm
jax.config.update("jax_enable_x64", True)

q  = 1e-1
s  = 0.95
q3 = 1e-3
r3 = 1.15
psi = np.deg2rad(10)

crit, cau = critical_and_caustic_curves(npts=1000, nlenses=3, q=q, s=s, q3=q3, r3=r3, psi=psi)
print(crit.shape, cau.shape)

import matplotlib.pyplot as plt

plt.plot(-q*s, 0 ,".",c="k")
plt.plot((1.0-q)*s, 0 ,".",c="k")
plt.plot(r3*np.cos(psi) + 0.5*s - q*s, r3*np.sin(psi) ,".",c="k")
plt.scatter(cau.ravel().real, cau.ravel().imag,   marker=".", color="r", s=1)
plt.scatter(crit.ravel().real, crit.ravel().imag, marker=".", color="g", s=1)
plt.axis("equal")
plt.show()