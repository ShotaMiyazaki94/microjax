import numpy as np

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib.pyplot as plt

from microjax.point_source import critical_and_caustic_curves
import MulensModel as mm
jax.config.update("jax_enable_x64", True)

q  = 1e+0
s  = 1.0
q3 = 1e-2
r3 = 1.0
psi = np.deg2rad(45)


crit, cau = critical_and_caustic_curves(npts=1000, nlenses=3, q=q, s=s, q3=q3, r3=r3, psi=psi)

import matplotlib.pyplot as plt

plt.scatter(cau.ravel().real, cau.ravel().imag,   marker=".", color="r", s=1)
plt.scatter(crit.ravel().real, crit.ravel().imag, marker=".", color="k", s=1)
plt.show()