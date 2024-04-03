import numpy as np

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib.pyplot as plt
from jax import jit, vmap, grad
from microjax.point_source import mag_point_source, critical_and_caustic_curves
import MulensModel as mm
jax.config.update("jax_enable_x64", True)


t  =  np.linspace(-10, 10, 500)

s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.1  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
q3 = 1e-2
r3 = 0.3+1.2j 
psi = jnp.arctan2(r3.imag, r3.real)

#rho = 0.01 # source radius in Einstein radii of the total mass.
alpha = np.deg2rad(40) # angle between lens axis and source trajectory
tE = 10 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.2 # impact parameter

# Position of the center of the source with respect to the center of mass.
tau = (t - t0)/tE
y1 = -u0*np.sin(alpha) + tau*np.cos(alpha)
y2 = u0*np.cos(alpha) + tau*np.sin(alpha)

w = jnp.array(y1 + 1j * y2, dtype=complex)


mag1 = mag_point_source(w, nlenses=2, q=q, s=s, q3=q3, r3=jnp.abs(r3), psi=psi)
crit1, cau1 = critical_and_caustic_curves(npts=1000, nlenses=2, q=q, s=s, q3=q3, r3=jnp.abs(r3), psi=psi)
mag3 = mag_point_source(w, nlenses=3, q=q, s=s, q3=q3, r3=jnp.abs(r3), psi=psi)
crit3, cau3 = critical_and_caustic_curves(npts=1000, nlenses=3, q=q, s=s, q3=q3, r3=jnp.abs(r3), psi=psi)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,2, figsize=(12,6)) 
ax[0].plot(t, mag1,c="red",label="binary-lens")
ax[0].plot(t, mag3,"--",c="k",label="triple-lens")
ax[0].legend()
ax[0].set_xlabel("time (day)")
ax[0].set_ylabel("magnification")
ax[0].set_yscale("log")
ax[1].plot(w.real, w.imag,)
ax[1].plot(-q*s, 0 ,".",c="k")
ax[1].plot((1.0-q)*s, 0 ,".",c="k")
ax[1].plot(r3.real - (0.5*s - s/(1 + q)), r3.imag ,".",c="k")
#ax[1].scatter(cau1.ravel().real, cau1.ravel().imag,   marker=".", color="r", s=1)
#ax[1].scatter(crit1.ravel().real, crit1.ravel().imag, marker=".", color="g", s=1)
ax[1].scatter(cau3.ravel().real, cau3.ravel().imag,   marker=".", color="red", s=1)
ax[1].scatter(crit3.ravel().real, crit3.ravel().imag, marker=".", color="green", s=1)
ax[1].axis("equal")
plt.show()