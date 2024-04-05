import numpy as np

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib.pyplot as plt
from jax import jit, vmap, grad, jacfwd
from microjax.point_source import mag_point_source, critical_and_caustic_curves
import MulensModel as mm
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt


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

def get_mag_binary(params):
    u0, t0, tE, s, q, alpha = params
    
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    
    return w_points, mag_point_source(w_points, nlenses=2, q=q, s=s) 

def get_mag_triple(params):
    u0, t0, tE, s, q, alpha, q3, r3, psi = params
    
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    
    return w_points, mag_point_source(w_points, nlenses=3, q=q, s=s, q3=q3, r3=jnp.abs(r3), psi=psi) 

params_bin = jnp.array([u0, t0, tE, s, q, alpha])
params_tri = jnp.array([u0, t0, tE, s, q, alpha, q3, jnp.abs(r3), psi])

mag_jac_bin = jit(jacfwd(lambda params: get_mag_binary(params)[1]))
mag_jac_tri = jit(jacfwd(lambda params: get_mag_triple(params)[1]))

jac_eval_bin = mag_jac_bin(params_bin)
jac_eval_tri = mag_jac_tri(params_tri)

mosaic="""
AG
AG
AG
BG
CG
DG
EG
FG
HG
"""
labels = [
    r'$A(t)$',
    r'$\frac{\partial A}{\partial u_0}$', r'$\frac{\partial A}{\partial t_0}$',r'$\frac{\partial A}{\partial t_E}$',
    r'$\frac{\partial A}{\partial q}$', r'$\frac{\partial A}{\partial s}$', r'$\frac{\partial A}{\partial \alpha}$'
]
fig,ax = plt.subplot_mosaic(mosaic=mosaic,figsize=(12,6))
ax["A"].plot(t, mag1,c="red",label="binary-lens")
ax["H"].set_xlabel("time (day)")
ax["A"].set_ylabel("magnification")
ax["A"].set_yscale("log")
lists=["B","C","D","E","F","H"]
for i, l in enumerate(lists):
    ax[l].plot(t,jac_eval_bin[:,i])
    ax[l].set_ylabel(labels[i+1])
ax["G"].scatter(cau1.ravel().real, cau1.ravel().imag,   marker=".", color="red", s=1)
ax["G"].scatter(crit1.ravel().real, crit1.ravel().imag, marker=".", color="green", s=1)
ax["G"].plot(w.real, w.imag,)
ax["G"].plot(-q*s, 0 ,".",c="k")
ax["G"].plot((1.0-q)*s, 0 ,".",c="k")
ax["G"].axis("equal")
plt.show()

mosaic="""
AJ
AJ
AJ
AJ
BJ
CJ
DJ
EJ
FJ
GJ
HJ
IJ
KJ
"""
labels = [
    r'$A(t)$',
    r'$\frac{\partial A}{\partial u_0}$', r'$\frac{\partial A}{\partial t_0}$',r'$\frac{\partial A}{\partial t_E}$',
    r'$\frac{\partial A}{\partial q}$', r'$\frac{\partial A}{\partial s}$', r'$\frac{\partial A}{\partial \alpha}$',
    r'$\frac{\partial A}{\partial q_3}$', r'$\frac{\partial A}{\partial r_3}$', r'$\frac{\partial A}{\partial \psi}$'
]
fig,ax = plt.subplot_mosaic(mosaic=mosaic,figsize=(12,6))
ax["A"].plot(t, mag3,c="red",label="triple-lens")
ax["K"].set_xlabel("time (day)")
ax["A"].set_ylabel("magnification")
ax["A"].set_yscale("log")
lists=["B","C","D","E","F","G","H","I","K"]
for i, l in enumerate(lists):
    ax[l].plot(t,jac_eval_tri[:,i])
    ax[l].set_ylabel(labels[i+1])
ax["J"].plot(w.real, w.imag,)
ax["J"].plot(-q*s, 0 ,".",c="k")
ax["J"].plot((1.0-q)*s, 0 ,".",c="k")
ax["J"].plot(r3.real - (0.5*s - s/(1 + q)), r3.imag ,".",c="k")
ax["J"].scatter(cau3.ravel().real, cau3.ravel().imag,   marker=".", color="red", s=1)
ax["J"].scatter(crit3.ravel().real, crit3.ravel().imag, marker=".", color="green", s=1)
ax["J"].axis("equal")
plt.show()
