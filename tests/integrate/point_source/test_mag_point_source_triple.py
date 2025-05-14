import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib.pyplot as plt
from jax import jit, vmap, grad, jacfwd
from microjax.point_source import mag_point_source, critical_and_caustic_curves
import MulensModel as mm
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.1  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
q3 = 5e-3
r3_complex = 0.3 + 1.2j 
psi = jnp.arctan2(r3_complex.imag, r3_complex.real)

alpha = jnp.deg2rad(40) # angle between lens axis and source trajectory
tE = 10 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.2 # impact parameter
t  =  t0 + jnp.linspace(-tE, tE, 500)

# Position of the center of the source with respect to the center of mass.
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
w = jnp.array(y1 + 1j * y2, dtype=complex)

import time
_params = {"q": q, "s": s, "q3": q3, "r3": jnp.abs(r3_complex), "psi": psi}
_ = mag_point_source(w, nlenses=3, **_params)
print("mag start")
start = time.time()
mag_tri = mag_point_source(w, nlenses=3, **_params)
mag_tri.block_until_ready()
end = time.time()
print("mag finish: ", end-start, "sec")
crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, nlenses=3, **_params)

def get_mag_triple(params):
    u0, t0, tE, s, q, alpha, q3, r3, psi = params    
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    _params = {"q": q, "s": s, "q3": q3, "r3": r3, "psi": psi}    
    return w_points, mag_point_source(w_points, nlenses=3, **_params) 

params_triple = jnp.array([u0, t0, tE, s, q, alpha, q3, jnp.abs(r3_complex), psi])
mag_jac_tri = jit(jacfwd(lambda params: get_mag_triple(params)[1])) 
_ = mag_jac_tri(params_triple)
print("jac start")
start = time.time()
jac_eval_tri = mag_jac_tri(params_triple)
jac_eval_tri.block_until_ready()
end = time.time()
print("jac finish:", end-start, "sec")
#jac_eval_tri = mag_jac_tri(jnp.array([u0, t0, tE, s, q, alpha, q3, jnp.abs(r3), psi]))

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
ax["A"].plot(t, mag_tri,c="red",label="triple-lens")
ax["K"].set_xlabel("time (day)")
ax["A"].set_ylabel("magnification")
#ax["A"].set_yscale("log")
lists=["B","C","D","E","F","G","H","I","K"]
for i, l in enumerate(lists):
    ax[l].plot(t,jac_eval_tri[:,i])
    ax[l].set_ylabel(labels[i+1])
ax["J"].plot(w.real, w.imag,)
ax["J"].plot(-q*s, 0 ,".",c="k")
ax["J"].plot((1.0-q)*s, 0 ,".",c="k")
ax["J"].plot(r3_complex.real - (0.5*s - s/(1 + q)), r3_complex.imag ,".",c="k")
ax["J"].scatter(cau_tri.ravel().real, cau_tri.ravel().imag,   marker=".", color="red", s=1)
ax["J"].scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1)
ax["J"].axis("equal")
plt.savefig("tests/integrate/point_source/test_mag_point_source_triple.pdf",bbox_inches="tight")
plt.show()
plt.close()
