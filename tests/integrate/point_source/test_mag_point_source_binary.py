import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib.pyplot as plt
from jax import jit, vmap, grad, jacfwd
from microjax.point_source import critical_and_caustic_curves
from microjax.point_source import mag_point_source
import MulensModel as mm
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.1  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
q3 = 1e-2
r3 = 0.3+1.2j 
psi = jnp.arctan2(r3.imag, r3.real)

alpha = jnp.deg2rad(40) # angle between lens axis and source trajectory
tE = 10 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.2 # impact parameter
t  =  t0 + jnp.linspace(-tE, tE, 1000)

# Position of the center of the source with respect to the center of mass.
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)

w = jnp.array(y1 + 1j * y2, dtype=complex)

_params = {"q": q, "s": s}
mag_binary = mag_point_source(w, nlenses=2, **_params) 
#mag_binary = mag_point_source_binary(w, s=s, q=q) 
crit_bin, cau_bin = critical_and_caustic_curves(npts=1000, nlenses=2, **_params)
#crit_bin, cau_bin = critical_and_caustic_curves_binary(npts=1000,)

def get_mag_binary(params):
    u0, t0, tE, s, q, alpha = params
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    _params = {"q": q, "s": s} 
    return w_points, mag_point_source(w_points, nlenses=2, **_params) 
    #return w_points, mag_point_source_binary(w_points, s=s, q=q) 

params_binary = jnp.array([u0, t0, tE, s, q, alpha])
mag_jac_bin = jit(jacfwd(lambda params: get_mag_binary(params)[1]))
jac_eval_bin = mag_jac_bin(params_binary)

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
ax["A"].plot(t, mag_binary,c="red",label="binary-lens")
ax["H"].set_xlabel("time (day)")
ax["A"].set_ylabel("magnification")
ax["A"].set_yscale("log")
lists=["B","C","D","E","F","H"]
for i, l in enumerate(lists):
    ax[l].plot(t,jac_eval_bin[:,i])
    ax[l].set_ylabel(labels[i+1])
ax["G"].scatter(cau_bin.ravel().real, cau_bin.ravel().imag,   marker=".", color="red", s=1)
ax["G"].scatter(crit_bin.ravel().real, crit_bin.ravel().imag, marker=".", color="green", s=1)
ax["G"].plot(w.real, w.imag,)
ax["G"].plot(-q*s, 0 ,".",c="k")
ax["G"].plot((1.0-q)*s, 0 ,".",c="k")
ax["G"].axis("equal")
plt.savefig("tests/integrate/point_source/test_mag_point_source_binary.pdf",bbox_inches="tight")
#plt.show()
plt.close()
