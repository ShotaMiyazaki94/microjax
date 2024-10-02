import jax 
from jax import lax, vmap, pmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
jax.config.update("jax_enable_x64", True)
from microjax.utils import match_points

w_center = jnp.complex128(-0.1 + 0.0j)
q = 0.1
s = 1.0
rho = 1e-3
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"q": q, "s": s, "a": a, "e1": e1}

N_limb = 100
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q)  # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1)  # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)  # center-of-mass coordinate

crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, q=q, s=s)

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
source = plt.Circle((w_center.real, w_center.imag), rho, color='b', fill=False)
ax.add_patch(source)
plt.title(r"$q={%.2f}, s={%.2f}, \rho={%.5f}$"%(q,s,rho))
plt.plot(-q/(1.0 + q) * s, 0 , "x",c="None", mec="k")
plt.plot(1.0/(1.0 + q) * s, 0 ,"x",c="gray")
plt.scatter(cau_tri.ravel().real, cau_tri.ravel().imag, marker=".", color="red", s=1)
plt.scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1)
#plt.plot(image_limb[mask].ravel().real, image_limb[mask].ravel().imag,color="purple")
plt.scatter(image_limb[mask].ravel().real, image_limb[mask].ravel().imag, s=10, color="None", ec="purple")
plt.axis("equal")
plt.grid(ls="--")
plt.show()
