import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt

from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.inverse_ray.image_area0 import image_area0

w_center = jnp.complex128(-0.14 - 0.1j)
q  = 0.5
s  = 1.0
rho = 0.1
a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}

N_limb = 1000
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, q=q, s=s)

# construct r-range!
image_start = image_limb[mask].ravel()
r_is = jnp.sqrt(image_start.real**2 + image_start.imag**2)



r  = rho * 0.1
th = jnp.arctan(r) 
r_ = jnp.arange(0, 1.5, r)
th_ = jnp.arange(0, 2*jnp.pi, th)
r_grid, th_grid = jnp.meshgrid(r_, th_) 
x_grid = r_grid.ravel()*jnp.cos(th_grid.ravel())
y_grid = r_grid.ravel()*jnp.sin(th_grid.ravel())
#x_ = jnp.arange(-1.5, 1.5, r)
#y_ = jnp.arange(-1.5, 1.5, r)
#x_grid, y_grid = jnp.meshgrid(x_, y_)
z_mesh = x_grid.ravel() + 1j * y_grid.ravel()
print(z_mesh.shape)

source_mesh = lens_eq(z_mesh - 0.5*s*(1 - q)/(1 + q), **_params) 
source_mask = jnp.abs(source_mesh - w_center + 0.5*s*(1 - q)/(1 + q)) < rho

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
source = plt.Circle((w_center.real, w_center.imag), rho, color='b', fill=False)
ax.add_patch(source)
plt.plot(w_center.real, w_center.imag, "*", color="k")
plt.plot(-q * s, 0 , ".",c="k")
plt.plot((1.0 - q) * s, 0 ,".",c="k")
plt.scatter(image_limb[mask].ravel().real, image_limb[mask].ravel().imag, s=1,color="purple")
plt.scatter(cau_tri.ravel().real, cau_tri.ravel().imag,   marker=".", color="red", s=1)
plt.scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1)
plt.axis("equal")
plt.scatter(z_mesh[source_mask].real, z_mesh[source_mask].imag, s=1, marker=".", zorder=-1)
plt.show()