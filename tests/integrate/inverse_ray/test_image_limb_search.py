import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.image_area_all import image_area_all
from microjax.image_limb_search import image_limb_search

w_center = jnp.complex128(-0.14 - 0.1j)
q  = 0.5
s  = 1.0
rho = 0.3

a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}
NBIN = 10
Nlimbs = 100

import time
start_time = time.time()
area_all, magnification, carry = image_limb_search(w_center, rho, NBIN=NBIN, Nlimbs=Nlimbs, nlenses=2, **_params)
(yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
jax.device_put(area_all).block_until_ready()
end_time = time.time()
print(f"Execution time: %.2f{end_time - start_time} seconds")


w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, Nlimbs)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, q=q, s=s)
fig = plt.figure()
ax = plt.axes()
source = plt.Circle((w_center.real, w_center.imag), rho, color='b', fill=False)
ax.add_patch(source)
plt.plot(w_center.real, w_center.imag, "*", color="k")

plt.plot(-q * s, 0 , ".",c="k")
plt.plot((1.0 - q) * s, 0 ,".",c="k")

mask_x = area_x>0
cmap = plt.get_cmap("coolwarm_r")
pos_neg = jnp.where(dys[mask_x] > 0, 1.0, 0.0)
for i in range(len(xmin[mask_x])):
    #plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i])
    plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i], color=cmap(pos_neg[i]))
    plt.plot(xmin[mask_x][i], y[mask_x][i], ".", color="None", mec="gray")
    plt.plot(xmax[mask_x][i], y[mask_x][i], ".", color="None", mec="k")

plt.scatter(image_limb[mask].ravel().real, image_limb[mask].ravel().imag, s=1,color="purple")
plt.scatter(cau_tri.ravel().real, cau_tri.ravel().imag,   marker=".", color="red", s=1)
plt.scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1)

w_center_mid = w_center - 0.5 * s * (1 - q) / (1 + q) 
z_inits_mid, z_mask = _images_point_source(w_center_mid, **_params)
z_inits = z_inits_mid + 0.5 * s * (1 - q) / (1 + q)
for i in range(len(z_inits[z_mask])):
        plt.scatter(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, marker="*", zorder=2, ec="k")
        plt.text(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, s="%d"%(i), zorder=2)
plt.axis("equal")
plt.show()
