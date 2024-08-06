import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.inverse_ray import image_area0, image_area_all, CarryData

NBIN=100
nlenses=2

w_center = jnp.complex128(-0.05 - 0.1j)
q  = 0.5
s  = 1.0
a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}
rho = 0.13

incr  = jnp.abs(rho / NBIN)
incr2 = incr * 0.5

w_center_mid = w_center - 0.5 * s * (1 - q) / (1 + q) 
z_inits_mid, z_mask = _images_point_source(w_center_mid, nlenses=nlenses, **_params)
z_inits = z_inits_mid + 0.5 * s * (1 - q) / (1 + q)


yi         = 0
area_all   = 0.0
area_image = jnp.zeros(10)
max_iter   = int(1e+6)
indx       = jnp.zeros((max_iter * 2, 5), dtype=int) # index for checking the overlaps
Nindx      = jnp.zeros((max_iter * 2), dtype=int)     # Number of images at y_index
xmin       = jnp.zeros((max_iter * 2))
xmax       = jnp.zeros((max_iter * 2)) 
area_x     = jnp.zeros((max_iter * 2)) 
y          = jnp.zeros((max_iter * 2)) 
dys        = jnp.zeros((max_iter * 2))
CM2MD = -0.5 * s * (1 - q)/(1 + q) 
dz2 = jnp.inf
dx = incr 
count_x = 0.0
count_all = 0.0
rho2 = rho * rho
finish = jnp.bool_(False)

z_init = z_inits[z_mask][1]
dy     = incr
carry  = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
area, carry = image_area0(w_center, rho, z_init, dy, carry, **_params)
(yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
print(area, yi)
print(y)
print(dy)
print(area_x)

dy     = -incr
carry  = (yi, indx, Nindx, xmax, xmin, area_x, y, dys)
area, carry = image_area0(w_center, rho, z_init, dy, carry, **_params)
(yi, indx, Nindx, xmax, xmin, area_x, y, dys) = carry
print(area, yi)
print(y)
print(dy)
print(area_x)

N_limb = 5000
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, q=q, s=s)

fig = plt.figure()
ax = plt.axes()

mask_x = area_x>0
for i in range(len(xmin[mask_x])):
    plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i])
    #plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i], color=cmap(pos_neg[i]))
    plt.plot(xmin[mask_x][i], y[mask_x][i], ".", color="None", mec="k")
    plt.plot(xmax[mask_x][i], y[mask_x][i], ".", color="None", mec="k")

source = plt.Circle((w_center.real, w_center.imag), rho, color='b', fill=False)
ax.add_patch(source)
plt.scatter(image_limb[mask].ravel().real, image_limb[mask].ravel().imag, s=1,color="purple")
plt.scatter(cau_tri.ravel().real, cau_tri.ravel().imag,   marker=".", color="red", s=1)
plt.scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1)
for i in range(len(z_inits[z_mask])):
        plt.scatter(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, marker="*", zorder=2, ec="k")
        plt.text(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, s="%d"%(i), zorder=2)
plt.axis("equal")
plt.show()