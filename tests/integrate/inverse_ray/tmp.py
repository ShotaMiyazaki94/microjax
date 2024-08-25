import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.image_area0 import image_area0

NBIN = 10
nlenses = 2

#w_center = jnp.complex128(-0.0 - 0.0j)
w_center = jnp.complex128(-0.10 - 0.15j)
q  = 0.5
s  = 1.0
rho = 1e-4

a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}

incr  = jnp.abs(rho / NBIN)
incr2 = incr * 0.5

w_center_mid = w_center - 0.5 * s * (1 - q) / (1 + q) 
z_inits_mid, z_mask = _images_point_source(w_center_mid, nlenses=nlenses, **_params)
z_inits = z_inits_mid + 0.5 * s * (1 - q) / (1 + q)

N_limb = 10
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate

z_inits = jnp.append(z_inits.ravel(),image_limb.ravel())
z_mask = jnp.append(z_mask.ravel(), mask.ravel())

x_inits = jnp.int_(z_inits.real / incr) * incr
y_inits = jnp.int_(z_inits.imag / incr) * incr
z_inits = x_inits + 1j * y_inits 

_, dub_indices = jnp.unique(z_inits.real, return_index=True)
dub_mask = jnp.zeros(z_inits.shape, dtype=bool)
dub_mask = dub_mask.at[dub_indices].set(True)

print(jnp.sum(z_mask))
z_mask = z_mask & dub_mask
print(jnp.sum(z_mask))

#z_inits = image_limb.ravel()

yi         = 0
area_all   = 0.0
area_image = jnp.zeros(10)
max_iter   = int(10 / incr)
indx       = jnp.zeros((max_iter * 2, 10), dtype=jnp.int32) # index for checking the overlaps
Nindx      = jnp.zeros((max_iter * 2), dtype=jnp.int32)     # Number of images at y_index
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
carry  = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)

for i in jnp.arange(len(z_inits[z_mask])):
    print('%d images positive %.5f %.5f '%(i, z_inits[z_mask][i].real, z_inits[z_mask][i].imag))
    z_init = z_inits[z_mask][i]
    dy     = incr
    #carry  = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    area, carry = image_area0(w_center, rho, z_init, dy, carry, **_params)
    #(yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
    #break
    #print(area)
    print('%d images negative'%i)
    dy     = -incr
    z_init = z_inits[z_mask][i] + 1j * dy
    #carry  = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    area, carry = image_area0(w_center, rho, z_init, dy, carry, **_params)
    #(yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry

(yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry

fig = plt.figure()
ax = plt.axes()

mask_x = area_x>0
cmap = plt.get_cmap("coolwarm")
pos_neg = jnp.where(dys[mask_x] > 0, 1.0, 0.0)
#pos_neg = jnp.where(dys[mask_x] > 0, 1.0, 0.0)
for i in range(len(xmin[mask_x])):
    #plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i])
    plt.hlines(y[mask_x][i],xmin[mask_x][i],xmax[mask_x][i], color=cmap(pos_neg[i]))
    plt.plot(xmin[mask_x][i], y[mask_x][i], ".", color="None", mec="k")
    plt.plot(xmax[mask_x][i], y[mask_x][i], ".", color="None", mec="k")

plt.plot(w_limb.ravel().real, w_limb.ravel().imag, "o", color="None",mec="k")

N_limb = 5000
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, q=q, s=s)

source = plt.Circle((w_center.real, w_center.imag), rho, color='b', fill=False)
ax.add_patch(source)
plt.plot(w_center.real, w_center.imag, "*", color="k")

plt.plot(-q * s, 0 , ".",c="k")
plt.plot((1.0 - q) * s, 0 ,".",c="k")

plt.scatter(image_limb[mask].ravel().real, image_limb[mask].ravel().imag, s=1,color="purple")
plt.scatter(cau_tri.ravel().real, cau_tri.ravel().imag,   marker=".", color="red", s=1)
plt.scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1)
for i in range(len(z_inits[z_mask])):
        plt.scatter(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, marker="*", zorder=2, ec="k")
        plt.text(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, s="%d"%(i), zorder=2)
plt.axis("equal")
plt.show()