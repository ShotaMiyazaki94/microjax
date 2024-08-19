import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.image_area0 import image_area0

NBIN = 20
nlenses = 2

w_center = jnp.complex128(-0.00 - 0.0j)
q  = 0.5
s  = 1.0
rho = 0.1
#rho = 1e-3

# extreme case
#NBIN = 5
#w_center = jnp.complex128(-0.0 - 0.0j)
#  = 1e-3
#  = 1.0
#rho = 1e-3

a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}

incr  = jnp.abs(rho / NBIN)
incr2 = incr * 0.5

w_center_mid = w_center - 0.5 * s * (1 - q) / (1 + q) 
z_inits_mid, z_mask = _images_point_source(w_center_mid, nlenses=nlenses, **_params)
z_inits = z_inits_mid + 0.5 * s * (1 - q) / (1 + q)


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
    print('%d images positive'%i)
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
    ##print(area)
if(0):
    plt.figure()
    for i in range(len(z_inits[z_mask])):
        plt.scatter(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, marker="*", zorder=2, ec="k")
        plt.text(z_inits[z_mask][i].real, z_inits[z_mask][i].imag, s="%d"%(i), zorder=2)
    plt.plot(xmin[area_x>0], y[area_x>0], ".", color="k", label="xmin")
    plt.plot(xmax[area_x>0], y[area_x>0], ".", color="red", label="xmax")
    plt.legend()
    plt.axis("equal")
    plt.show()
    plt.close()

print("identify the protruding areas that are missed!!")
(yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
xmin_diff = jnp.where(jnp.diff(xmin)==0, jnp.inf, jnp.diff(xmin))
xmax_diff = jnp.where(jnp.diff(xmax)==0,-jnp.inf, jnp.diff(xmax)) 
y_diff    = jnp.where(jnp.diff(y)==0, jnp.inf, jnp.diff(y))
fac_marg = 2.0
upper_left  = (xmin_diff < -fac_marg * incr) & (dys[:-1] < 0) & (jnp.abs(y_diff) <= 2.0 * incr) 
lower_left  = (xmin_diff < -fac_marg * incr) & (dys[:-1] > 0) & (jnp.abs(y_diff) <= 2.0 * incr) 
upper_right = (xmax_diff > fac_marg * incr)  & (dys[:-1] < 0) & (jnp.abs(y_diff) <= 2.0 * incr) 
lower_right = (xmax_diff > fac_marg * incr)  & (dys[:-1] > 0) & (jnp.abs(y_diff) <= 2.0 * incr) 

fac = 0.0
for k in jnp.where(upper_left)[0]:
    offset_factor = fac * jnp.abs((xmin[k + 2] - xmin[k + 1]) / incr).astype(int)
    z_init = jnp.complex128(xmin[k + 1] + offset_factor * incr + 1j * (y[k + 1] + incr))
    yi += 1
    carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    area, carry = image_area0(w_center, rho, z_init, incr, carry, **_params) 
    (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
    area_all += area
    yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)
    print("upper left (yi=%d)"%(k), "%.2f"%y[k], area!=0)

for k in jnp.where(upper_right)[0]:
    #k + 1 is 伸びてる部分のyi
    offset_factor = fac * jnp.abs((xmax[k + 2] - xmax[k + 1]) / incr).astype(int)
    z_init = jnp.complex128(xmax[k + 1] - offset_factor * incr + 1j * (y[k + 1] + incr))
    yi += 1
    carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    area, carry = image_area0(w_center, rho, z_init, incr, carry, **_params) 
    (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
    area_all += area
    yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)
    print("upper right (yi=%d)"%(k), "%.2f"%y[k], area!=0)

for k in jnp.where(lower_left)[0]:
    offset_factor = fac * jnp.abs((xmin[k + 2] - xmin[k + 1]) / incr).astype(int)
    z_init = jnp.complex128(xmin[k + 1] + offset_factor * incr + 1j * (y[k + 1] - incr))
    yi += 1
    carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    area, carry = image_area0(w_center, rho, z_init, -incr, carry, **_params) 
    (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
    area_all += area
    yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)
    print("lower left (yi=%d)"%(k), "%.2f"%z_init.real, "%.2f"%z_init.imag, area!=0)
        
for k in jnp.where(lower_right)[0]:
    offset_factor = fac * jnp.abs((xmax[k + 2] - xmax[k + 1]) / incr).astype(int)
    z_init = jnp.complex128(xmax[k + 1] - offset_factor * incr + 1j * (y[k + 1] - incr))
    yi += 1
    carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    area, carry = image_area0(w_center, rho, z_init, -incr, carry, **_params) 
    (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
    area_all += area
    yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)
    print("lower right (yi=%d)"%(k), "%.2f"%z_init.real, "%.2f"%z_init.imag, area!=0)


N_limb = 5000
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, q=q, s=s)

fig = plt.figure(figsize=(8,8))
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