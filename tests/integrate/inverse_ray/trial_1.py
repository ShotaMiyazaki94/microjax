import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.inverse_ray.image_area0 import image_area0

@jax.jit
def merge_intervals(arr, offset=1.0):
    intervals = jnp.stack([arr - offset, arr + offset], axis=1)
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]

    def merge_scan_fn(carry, next_interval):
        current_interval = carry

        start_max = jnp.maximum(current_interval[0], next_interval[0])
        start_min = jnp.minimum(current_interval[0], next_interval[0])
        end_max = jnp.maximum(current_interval[1], next_interval[1])
        end_min = jnp.minimum(current_interval[1], next_interval[1])

        overlap_exists = start_max <= end_min
        updated_current_interval = jnp.where(
            overlap_exists,
            jnp.array([start_min, end_max]),
            next_interval
        )

        return updated_current_interval, updated_current_interval
    
    _, merged_intervals = jax.lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)

    return merged_intervals, mask

def merge_intervals_circ(arr, offset=1.0):
    intervals = jnp.stack([th_is - offset, th_is + offset], axis=1)
    intervals = jnp.clip(intervals, 0, 2*jnp.pi)
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]

    def merge_scan_fn(carry, next_interval):
        current_interval = carry

        start_max = jnp.maximum(current_interval[0], next_interval[0])
        start_min = jnp.minimum(current_interval[0], next_interval[0])
        end_max = jnp.maximum(current_interval[1], next_interval[1])
        end_min = jnp.minimum(current_interval[1], next_interval[1])

        overlap_exists = start_max <= end_min
        updated_current_interval = jnp.where(
            overlap_exists,
            jnp.array([start_min, end_max]),
            next_interval
        )

        return updated_current_interval, updated_current_interval

    _, merged_intervals = jax.lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)

    return merged_intervals, mask

w_center = jnp.complex128(-0.04 - 0.1j)
q  = 0.5
s  = 1.0
rho = 1.0
a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}

NBIN = 10
offset = 0.5
GridRatio = 5.0

N_limb = 1000
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
crit_tri, cau_tri = critical_and_caustic_curves(npts=1000, q=q, s=s)

# construct r-range!
dr  = rho / NBIN / GridRatio
dth = rho / NBIN 

image_start = image_limb[mask].ravel()
r_is = jnp.sqrt(image_start.real**2 + image_start.imag**2)
r_is = jnp.sort(r_is)
r_, r_mask = merge_intervals(r_is, offset=offset*rho)
r_ = jnp.concatenate([jnp.arange(x1, x2, dr) for x1, x2 in r_[r_mask]])
# construct theta-range!
th_is = jnp.mod(jnp.arctan2(image_start.imag, image_start.real), 2*jnp.pi) # normalize
th_is = jnp.sort(th_is)
margin_th = jnp.arctan2(offset*rho, jnp.min(r_is)) 
intervals = jnp.stack([th_is - offset, th_is + offset], axis=1)
intervals = jnp.clip(intervals, 0, 2*jnp.pi)
print(intervals)


margin_th = jnp.arctan2(offset*rho, jnp.min(r_is)) 
th_, th_mask = merge_intervals_circ(th_is, offset=margin_th)
th_ = jnp.concatenate([jnp.arange(jnp.where(x2==2*jnp.pi, -dth, x1), 
                                  jnp.where(x2==2*jnp.pi, x1-2*jnp.pi, x2), 
                                  jnp.where(x2==2*jnp.pi, -dth, dth)) for x1, x2 in th_[th_mask]])
#print(th_)
th_ = jnp.sort(jnp.mod(th_, 2*jnp.pi))

th_grid, r_grid = jnp.meshgrid(th_, r_)
#th_grid, r_grid = jnp.meshgrid(th_, r_) 
x_grid = r_grid.ravel()*jnp.cos(th_grid.ravel())
y_grid = r_grid.ravel()*jnp.sin(th_grid.ravel())
z_mesh = x_grid + 1j * y_grid

import time
z_mesh.block_until_ready()
start = time.time()
image_mesh = lens_eq(z_mesh - 0.5*s*(1 - q)/(1 + q), **_params) 
image_mask = jnp.abs(image_mesh - w_center + 0.5*s*(1 - q)/(1 + q)) < rho
image_mask.block_until_ready()
end = time.time()
print("time:",end - start)
image_area  = jnp.sum(r_grid.T.ravel() * image_mask.astype(float) * dr * dth) 
source_area = rho**2 * jnp.pi 
#print(image_area, source_area)
magnification = image_area / source_area
print("magnification:",magnification)

fig = plt.figure(figsize=(5,5))
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
plt.scatter(z_mesh[image_mask].real, z_mesh[image_mask].imag, s=1, marker=".", zorder=-1)
plt.scatter(z_mesh.real, z_mesh.imag, s=1, marker=".", zorder=-2, color="gray", alpha=0.3)
#cond = (jnp.diff(image_mask.astype(int))==1) 
#edge_mask = jnp.where(cond)
#edge_mask_ = jnp.array(edge_mask[0]+1)
#plt.plot(z_mesh[edge_mask].real, z_mesh[edge_mask].imag, "o", color="None", mec="r")
#plt.plot(z_mesh[edge_mask_].real, z_mesh[edge_mask].imag, "o", color="None", mec="g")
#plt.grid(ls="--")
#plt.tight_layout()
plt.show()