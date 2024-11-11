import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
from microjax.inverse_ray.merge_area import calc_source_limb, merge_intervals 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, vmap

def _compute_in_mask(r_limb, th_limb, r_use, th_use):
    M = r_use.shape[0]  
    K = th_use.shape[0]  
    N = r_limb.shape[0]
    r_limb_expanded = r_limb.reshape(1, 1, N)
    th_limb_expanded = th_limb.reshape(1, 1, N)
    r_use_min = r_use[:, 0].reshape(M, 1, 1)
    r_use_max = r_use[:, 1].reshape(M, 1, 1)
    th_use_min = th_use[:, 0].reshape(1, K, 1)
    th_use_max = th_use[:, 1].reshape(1, K, 1)

    r_condition = (r_limb_expanded > r_use_min) & (r_limb_expanded < r_use_max)  # shape: (M, 1, N)
    th_condition = (th_limb_expanded > th_use_min) & (th_limb_expanded < th_use_max)  # shape: (1, K, N)
    combined_condition = r_condition & th_condition  # shape: (M, K, N)

    # condition for all the combination
    in_mask = jnp.any(combined_condition, axis=2)  # shape: (M, K)
    return in_mask


#@jit
def merge_intervals_theta_new(arr, offset=1.0, fac=10.0):
    diff = jnp.diff(arr)
    diff_neg = jnp.where(diff[:-1] > fac * diff[1:],  fac * diff[1:], diff[:-1])
    diff_pos = jnp.where(diff[1:]  > fac * diff[:-1], fac * diff[:-1], diff[1:])
    arr_start = arr[1:-1] - diff_neg - offset 
    arr_end   = arr[1:-1] + diff_pos  + offset
    #plt.plot(arr_start, ".")
    #plt.plot(arr_end, ".")
    #plt.plot(arr, ".", c="k")
    #plt.yscale("log")
    #plt.grid()
    #plt.show()
    intervals = jnp.stack([arr_start, arr_end], axis=1)
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]
    def merge_scan_fn(carry, next_interval):
        current_interval = carry
        overlap_exists = current_interval[1] >= next_interval[0]
        merged_interval = jnp.where(
            overlap_exists,
            jnp.array([current_interval[0], jnp.maximum(current_interval[1], next_interval[1])]),
            next_interval
        )
        return merged_interval, merged_interval

    _, merged_intervals = lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])

    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)
    merged_intervals = jnp.clip(merged_intervals, 0, 2*jnp.pi)

    return merged_intervals, mask

#@jit
def calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th):
    r_limb = jnp.abs(image_limb.ravel())
    r_is = jnp.where(mask_limb.ravel(), r_limb, -rho * offset_r)
    r_, r_mask = merge_intervals(r_is, offset=offset_r*rho) 
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2 * jnp.pi)
    th_is = jnp.sort(jnp.where(mask_limb.ravel(), th_limb.ravel(), 0.0))
    th_is = jnp.clip(th_is, 0, 2 * jnp.pi)
    offset_th = jnp.arctan2(offset_th * rho, jnp.max(jnp.max(r_, axis=1)*r_mask))
    th_, th_mask = merge_intervals_theta_new(th_is, offset=offset_th)
    return r_, r_mask, th_, th_mask

w_center = jnp.complex128(-0.0672 + 0.0j)
#w_center = jnp.complex128(0.209 + 0.126j)
q = 0.1
s = 1.0
rho = 1e-4
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"q": q, "s": s, "a": a, "e1": e1}

r_resolution = 500
th_resolution = 500
Nlimb = 500
offset_r = 10.0
offset_th  = 10.0 

shifted = 0.5 * s * (1 - q) / (1 + q)
w_center_shifted = w_center - shifted
image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)
r_use  = r_ * r_mask.astype(float)[:, None]
th_use = th_ * th_mask.astype(float)[:, None]
r_use  = r_use[jnp.argsort(r_use[:,1])][-5:] # mergeできていない場合は5とは限らない・・・
th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
r_limb = jnp.abs(image_limb)
th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
in_mask = _compute_in_mask(r_limb.ravel(), th_limb.ravel(), r_use, th_use)
r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
# binary-lens should have less than 5 images.
r_vmap   = r_masked[jnp.argsort(r_masked[:,1] == 0)][:5]
th_vmap  = th_masked[jnp.argsort(th_masked[:,1] == 0)][:5] 
#print("r_use: ", r_use)
#print("th_use: ", th_use)
#r_use  = r_[r_mask]
#th_use = th_[th_mask]
#th_resolution = int(resolution * GRID_RATIO)
r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

def plot(r_range, th_range):
    r_values = r_grid_norm * (r_range[1] - r_range[0]) + r_range[0]
    th_values = th_grid_norm * (th_range[1] - th_range[0]) + th_range[0]
    r_mesh, th_mesh = jnp.meshgrid(r_values, th_values, indexing='ij')
    z_grid = r_mesh * (jnp.cos(th_mesh) + 1j * jnp.sin(th_mesh))
    image_mesh = lens_eq(z_grid - shifted, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    in_source = (distances - rho < 0.0)
    return z_grid.real, z_grid.imag, in_source
vmap_plot = vmap(plot, in_axes=(0, 0))
x_grids, y_grids, in_sources = vmap_plot(r_vmap, th_vmap)
fig = plt.figure(figsize=(6,6))
ax = plt.axes()
for x_grid, y_grid, in_source in zip(x_grids, y_grids, in_sources):
    ax.scatter(x_grid.ravel(), y_grid.ravel(), c='lightgray', s=1)
    ax.scatter(x_grid[in_source].ravel(), y_grid[in_source].ravel(), c='orange', s=1, zorder=2)
from microjax.point_source import critical_and_caustic_curves
critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=1000, s=s, q=q) 
plt.scatter(critical_curves.ravel().real, critical_curves.ravel().imag, marker=".", color="green", s=3)
plt.scatter(caustic_curves.ravel().real, caustic_curves.ravel().imag, marker=".", color="crimson", s=3)
plt.scatter(image_limb[mask_limb].ravel().real, 
            image_limb[mask_limb].ravel().imag, 
            s=1,color="purple", zorder=2)
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
ax.scatter(w_limb.real, w_limb.imag, color="blue", s=1)
plt.plot(w_center.real, w_center.imag, "*", color="k")
plt.plot(-q * s, 0 , ".",c="k")
plt.plot((1.0 - q) * s, 0 ,".",c="k")
ax.set_aspect('equal')
plt.show()