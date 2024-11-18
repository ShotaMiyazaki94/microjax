import jax 
import jax.numpy as jnp
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
#from microjax.inverse_ray.merge_area import calc_source_limb, calculate_overlap_and_range 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, vmap, lax

def merge_intervals_r(arr, offset=1.0, margin_fac=100.0):
    arr = jnp.sort(arr)
    diff = jnp.diff(arr)
    diff_neg = jnp.where(diff[:-1] > margin_fac * diff[1:],  margin_fac * diff[1:], diff[:-1])
    diff_pos = jnp.where(diff[1:]  > margin_fac * diff[:-1], margin_fac * diff[:-1], diff[1:])
    arr_start = arr[1:-1] - diff_neg - offset 
    arr_end   = arr[1:-1] + diff_pos  + offset
    intervals = jnp.stack([jnp.maximum(arr_start, 0.0), arr_end], axis=1) 
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

    # Initial merge
    _, merged_intervals = lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)
    return merged_intervals, mask
    # Apply the range factor to expand the merged intervals
    """
    center = (merged_intervals[:, 0] + merged_intervals[:, 1]) / 2
    half_width = (merged_intervals[:, 1] - merged_intervals[:, 0]) / 2 * expand_fac
    expanded_intervals = jnp.stack([center - half_width, center + half_width], axis=1)
    sorted_expanded_intervals = expanded_intervals[jnp.argsort(expanded_intervals[:, 0])]
    # Final merge
    _, final_merged_intervals = lax.scan(merge_scan_fn, sorted_expanded_intervals[0], sorted_expanded_intervals[1:])
    final_merged_intervals = jnp.vstack([sorted_expanded_intervals[0], final_merged_intervals])
    final_mask = jnp.append(jnp.diff(final_merged_intervals[:, 0]) != 0, True)

    return final_merged_intervals, final_mask
    """
def merge_intervals_theta(arr, offset=1.0, fac=100.0):
    arr = jnp.sort(arr)
    diff = jnp.diff(arr)
    diff_neg = jnp.where(diff[:-1] > fac * diff[1:],  fac * diff[1:], diff[:-1])
    diff_pos = jnp.where(diff[1:]  > fac * diff[:-1], fac * diff[:-1], diff[1:])
    arr_start = arr[1:-1] - diff_neg - offset 
    arr_end   = arr[1:-1] + diff_pos  + offset
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


#@partial(jit, static_argnums=(2,))
def calc_source_limb(w_center, rho, N_limb=100, **_params):
    s, q = _params["s"], _params["q"]
    a = 0.5 * s
    e1 = q / (1.0 + q)
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
    w_limb_shift = w_limb - 0.5 * s * (1 - q) / (1 + q)
    image, mask = _images_point_source(w_limb_shift, a=a, e1=e1)
    image_limb = image + 0.5 * s * (1 - q) / (1 + q)
    return image_limb, mask

def calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th):
    r_is   = jnp.ravel(jnp.abs(image_limb*mask_limb))
    r_, r_mask = merge_intervals_r(r_is, offset=offset_r*rho) 
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2 * jnp.pi)
    th_is = jnp.sort(jnp.where(mask_limb.ravel(), th_limb.ravel(), 0.0))
    th_is = jnp.clip(th_is, 0, 2 * jnp.pi)
    offset_th = jnp.arctan2(offset_th * rho, jnp.max(jnp.max(r_, axis=1)*r_mask))
    th_, th_mask = merge_intervals_theta(th_is, offset=offset_th)
    return r_, r_mask, th_, th_mask


def _compute_in_mask(r_limb, theta_limb, r_intervals, theta_intervals):
    """
    Computes a boolean mask indicating whether any limb points fall within specified radial and angular intervals.

    Parameters:
    -----------
    r_limb : array_like
        1D array of radial positions of limb points.
    theta_limb : array_like
        1D array of angular positions (theta) of limb points.
    r_intervals : array_like
        2D array of shape (M, 2), where each row defines an [r_min, r_max] interval.
    theta_intervals : array_like
        2D array of shape (K, 2), where each row defines a [theta_min, theta_max] interval.

    Returns:
    --------
    in_mask : array_like
        2D boolean array of shape (M, K). Each element (i, j) is True if any limb point falls within
        both r_intervals[i] and theta_intervals[j], and False otherwise.
    """
    num_r_intervals = r_intervals.shape[0]       # Number of radial intervals (M)
    num_theta_intervals = theta_intervals.shape[0]  # Number of angular intervals (K)
    num_limb_points = r_limb.shape[0]            # Number of limb points (N)
    # Reshape arrays to enable broadcasting over intervals and limb points
    r_limb_expanded = r_limb.reshape(1, 1, num_limb_points)
    theta_limb_expanded = theta_limb.reshape(1, 1, num_limb_points)
    r_min = r_intervals[:, 0].reshape(num_r_intervals, 1, 1)
    r_max = r_intervals[:, 1].reshape(num_r_intervals, 1, 1)
    theta_min = theta_intervals[:, 0].reshape(1, num_theta_intervals, 1)
    theta_max = theta_intervals[:, 1].reshape(1, num_theta_intervals, 1)
    # Create boolean conditions for limb points within each radial and angular interval
    r_condition     = (r_min < r_limb_expanded) & (r_limb_expanded < r_max)      # Shape: (M, 1, N)
    theta_condition = (theta_min < theta_limb_expanded) & (theta_limb_expanded < theta_max)  # Shape: (1, K, N)
    combined_condition = r_condition & theta_condition  # Shape: (M, K, N)
    # Determine if any limb point satisfies both conditions for each interval pair
    in_mask = jnp.any(combined_condition, axis=2)  # Shape: (M, K)
    return in_mask

#w_center = jnp.complex128(-0.06688372+0.00092252j)
#rho = 0.001
w_center = jnp.complex128(-0.25439312+3.00666326e-01j)
rho = 0.0001
q = 0.1
s = 1.0
a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"q": q, "s": s, "a": a, "e1": e1}

r_resolution  = 500
th_resolution = 500
Nlimb = 500
offset_r = 1.0
offset_th  = 1.0 

shifted = 0.5 * s * (1 - q) / (1 + q)
w_center_shifted = w_center - shifted
image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)
r_use  = r_ * r_mask.astype(float)[:, None]
th_use = th_ * th_mask.astype(float)[:, None]
r_use  = r_use[jnp.argsort(r_use[:,1])][-10:] # mergeできていない場合は5とは限らない・・・
th_use = th_use[jnp.argsort(th_use[:,1])][-10:]
r_limb = jnp.abs(image_limb)
th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
in_mask = _compute_in_mask(r_limb.ravel()*mask_limb.ravel(), th_limb.ravel()*mask_limb.ravel(), r_use, th_use)
r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]

# binary-lens should have less than 5 images.
r_vmap   = r_masked[jnp.argsort(r_masked[:,1] == 0)][0:10]
th_vmap  = th_masked[jnp.argsort(th_masked[:,1] == 0)][0:10] 
r_grid_norm = jnp.linspace(0, 1, r_resolution, endpoint=False)
th_grid_norm = jnp.linspace(0, 1, th_resolution, endpoint=False)

print("r_use:",r_use)
print("th_use:", th_use)
#print("r_use_expand:",r_use_expanded)
#print("th_use_expand:", theta_use_expanded)
print("-------------")
for r ,th in zip(r_vmap, th_vmap):
    print(r, th)
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
critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
plt.scatter(critical_curves.ravel().real, 
            critical_curves.ravel().imag, marker=".", color="green", s=3)
plt.scatter(caustic_curves.ravel().real, 
            caustic_curves.ravel().imag, marker=".", color="crimson", s=3)
plt.scatter(image_limb[mask_limb].ravel().real, 
            image_limb[mask_limb].ravel().imag, 
            s=1,color="purple", zorder=2)
plt.scatter((image_limb[~mask_limb].real).ravel(), 
            (image_limb[~mask_limb].imag).ravel(), 
            s=1,color="blue", zorder=1)
#plt.scatter((image_limb.real*mask_limb).ravel(), 
#            (image_limb.imag*mask_limb).ravel(), 
#            s=1,color="blue", zorder=2)
w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
ax.scatter(w_limb.real, w_limb.imag, color="blue", s=1)
plt.plot(w_center.real, w_center.imag, "*", color="k")
plt.plot(-q * s, 0 , ".",c="k")
plt.plot((1.0 - q) * s, 0 ,".",c="k")
ax.set_aspect('equal')
plt.show()