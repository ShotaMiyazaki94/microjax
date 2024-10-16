import jax 
from jax import jit, vmap, lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from functools import partial
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import numpy as np

# Define functions
@jax.jit
def merge_intervals(arr, offset=1.0):
    intervals = jnp.stack([jnp.maximum(arr - offset, 0), arr + offset], axis=1)
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

@jax.jit
def merge_intervals_circ(arr, offset=1.0):
    arr_start  = jnp.clip(arr - offset, 0, 2*jnp.pi)
    arr_end    = jnp.clip(arr + offset, 0, 2*jnp.pi)
    intervals = jnp.stack([arr_start, arr_end], axis=1)
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

@partial(jax.jit, static_argnums=(2,))
def calc_source_limb(w_center, rho, N_limb=100, **_params):
    s, q = _params["s"], _params["q"]
    a = 0.5 * s
    e1 = q / (1.0 + q)
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
    w_limb_shift = w_limb - 0.5 * s * (1 - q) / (1 + q)
    image, mask = _images_point_source(w_limb_shift, a=a, e1=e1)
    image_limb = image + 0.5 * s * (1 - q) / (1 + q)
    return image_limb, mask

@jax.jit
def calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th):
    r_limb = jnp.abs(image_limb.ravel())
    r_is = jnp.where(mask_limb.ravel(), r_limb, -rho * offset_r)
    r_, r_mask = merge_intervals(r_is, offset=offset_r*rho) 
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi).ravel()
    th_is = jnp.where(mask_limb.ravel(), th_limb.ravel(), 0.0)
    th_is = jnp.where(th_is < 0.0, th_is + 2*jnp.pi, th_is)
    offset_th = jnp.arctan2(offset_th * rho, jnp.max(jnp.max(r_, axis=1)*r_mask))
    th_, th_mask = merge_intervals_circ(th_is, offset=offset_th)

    return r_, r_mask, th_, th_mask


resolution = 200
Nlimb = 1000
offset_r = 5.0
offset_th = 10.0
GRID_RATIO = 5

w_center = jnp.complex128(-0.122 + 0.00j)
q = 0.2
s = 1
rho = 0.01
a  = 0.5 * s
e1 = q / (1.0 + q)
_params = {"q": q, "s": s, "a": a, "e1": e1}
shifted = 0.5 * s * (1 - q) / (1 + q) 
w_center_shifted = w_center - shifted

image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)  
r_use  = r_ * r_mask.astype(float)[:, None]
th_use = th_ * th_mask.astype(float)[:, None]
# 10 is maximum number of images for triple-lens 
r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
r_limb = jnp.abs(image_limb)
th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)

num_elements = resolution * GRID_RATIO

z_meshes_acc = jnp.zeros((num_elements,), dtype=jnp.complex128)
z_masks_acc  = jnp.zeros((num_elements,), dtype=bool)
r_grid_normalized = jnp.linspace(0, 1, resolution, endpoint=False)
th_grid_normalized = jnp.linspace(0, 1, resolution * GRID_RATIO, endpoint=False)
r_mesh_norm, th_mesh_norm = jnp.meshgrid(r_grid_normalized, th_grid_normalized, indexing='ij') 

def compute_for_range(r_range, th_range, z_meshes_acc, z_masks_acc):
    in_mask = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) & 
                        (th_limb > th_range[0]) & (th_limb < th_range[1]))
    def compute_if_in():
        dr = (r_range[1] - r_range[0]) / resolution
        dth = (th_range[1] - th_range[0]) / (resolution * GRID_RATIO)
        r_mesh = r_mesh_norm * (r_range[1] - r_range[0]) + r_range[0]
        th_mesh = th_mesh_norm * (th_range[1] - th_range[0]) + th_range[0]
        z_mesh = jnp.ravel(r_mesh * (jnp.cos(th_mesh) + 1j * jnp.sin(th_mesh)))
        image_mesh = lens_eq(z_mesh - shifted, **_params)
        distances  = jnp.abs(image_mesh - w_center_shifted) 
        image_mask = distances < rho
        z_meshes_acc = z_meshes_acc.at[:z_mesh.size].set(z_mesh)  # 配列にデータを追加
        z_masks_acc = z_masks_acc.at[:image_mask.size].set(image_mask)  # 配列にデータを追加
        return jnp.sum(r_mesh.ravel() * image_mask.astype(float) * dr * dth), z_meshes_acc, z_masks_acc

    def compute_if_not_in():
        return 0.0, z_meshes_acc, z_masks_acc

    result, z_meshes_acc, z_masks_acc = jax.lax.cond(in_mask, compute_if_in, compute_if_not_in)
    return result, z_meshes_acc, z_masks_acc

compute_vmap = vmap(vmap(lambda r_range, th_range: compute_for_range(r_range, th_range, z_meshes_acc, z_masks_acc), 
                         in_axes=(None, 0)), 
                    in_axes=(0, None))

# image_areas と z_meshes_acc, z_masks_acc を取得
image_areas, z_meshes_acc, z_masks_acc = compute_vmap(r_use, th_use)


print(z_masks_acc)
