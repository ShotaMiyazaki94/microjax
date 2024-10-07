
import jax 
from jax import jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from functools import partial
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

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

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, ))
def mag_simple(w_center, rho, resolution=150, Nlimb=100, offset_r = 1.0, offset_th = 5.0, GRID_RATIO=5, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1} 
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)  
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    # 10 is maximum number of images for triple-lens 
    r_use  = r_use[jnp.argsort(r_use[:,1])][-10:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-10:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
    
    def compute_for_range(r_range, th_range):
        in_ = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) &
                      (th_limb > th_range[0]) & (th_limb < th_range[1]))
        def compute_if_in():
            r_grid = jnp.linspace(r_range[0], r_range[1], resolution, endpoint=False)
            th_grid = jnp.linspace(th_range[0], th_range[1], resolution * GRID_RATIO, endpoint=False)
            r_mesh, th_mesh = jnp.meshgrid(r_grid, th_grid)
            r_mesh, th_mesh = r_mesh.T, th_mesh.T
            x_mesh = r_mesh.ravel() * jnp.cos(th_mesh.ravel())
            y_mesh = r_mesh.ravel() * jnp.sin(th_mesh.ravel())
            z_mesh = x_mesh + 1j * y_mesh
            image_mesh = lens_eq(z_mesh - 0.5 * s * (1 - q) / (1 + q), **_params)
            image_mask = jnp.abs(image_mesh - w_center + 0.5 * s * (1 - q) / (1 + q)) < rho
            dr = jnp.median(jnp.diff(r_grid))
            dth = jnp.median(jnp.diff(th_grid))
            return jnp.sum(r_mesh.ravel() * image_mask.astype(float) * dr * dth)

        def compute_if_not_in():
            return 0.0

        return jnp.where(in_, compute_if_in(), compute_if_not_in())
    
    compute_vmap = vmap(vmap(compute_for_range, in_axes=(None, 0)), in_axes=(0, None))
    image_areas = compute_vmap(r_use, th_use)
    magnification = jnp.sum(image_areas) / rho**2 / jnp.pi 
    
    return magnification

if __name__ == "__main__":
    resolution = 150
    Nlimb = 100
    offset_r = 1.0
    offset_th = 10.0
    GRID_RATIO = 5
    
    w_center = jnp.complex128(-0.1 + 0.00j)
    q = 0.2
    s = 1
    rho = 1e-4 
    _params = {"q": q, "s": s}
    magnification = mag_simple(w_center, rho, resolution=resolution, Nlimb=Nlimb, 
                        offset_r=offset_r, offset_th=offset_th, GRID_RATIO = GRID_RATIO, **_params)
    print(magnification)

"""
# Now measure execution time without JIT compilation overhead
# First block: Source limb image
start = time.time()
image_limb, mask_limb = calc_source_limb(w_center, rho, N_limb, **_params)
end = time.time()
print("time (solve image_limbs): %.2f second"%(end - start))

# Second block: Overlap and range calculation
start = time.time()
r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th) 
end = time.time()
print("time (merge 1d-regions): %.2f second"%(end - start))


r_idx = jnp.arange(r_.shape[0])
th_idx = jnp.arange(th_.shape[0])
r_idx_grid, th_idx_grid = jnp.meshgrid(r_idx, th_idx, indexing='ij')
r_use  = r_[r_mask]
th_use = th_[th_mask] 

# grid construction
start = time.time()
z_meshes = []
z_masks = []
areas = []
r_limb = jnp.abs(image_limb)
th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)
r_new  = jnp.where(r_mask[:, None], r_, 0)
th_new = jnp.where(th_mask[:, None], th_, 0)

for r_range in r_new:
    if (jnp.sum(r_range)==0):
        continue
    for th_range in th_new:
        if (jnp.sum(th_range)==0):
            continue
        in_  = jnp.any((r_limb > r_range[0]) & (r_limb < r_range[1]) & (th_limb > th_range[0]) & (th_limb < th_range[1])) & ~(jnp.sum(r_range)==0) & ~(jnp.sum(th_range)==0) 
        if (1):
            print("%.2f %.2f %.2f %.2f "%(r_range[0], r_range[1], th_range[0], th_range[1])) 
            r_grid  = jnp.linspace(r_range[0], r_range[1], resolution, endpoint=False)
            th_grid = jnp.linspace(th_range[0], th_range[1], GRID_RATIO*resolution, endpoint=False)
            r_mesh, th_mesh = jnp.meshgrid(r_grid, th_grid)
            r_mesh, th_mesh = r_mesh.T, th_mesh.T
            x_mesh = r_mesh.ravel() * jnp.cos(th_mesh.ravel())
            y_mesh = r_mesh.ravel() * jnp.sin(th_mesh.ravel())
            z_mesh = x_mesh.ravel() + 1j * y_mesh.ravel()
            image_mesh = lens_eq(z_mesh - 0.5*s*(1 - q)/(1 + q), **_params) 
            image_mask = jnp.abs(image_mesh - w_center + 0.5*s*(1 - q)/(1 + q)) < rho
            dr  = jnp.mean(jnp.diff(r_grid))
            dth = jnp.mean(jnp.diff(th_grid))
            image_area = jnp.sum(r_mesh.ravel() * image_mask.astype(float) * dr * dth)
            areas.append(image_area)
            z_meshes.append(z_mesh)
            z_masks.append(image_mask)
end = time.time()
print("time (grid & mask): %.2f second"%(end - start))
z_meshes = jnp.array(z_meshes)
z_masks = jnp.array(z_masks)
areas = jnp.array(areas)
magnification = jnp.sum(areas / (rho**2 * jnp.pi))
print("magnification:%.3f"%magnification)  

w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)

fig = plt.figure(figsize=(6,6))
ax = plt.axes()
plt.scatter(w_limb.real, w_limb.imag, color="b", s=1)
plt.plot(w_center.real, w_center.imag, "*", color="k")
plt.plot(-q/(1+q) * s, 0 , "x",c="k")
plt.plot(1.0/(1+q) * s, 0 ,"x",c="k")
plt.scatter(image_limb[mask_limb].ravel().real, image_limb[mask_limb].ravel().imag, s=1,color="purple")
plt.scatter(cau_tri.ravel().real, cau_tri.ravel().imag, marker=".", color="red", s=1)
plt.scatter(crit_tri.ravel().real, crit_tri.ravel().imag, marker=".", color="green", s=1)
plt.axis("equal")
plt.grid(ls=":")

plt.scatter(z_meshes[z_masks].real, z_meshes[z_masks].imag, s=2, marker="o", zorder=-1, color="None", ec="blue", alpha=0.5)
plt.scatter(z_meshes[~z_masks].real, z_meshes[~z_masks].imag, s=1, marker=".", zorder=-2, color="gray", alpha=0.3)
plt.show()
"""