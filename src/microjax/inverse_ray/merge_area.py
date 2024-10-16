import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source

@jit
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

    _, merged_intervals = lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)

    return merged_intervals, mask

@jit
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

    _, merged_intervals = lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])
    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)

    return merged_intervals, mask

@partial(jit, static_argnums=(2,))
def calc_source_limb(w_center, rho, N_limb=100, **_params):
    s, q = _params["s"], _params["q"]
    a = 0.5 * s
    e1 = q / (1.0 + q)
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, N_limb)), dtype=complex)
    w_limb_shift = w_limb - 0.5 * s * (1 - q) / (1 + q)
    image, mask = _images_point_source(w_limb_shift, a=a, e1=e1)
    image_limb = image + 0.5 * s * (1 - q) / (1 + q)
    return image_limb, mask

@jit
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

if __name__ == "__main__":
    import jax
    from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    w_center = jnp.array([0.0 + 0.0j], dtype=complex)
    q, s = 0.1, 1.0
    rho  = 1e-4
    Nlimb = 1000
    offset_r = 1.0
    offset_th = 5.0

    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q) 
    w_center_shifted = w_center - shifted


    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)  
    
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    r_use  = r_use[jnp.argsort(r_use[:,1])][-5:]
    th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
    print(r_use)
    print(th_use)
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)