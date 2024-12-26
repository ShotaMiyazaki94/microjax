import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source

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

@partial(jit, static_argnames=("offset", "margin_fac"))
def merge_intervals_r(arr, offset=1.0, margin_fac=100.0):
    arr = jnp.sort(arr)
    diff = jnp.diff(arr)
    diff_neg = jnp.where(diff[:-1] > margin_fac * diff[1:],  margin_fac * diff[1:], diff[:-1])
    diff_pos = jnp.where(diff[1:]  > margin_fac * diff[:-1], margin_fac * diff[:-1], diff[1:])
    arr_start = arr[1:-1] - diff_neg - offset 
    arr_end   = arr[1:-1] + diff_pos  + offset
    intervals = jnp.stack([jnp.maximum(arr_start, 0.0), arr_end], axis=1) 
    #intervals = jnp.stack([jnp.maximum(arr - offset, 0), arr + offset], axis=1)
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

@partial(jit, static_argnames=("offset", "fac"))
def merge_intervals_theta(arr, offset=1.0, fac=100.0):
    diff = jnp.diff(jnp.sort(arr))
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


@partial(jit, static_argnames=("Nlimb"))
def calc_source_limb(w_center, rho, Nlimb=100, **_params):
    s, q = _params["s"], _params["q"]
    a = 0.5 * s
    e1 = q / (1.0 + q)
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
    w_limb_shift = w_limb - 0.5 * s * (1 - q) / (1 + q)
    image, mask = _images_point_source(w_limb_shift, a=a, e1=e1)
    image_limb = image + 0.5 * s * (1 - q) / (1 + q)
    return image_limb, mask

def calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th):
    r_limb = jnp.abs(image_limb.ravel())
    r_is = jnp.where(mask_limb.ravel(), r_limb, 0.0)
    r_, r_mask = merge_intervals_r(r_is, offset=offset_r*rho) 
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2 * jnp.pi)
    th_is = jnp.sort(jnp.where(mask_limb.ravel(), th_limb.ravel(), 0.0))
    th_is = jnp.clip(th_is, 0, 2 * jnp.pi)
    offset_th = jnp.arctan2(offset_th * rho, jnp.max(jnp.max(r_, axis=1)*r_mask))
    th_, th_mask = merge_intervals_theta(th_is, offset=offset_th)
    return r_, r_mask, th_, th_mask


if __name__ == "__main__":

    "speed test!!!"

    import jax
    import jax.numpy as jnp
    import numpy as np
    from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    import time, timeit

    w_center = jnp.array([0.1 + 0.1j], dtype=complex)
    q, s = 0.1, 1.0
    rho  = 0.1
    Nlimb = 1000
    offset_r = 1.0
    offset_th = 5.0

    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = 0.5 * s * (1 - q) / (1 + q) 
    w_center_shifted = w_center - shifted

    # Warm up JIT-compiled functions
    start = time.time()
    calc_source_limb(w_center, rho, Nlimb, **_params)
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)
    end = time.time()
    print("JIT time  : %.3f sec"%(end - start))

    # Define wrapper functions for timing
    def time_calc_source_limb(w_center):
        calc_source_limb(w_center, rho, Nlimb, **_params)

    def time_calculate_overlap_and_range(w_center):
        image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
        calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)

    def time_calc(w_center):
        image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
        r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)
        r_use = r_ * r_mask.astype(float)[:, None]
        th_use = th_ * th_mask.astype(float)[:, None]
        r_use = r_use[jnp.argsort(r_use[:,1])][-5:]
        th_use = th_use[jnp.argsort(th_use[:,1])][-5:]
        r_limb = jnp.abs(image_limb)
        th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi)

    # Run time measurement with statistics for different w_center
    n_repeats = 10
    time_limb = timeit.repeat("time_calc_source_limb(w_center)", globals=globals(), repeat=n_repeats, number=1)
    print("image_limb: %.4f sec ± %.4f sec (n=%d)" % (np.mean(time_limb), np.std(time_limb), n_repeats))
    time_overlap = timeit.repeat("time_calculate_overlap_and_range(w_center)", globals=globals(), repeat=n_repeats, number=1)
    print("merge_1d  : %.4f sec ± %.4f sec (n=%d)" % (np.mean(time_overlap), np.std(time_overlap), n_repeats))
    time_eliminate = timeit.repeat("time_calc(w_center)", globals=globals(), repeat=n_repeats, number=1)
    print("eliminate : %.4f sec ± %.4f sec (n=%d)" % (np.mean(time_eliminate), np.std(time_eliminate), n_repeats))