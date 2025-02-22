import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source

def grid_intervals(image_limb, mask_limb, rho, nlenses=2, bins=100, max_cluster=5):
    image_limb = image_limb.ravel()
    mask_limb = mask_limb.ravel()
    r     = jnp.abs(image_limb * mask_limb)
    theta = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi) * mask_limb
    # determine 1D regions by binning method
    r_mins, r_maxs   = cluster_1d(r, bins=bins, max_cluster=max_cluster)
    th_mins, th_maxs = cluster_1d(theta, bins, mode_r=False, max_cluster=2*max_cluster)
    r_map = jnp.array([r_mins, r_maxs]).T
    th_map = jnp.array([th_mins, th_maxs]).T
    th_map = merge_theta(th_map)[-max_cluster:]
    # select combinations that contain the image limb
    r_scan, th_scan = select_intervals(image_limb, mask_limb, r_map, th_map, max_regions=max_cluster)
    # refine intervals?

def select_intervals(image_limb, mask_limb, r_map, th_map, max_regions=5):
    r_limb = jnp.abs(image_limb.ravel() * mask_limb.ravel())
    M = r_map.shape[0]
    K = th_map.shape[0]
    N = r_limb.shape[0]
    
    r_limb_expanded = r_limb.reshape(1, 1, N)
    r_map_min = r_map[:, 0].reshape(M, 1, 1)
    r_map_max = r_map[:, 1].reshape(M, 1, 1)
    r_condition = (r_map_min < r_limb_expanded) & (r_limb_expanded < r_map_max)  # shape: (M, 1, N)

    th_map_min = th_map[:, 0].reshape(1, K, 1)
    th_map_max = th_map[:, 1].reshape(1, K, 1)
    th_limb_pipi = jnp.arctan2(image_limb.imag, image_limb.real)
    th_limb_2pi = jnp.mod(th_limb_pipi, 2*jnp.pi)
    th_limb_pipi_ex = jnp.where(mask_limb.ravel(), th_limb_pipi, jnp.inf)
    th_limb_2pi_ex  = jnp.where(mask_limb.ravel(), th_limb_2pi, jnp.inf)
    th_cond_pipi = (th_map_min < th_limb_pipi_ex) & (th_limb_pipi_ex < th_map_max) # shape: (1, K, N)    
    th_cond_2pi  = (th_map_min < th_limb_2pi_ex) & (th_limb_2pi_ex < th_map_max)   # shape: (1, K, N)
    th_condition = (th_cond_pipi)|(th_cond_2pi) # shape: (1, K, N)

    combined_condition = r_condition & th_condition  # shape: (M, K, N)
    in_mask = jnp.any(combined_condition, axis=2)  # shape: (M, K)
    r_repeat = jnp.repeat(r_map, K, axis=0) * in_mask.ravel()[:, None]
    th_tiled = jnp.tile(th_map, (M, 1)) * in_mask.ravel()[:, None]
    r_scan  = r_repeat[jnp.argsort(r_repeat[:,1] == 0)][:max_regions]
    th_scan = th_tiled[jnp.argsort(th_tiled[:,1] == 0)][:max_regions]
    return r_scan, th_scan

def merge_theta(arr):
    # We might be cared for that bins should not be less than 100 because of these conditions.
    # Especially, arr[:, 1] == 2*jnp.pi may cause bugs.  
    start_zero = (arr[:, 0] == 0)&(arr[:, 1] != 0) 
    end_twopi  = (arr[:, 0] != 0)&(jnp.isclose(arr[:, 1], 2*jnp.pi, atol=1e-6))
    #end_twopi  = (arr[:, 0] != 0)&(arr[:, 1] == 2*jnp.pi) 
    start_pick = jnp.min(jnp.where(end_twopi,  arr[:, 0] - 2*jnp.pi, 0.0))
    end_pick   = jnp.max(jnp.where(start_zero, arr[:, 1], 0.0))
    merge = jnp.where(end_twopi[:, None], jnp.array([0.0, 0.0]), arr)
    merge = jnp.where(start_zero[:, None], jnp.array([start_pick, end_pick]), merge)
    return merge[merge[:, 1].argsort()]

@partial(jit, static_argnames=["bins", "max_cluster", "mode_r"])
def cluster_1d(arr, bins=100, max_cluster=5, mode_r=True):
    # This might cause the gradient error.
    if mode_r:
        bin_min = jnp.min(jnp.where(arr == 0, jnp.inf, arr))
        bin_max = jnp.max(jnp.where(arr == 0, -jnp.inf, arr))
    else:
        # Here I do not want to optimize the intervals so much.
        bin_min = 0
        bin_max = 2 * jnp.pi
    delta = (bin_max - bin_min) / bins
    bin_edges = jnp.linspace(bin_min - delta, bin_max + delta, bins + 3, endpoint=True) # [0]-[bin+1]
    bin_indices = jnp.digitize(arr, bin_edges) - 1  # [0]-[bin+1]
    bin_indices = jnp.clip(bin_indices, 1, bins)    # [1]-[bin]
    counts = jnp.bincount(bin_indices, length=bins + 2)
    bin_mask = counts > 0 # bins + 2 
    diff_mask = jnp.diff(bin_mask.astype(int))  #length: bin + 1
    start_mask = diff_mask == 1 
    end_mask   = diff_mask == -1

    start_edges = jnp.sort(bin_edges[1:-1] * start_mask.astype(float), descending=False)[-max_cluster:]
    end_edges   = jnp.sort(bin_edges[1:-1] * end_mask.astype(float), descending=False)[-max_cluster:] 
    return start_edges, end_edges


def determine_grid_regions(image_limb, mask_limb, rho, offset_r, offset_th, nlenses=2):
    """
    determine the regions to be grid-spaced for the inverse-ray integration.
    """
    if nlenses == 2:
        nimages_init = 10
        nimage_real = 5
    elif nlenses == 3:
        nimages_init = 18
        nimage_real = 9
    else:
        raise ValueError("Only 2 or 3 lenses are supported.")
    # one-dimensional overlap search and merging.
    r_, r_mask, th_, th_mask = calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th)
    r_use  = r_ * r_mask.astype(float)[:, None]
    th_use = th_ * th_mask.astype(float)[:, None]
    # if merging is correct, 5 may be emperically sufficient for binary-lens and 9 is for triple-lens
    r_use  = r_use[jnp.argsort(r_use[:, 1])][-nimages_init:]
    th_use = th_use[jnp.argsort(th_use[:, 1])][-nimages_init:]
    r_limb = jnp.abs(image_limb)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2 * jnp.pi)
    # select matched regions including image limbs. binary-lens microlensing should have less than 5 images.
    in_mask = _compute_in_mask(r_limb.ravel() * mask_limb.ravel(), th_limb.ravel() * mask_limb.ravel(), r_use, th_use)
    r_masked  = jnp.repeat(r_use, r_use.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_use, (r_use.shape[0], 1)) * in_mask.ravel()[:, None]
    # select the first regions (5 for binary, 9 for triple) for the integration.
    r_excess   = r_masked[jnp.argsort(r_masked[:, 1] == 0)][:nimages_init]
    th_excess  = th_masked[jnp.argsort(th_masked[:, 1] == 0)][:nimages_init]
    r_scan, th_scan = merge_final(r_excess, th_excess)
    return r_scan[:nimage_real], th_scan[:nimage_real]


def merge_final(r_vmap, th_vmap):
    """
    Merge continuous regions of search space.
    The continuous regions but separated are due to the definition of the angle [0 ~ 2pi].
    This checks the next and next-next elements.
    The regions merged and not mearged are within [-pi, pi] and [0, 2pi], respectively.
    """
    r_next1 = jnp.roll(r_vmap, -1, axis=0)
    th_next1 = jnp.roll(th_vmap, -1, axis=0)
    r_next2 = jnp.roll(r_vmap, -2, axis=0)
    th_next2 = jnp.roll(th_vmap, -2, axis=0)
    
    # Adjust the shape of the condition arrays to match (10, 2)
    same_r1 = jnp.all(r_vmap == r_next1, axis=1) 
    same_r2 = jnp.all(r_vmap == r_next2, axis=1)
    continuous_th1 = (th_vmap[:, 0] == 0) & (th_next1[:, 1] == 2 * jnp.pi)
    continuous_th2 = (th_vmap[:, 0] == 0) & (th_next2[:, 1] == 2 * jnp.pi)

    # Broadcast conditions to match the shape of r_vmap and th_vmap
    merge1 = same_r1 & continuous_th1
    merge2 = same_r2 & continuous_th2

    # Apply conditions
    merged_r = jnp.where(merge1[:, None], r_next1, r_vmap)
    merged_th = jnp.where(
        merge1[:, None], 
        jnp.stack([th_next1[:, 0] - 2 * jnp.pi, th_vmap[:, 1]], axis=-1), 
        th_vmap
    )
    merged_r = jnp.where(merge2[:, None], r_next2, merged_r)
    merged_th = jnp.where(
        merge2[:, None], 
        jnp.stack([th_next2[:, 0] - 2 * jnp.pi, merged_th[:, 1]], axis=-1), 
        merged_th
    )
    # Zero out the merged regions
    zero_out_1 = jnp.roll(merge1, 1)
    zero_out_2 = jnp.roll(merge2, 2)
    zero_out_mask = zero_out_1 | zero_out_2
    merged_r = jnp.where(zero_out_mask[:, None], 0.0, merged_r)
    merged_th = jnp.where(zero_out_mask[:, None], 0.0, merged_th)

    sort_order = jnp.argsort(merged_r[:, 1] == 0)
    return merged_r[sort_order], merged_th[sort_order]


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

#@partial(jit, static_argnames=("offset", "margin_fac"))
def merge_intervals_r(arr, offset=1.0, margin_fac=100.0):
    arr = jnp.sort(arr)
    diff = jnp.diff(arr)
    diff_neg = jnp.minimum(diff[:-1], margin_fac * diff[1:])
    diff_pos = jnp.minimum(diff[1:], margin_fac * diff[:-1])
    arr_start = arr[1:-1] - diff_neg - offset 
    arr_end   = arr[1:-1] + diff_pos + offset
    intervals = jnp.stack([arr_start, arr_end], axis=1)
    sorted_intervals = intervals[jnp.argsort(intervals[:, 0])]

    def merge_scan_fn(carry, next_interval):
        overlap_exists = carry[1] >= next_interval[0]
        merged_interval = jnp.where(
            overlap_exists,
            jnp.array([carry[0], jnp.maximum(carry[1], next_interval[1])]),
            next_interval
        )
        return merged_interval, merged_interval

    _, merged_intervals = lax.scan(merge_scan_fn, sorted_intervals[0], sorted_intervals[1:])
    merged_intervals = jnp.vstack([sorted_intervals[0], merged_intervals])

    mask = jnp.append(jnp.diff(merged_intervals[:, 0]) != 0, True)
    #merged_intervals = jnp.clip(merged_intervals, 0, jnp.max(arr) + offset)
    return merged_intervals, mask

def _merge_intervals_r(arr, offset=1.0, margin_fac=100.0):
    arr = jnp.sort(arr)
    diff = jnp.diff(arr)
    diff_neg = jnp.where(diff[:-1] > margin_fac * diff[1:],  margin_fac * diff[1:], diff[:-1])
    diff_pos = jnp.where(diff[1:]  > margin_fac * diff[:-1], margin_fac * diff[:-1], diff[1:])
    arr_start = arr[1:-1] - diff_neg - offset 
    arr_end   = arr[1:-1] + diff_pos + offset
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

#@partial(jit, static_argnames=("offset", "fac"))
def merge_intervals_theta(arr, offset=1.0, fac=100.0):
    diff = jnp.diff(jnp.sort(arr))
    diff_neg = jnp.minimum(diff[:-1], fac * diff[1:])
    diff_pos = jnp.minimum(diff[1:],  fac * diff[:-1])
    arr_start = arr[1:-1] - diff_neg - offset 
    arr_end   = arr[1:-1] + diff_pos + offset
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


@partial(jit, static_argnames=("Nlimb", "nlenses"))
def calc_source_limb(w_center, rho, Nlimb=100, nlenses=2, **_params):
    s, q = _params["s"], _params["q"]
    a = 0.5 * s
    e1 = q / (1.0 + q)
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
    w_limb_shift = w_limb - 0.5 * s * (1 - q) / (1 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    image, mask = _images_point_source(w_limb_shift, nlenses=nlenses, **_params)
    image_limb = image + 0.5 * s * (1 - q) / (1 + q)
    return image_limb, mask

@partial(jit, static_argnames=("offset_r", "offset_th"))
def calculate_overlap_and_range(image_limb, mask_limb, rho, offset_r, offset_th):
    r_limb = jnp.abs(image_limb.ravel())
    r_is = jnp.where(mask_limb.ravel(), r_limb, 0.0)
    r_, r_mask = merge_intervals_r(r_is, offset=offset_r*rho)
    th_limb = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2 * jnp.pi)
    th_is = jnp.sort(jnp.where(mask_limb.ravel(), th_limb.ravel(), 0.0))
    th_is = jnp.clip(th_is, 0, 2 * jnp.pi)
    offset_th = jnp.deg2rad(offset_th)
    th_, th_mask = merge_intervals_theta(th_is, offset=offset_th) 
    return r_, r_mask, th_, th_mask
