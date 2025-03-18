import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source

def define_regions(image_limb, mask_limb, rho, bins_r, bins_th, margin_r=0.5, margin_th=0.1, nlenses=2):
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

    image_limb = image_limb.ravel()
    mask_limb = mask_limb.ravel()

    r_limb     = jnp.abs(image_limb * mask_limb)
    th_limb    = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi) * mask_limb
    r_mins, r_maxs   = cluster_1d(r_limb, bins=bins_r, max_cluster=nimages_init)
    th_mins, th_maxs = cluster_1d(th_limb, bins=bins_th, mode_r=False, max_cluster=nimages_init)

    r_map  = jnp.array([r_mins, r_maxs]).T
    th_map = jnp.array([th_mins, th_maxs]).T
    th_map = jnp.where(jnp.isclose(th_map, 0.0), 0.0, th_map)
    th_map = jnp.where(jnp.isclose(th_map, 2*jnp.pi), 2*jnp.pi, th_map)

    r_map_refine = _refine_addmargin_merge_r(r_limb, r_map, margin=margin_r*rho)

    in_mask   = _compute_in_mask(r_limb, th_limb, r_map_refine, th_map)
    r_masked  = jnp.repeat(r_map_refine, th_map.shape[0], axis=0) * in_mask.ravel()[:, None]
    th_masked = jnp.tile(th_map, (r_map_refine.shape[0], 1)) * in_mask.ravel()[:, None]

    r_excess   = r_masked[jnp.argsort(jnp.isclose(r_masked[:, 1], 0.0))][:nimages_init]
    th_excess  = th_masked[jnp.argsort(jnp.isclose(th_masked[:, 1], 0.0))][:nimages_init]
    r_scan, th_scan = _merge_final(r_excess, th_excess)
    r_scan, th_scan = r_scan[:nimage_real], th_scan[:nimage_real]
    # refine intervals
    r_scan_refine, th_scan_refine = _refine_final(r_limb, th_limb, r_scan, th_scan, margin_r=margin_r*rho, margin_th=jnp.deg2rad(margin_th))  
    #return r_scan, th_scan
    return r_scan_refine, th_scan_refine

import jax.numpy as jnp

def _refine_final(r_limb, th_limb, r_scan, th_scan, margin_r=0.0, margin_th=0.0):
    r_min, r_max = r_scan[:, 0], r_scan[:, 1] # (M,) shape
    th_min, th_max = th_scan[:, 0], th_scan[:, 1] # (M,) shape
    cond_r = (r_min[:, None] < r_limb) & (r_limb < r_max[:, None])  # (M, N) shape
    cond_flip = (-jnp.pi < th_min) & (th_min < 0.0) & (0.0 < th_max) & (th_max < jnp.pi) # (M,) shape
    cond_flip_ex = cond_flip[:, None]
    th_limb_new = jnp.where(cond_flip_ex, 
                            jnp.where(th_limb[None, :] > jnp.pi, th_limb[None, :] - 2*jnp.pi, th_limb[None, :]), 
                            th_limb[None, :]) # (M, N) shape
    
    cond_th = (th_min[:, None] < th_limb_new) & (th_limb_new < th_max[:, None]) & (th_limb_new != 0) & (r_limb != 0)  # (M, N)
    in_mask = cond_r & cond_th  # (M, N)
    # r refine
    r_min_ref  = jnp.min(jnp.where(in_mask, r_limb[None, :], 1e+10), axis=1) # (M,)
    r_max_ref  = jnp.max(jnp.where(in_mask, r_limb[None, :], 0.0), axis=1)
    r_min_ref  = jnp.maximum(0.0, r_min_ref - margin_r)
    r_max_ref  = jnp.where(r_max_ref != 0, r_max_ref + margin_r, 0.0)
    # theta refine
    th_min_ref = jnp.min(jnp.where(in_mask, th_limb_new, 1e+10), axis=1)
    th_max_ref = jnp.max(jnp.where(in_mask, th_limb_new, -1e+10), axis=1)
    # apply or not 
    case_org = (th_min > 0.0)|(cond_flip)
    r_min_new  = jnp.where(case_org, r_min_ref, r_min)
    r_max_new  = jnp.where(case_org, r_max_ref, r_max) 
    th_min_new = jnp.where(case_org, th_min_ref, th_min) - margin_th
    th_max_new = jnp.where(case_org, th_max_ref, th_max) + margin_th
    r_scan_refine = jnp.stack([r_min_new, r_max_new], axis=1)
    th_scan_refine = jnp.stack([th_min_new, th_max_new], axis=1)

    return r_scan_refine, th_scan_refine

def refine_final(r_limb, th_limb, r_scan, th_scan, margin_r=0.0, margin_th=0.0):
    M = r_scan.shape[0] # number of regions
    N = r_limb.shape[0] # number of image limb points
    arr=[]
    for r_inteval, th_interval in zip(r_scan, th_scan):
        r_min, r_max = r_inteval
        th_min, th_max = th_interval 
        case_0 = th_min > 0.0 # (N,) shape
        cond_r = (r_min < r_limb)&(r_limb < r_max) # (N,) shape
        cond_th = (th_min < th_limb)&(th_limb < th_max)&(th_limb!=0)
        in_mask = (cond_r)&(cond_th)
        th_min_   = jnp.min(jnp.where(in_mask, th_limb, 1e+10))  
        th_max_   = jnp.max(jnp.where(in_mask, th_limb, -1e+10))
        r_min_new = jnp.min(jnp.where(in_mask, r_limb, 1e+10))
        r_max_new = jnp.max(jnp.where(in_mask, r_limb, -1e+10))
        r_min_new  = jnp.maximum(0.0, r_min_new - margin_r)
        r_max_new  = r_max_new + margin_r 
        th_min_new = jnp.where(case_0, th_min_, th_min)
        th_max_new = jnp.where(case_0, th_max_, th_max)
        th_min_new = th_min_new - margin_th
        th_max_new = th_max_new + margin_th
        #arr.append([r_min, r_max, th_min_new, th_max_new])
        arr.append([r_min_new, r_max_new, th_min_new, th_max_new])
    arr = jnp.vstack(arr)
    r_scan_refine = arr[:, :2]
    th_scan_refine = arr[:, 2:]
    return r_scan_refine, th_scan_refine
    """
    for r_inteval, th_interval in zip(r_scan, th_scan):
        r_min, r_max = r_inteval
        th_min, th_max = th_interval
        case_0 = th_min > 0.0 # (N,) shape
        cond_r = (r_min < r_limb)&(r_limb < r_max) # (N,) shape
        th_max_1 = (0 < th_max)&(th_max < 0.5*jnp.pi)
        th_max_2 = (0.5*jnp.pi < th_max)&(th_max < jnp.pi)
        th_max_3 = (jnp.pi < th_max)&(th_max < 1.5*jnp.pi)
        th_max_4 = (1.5*jnp.pi < th_max)&(th_max < 2*jnp.pi)
        th_min_1 = (-2*jnp.pi < th_min)&(th_min < -1.5*jnp.pi)
        th_min_2 = (-1.5*jnp.pi < th_min)&(th_min < -jnp.pi)
        th_min_3 = (-jnp.pi < th_min)&(th_min < -0.5*jnp.pi)
        th_min_4 = (-0.5*jnp.pi < th_min)&(th_min < 0.0)

        case_1 = ((th_max_1)&(th_min_4))|((th_max_1)&(th_min_3))|((th_max_2)&(th_min_4))|((th_max_2)&(th_min_3))
        case_1 = (~case_0)&(case_1)
        case_2 = (th_max_1)&(th_min_2)
        case_2 = (~case_0)&(case_2)
        case_3 = (th_max_3)&(th_min_4)
        case_3 = (~case_0)&(case_3)
        th_limb_cor = jnp.where((case_1)&(th_limb > jnp.pi), th_limb - 2*jnp.pi, th_limb)
        th_limb_cor = jnp.where((case_2)&(th_limb > 0.5*jnp.pi), th_limb_cor - 2*jnp.pi, th_limb_cor)
        th_limb_cor = jnp.where((case_3)&(1.5*jnp.pi < th_limb)&(th_limb < 2*jnp.pi), th_limb_cor - 2*jnp.pi, th_limb_cor)

        in_mask = (cond_r)&(th_min < th_limb_cor)&(th_limb_cor < th_max)&(th_limb_cor!=0)
        r_min_new = jnp.min(jnp.where(in_mask, r_limb, jnp.inf))
        r_max_new = jnp.max(jnp.where(in_mask, r_limb, -jnp.inf))
        th_min_new = jnp.min(jnp.where(in_mask, th_limb_cor, jnp.inf))
        th_max_new = jnp.max(jnp.where(in_mask, th_limb_cor, -jnp.inf)) 
        # add margin
        r_min_new  = jnp.maximum(0.0, r_min_new - margin_r)
        r_max_new  = r_max_new + margin_r 
        th_min_new = th_min_new - margin_th
        th_max_new = th_max_new + margin_th
        arr.append([r_min_new, r_max_new, th_min_new, th_max_new])"
    arr = jnp.array(arr)
    r_scan_refine = arr[:, :2]
    th_scan_refine = arr[:, 2:]
    return r_scan_refine, th_scan_refine
    """

def cluster_1d(arr, bins=100, max_cluster=5, mode_r=True):
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

def _merge_final(r_map, th_map):
    r_mins, r_maxs   = r_map[:, 0],  r_map[:, 1]  # (N,)
    th_mins, th_maxs = th_map[:, 0], th_map[:, 1]  # (N,)
    matrix = jnp.stack([r_mins, r_maxs, th_mins, th_maxs], axis=1)  # (N, 4)

    r_min_col = matrix[:, 0:1]   # (N, 1)
    r_max_col = matrix[:, 1:2]   # (N, 1)
    th_min_col = matrix[:, 2:3]  # (N, 1)
    th_max_col = matrix[:, 3:4]  # (N, 1)
    
    same_r = (r_min_col == r_mins) & (r_max_col == r_maxs) & (~jnp.all(matrix[:, None, :] == matrix, axis=2))  # (N, N)
    cond_merge = (th_min_col == 0) & (same_r) & (th_maxs == 2 * jnp.pi) # (N, N)
    th_min_adjusted = jnp.where(cond_merge, th_mins - 2 * jnp.pi, 0.0)  # (N, N)
    th_min_new = jnp.sum(th_min_adjusted, axis=1)  # (N,)
    th_min_new = jnp.where(th_min_new == 0, th_mins, th_min_new) # (N,)

    new_matrix = jnp.stack([r_mins, r_maxs, th_min_new, th_maxs], axis=1)  # (N, 4)
    cond_vanish = jnp.any(jnp.any(same_r, axis=1)&(th_mins == 0)) & (th_maxs == 2*jnp.pi)  # (N, ) shape
    new_matrix = jnp.where(cond_vanish[:, None], 0.0, new_matrix)  # (N, 4) shape
    
    sort_order = jnp.argsort(new_matrix[:, 1] == 0)
    new_matrix = new_matrix[sort_order]
    new_r_map = new_matrix[:, :2]
    new_th_map = new_matrix[:, 2:]
    return new_r_map, new_th_map

def _refine_addmargin_merge_r(r_limb, r_ranges, margin=0.0):
    # refine range
    M = r_ranges.shape[0]  
    N = r_limb.shape[0]
    r_limb_expanded = r_limb.reshape(1, N)
    r_mins  = r_ranges[:, 0].reshape(M, 1)
    r_maxs  = r_ranges[:, 1].reshape(M, 1)
    cond_in = (r_limb_expanded > r_mins) & (r_limb_expanded < r_maxs)  # shape: (M, N)
    valid_mins = jnp.where(cond_in, r_limb_expanded, jnp.inf)
    valid_maxs = jnp.where(cond_in, r_limb_expanded, -jnp.inf)
    r_mins = jnp.where(jnp.any(cond_in, axis=1), jnp.min(valid_mins, axis=1), 0.0)
    r_maxs = jnp.where(jnp.any(cond_in, axis=1), jnp.max(valid_maxs, axis=1), 0.0)
    # add margin
    r_mins = jnp.maximum(0.0, r_mins - margin)
    r_maxs = jnp.where(r_maxs!=0, r_maxs + margin, 0.0)
    # merge
    rows = jnp.array([r_mins, r_maxs]).T # (N, 2) shape
    N = rows.shape[0]
    rows_new = rows.copy()
    for i in range(N-1):
        row_0 = rows_new[i]
        row_1 = rows_new[i+1]
        update = row_0[1] > row_1[0]
        def true_fn(rows):
            rows = rows.at[i].set(jnp.array([0.0, 0.0]))
            rows = rows.at[i + 1].set(jnp.array([row_0[0], row_1[1]]))
            return rows
        def false_fn(rows):
            return rows
        rows_new = lax.cond(update, true_fn, false_fn, rows_new)
    return rows_new