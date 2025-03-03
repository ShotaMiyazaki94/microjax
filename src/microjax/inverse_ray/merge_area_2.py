import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source

def grid_intervals(image_limb, mask_limb, rho, bins=100, max_cluster=5, optimize=False, margin_r=0.5, margin_th=0.1):
    """
    margin_r:   margin in r direction. unit is rho.
    margin_th:  margin in theta direction, unit is deg.
    """
    image_limb = image_limb.ravel()
    mask_limb = mask_limb.ravel()
    # theta is defined within the (0, 2*jnp.pi) range
    r     = jnp.abs(image_limb * mask_limb)
    theta = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi) * mask_limb
    # determine 1D regions by binning method
    r_mins, r_maxs   = cluster_1d(r, bins=bins, max_cluster=max_cluster)
    th_mins, th_maxs = cluster_1d(theta, bins, mode_r=False, max_cluster=2*max_cluster)
    r_map = jnp.array([r_mins, r_maxs]).T
    th_map = jnp.array([th_mins, th_maxs]).T
    th_map = merge_theta(th_map)[-max_cluster:]
    # select combinations that contain the image limb
    r_scan, th_scan = select_intervals(image_limb, mask_limb, r_map, th_map, 
                                       max_regions=max_cluster, optimize=optimize, 
                                       margin_r=margin_r*rho, margin_th=jnp.deg2rad(margin_th))
    return r_scan, th_scan

@partial(jit, static_argnames=["max_regions", "optimize", "margin_r", "margin_th"])
def select_intervals(image_limb, mask_limb, r_map, th_map, max_regions=5, optimize=False, margin_r=0.0, margin_th=0.0):
    # assume that image_limb and mask_limb are 1 dimention arrays.
    r_limb = jnp.abs(image_limb * mask_limb)
    M, K, N = r_map.shape[0], th_map.shape[0], r_limb.shape[0]
    
    r_limb_expanded = r_limb.reshape(1, 1, N)
    r_map_min, r_map_max  = r_map[:, 0].reshape(M, 1, 1), r_map[:, 1].reshape(M, 1, 1)
    r_condition = (r_map_min < r_limb_expanded) & (r_limb_expanded < r_map_max)  # shape: (M, 1, N)

    th_map_min, th_map_max = th_map[:, 0].reshape(1, K, 1), th_map[:, 1].reshape(1, K, 1)
    th_limb_pipi = jnp.arctan2(image_limb.imag, image_limb.real) # shape: [N]
    th_limb_2pi  = jnp.mod(th_limb_pipi, 2*jnp.pi)               # shape: [N]
    th_limb_pipi_expanded = jnp.where(mask_limb.ravel(), th_limb_pipi, jnp.inf)
    th_limb_2pi_expanded  = jnp.where(mask_limb.ravel(), th_limb_2pi, jnp.inf)
    th_cond_pipi = (th_map_min < th_limb_pipi_expanded) & (th_limb_pipi_expanded < th_map_max) # shape: (1, K, N)    
    th_cond_2pi  = (th_map_min < th_limb_2pi_expanded) & (th_limb_2pi_expanded < th_map_max)   # shape: (1, K, N)
    th_condition = (th_cond_pipi)|(th_cond_2pi) # shape: (1, K, N)

    combined_condition = r_condition & th_condition  # shape: (M, K, N)
    in_mask = jnp.any(combined_condition, axis=2)  # shape: (M, K)

    r_repeat = jnp.repeat(r_map, K, axis=0) * in_mask.ravel()[:, None]  # shape: (M * K, 2)
    th_tiled = jnp.tile(th_map, (M, 1)) * in_mask.ravel()[:, None]      # shape: (M * K, 2) 
    r_scan  = r_repeat[jnp.argsort(r_repeat[:, 1] == 0)][:max_regions]  # shape: (max_regions, 2)
    th_scan = th_tiled[jnp.argsort(th_tiled[:, 1] == 0)][:max_regions]  # shape: (max_regions, 2)

    if optimize:
        def refine_intervals(r_range, th_range, limb, mask, margin_r=margin_r, margin_th=margin_th):
            # r_range and th_range: [2] shape
            # limb and mask: 1-D shape
            r_values = jnp.abs(limb)
            th_values = jnp.arctan2(limb.imag, limb.real)
            twopi_mode = th_range[0] * th_range[1] > 0
            th_values = jnp.where(twopi_mode, jnp.mod(th_values, 2 * jnp.pi), th_values)
            r_in = (r_range[0] < r_values) & (r_values < r_range[1])
            th_in = (th_range[0] < th_values) & (th_values < th_range[1])
            mask_in = r_in & th_in
            valid_mask = mask & mask_in
            r_values_masked = jnp.where(valid_mask, r_values, jnp.nan)
            th_values_masked = jnp.where(valid_mask, th_values, jnp.nan)
            r_min  = jnp.nanmin(r_values_masked) - margin_r
            r_min  = jnp.maximum(0, r_min) 
            r_max  = jnp.nanmax(r_values_masked) + margin_r
            th_min = jnp.nanmin(th_values_masked) - margin_th
            th_max = jnp.nanmax(th_values_masked) + margin_th
            r_min  = jnp.nan_to_num(r_min, nan=0.0) 
            r_max  = jnp.nan_to_num(r_max, nan=0.0) 
            th_min = jnp.nan_to_num(th_min, nan=0.0) 
            th_max = jnp.nan_to_num(th_max, nan=0.0) 
            return jnp.array([r_min, r_max]), jnp.array([th_min, th_max])
        r_scan, th_scan = vmap(refine_intervals, in_axes=(0, 0, None, None))(r_scan, th_scan, image_limb, mask_limb)
    # margin
    #r_scan  = jnp.array([r_scan[:, 0] - margin_r, r_scan[:, 1] + margin_r]).T
    #th_scan = jnp.array([th_scan[:, 0] - margin_th, th_scan[:, 1] + margin_th]).T
    return r_scan, th_scan

def merge_theta(arr):
    start_zero = (arr[:, 0] == 0)&(arr[:, 1] != 0) 
    end_twopi  = (arr[:, 0] != 0)&(jnp.isclose(arr[:, 1], 2*jnp.pi, atol=1e-6))&jnp.any(start_zero)
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