import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from microjax.point_source import _images_point_source
from microjax.inverse_ray.merge_area import calc_source_limb, determine_grid_regions
from microjax.point_source import critical_and_caustic_curves
import time
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
from functools import partial

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

def merge_theta(arr):
    """
    merge 1-D theta ranges separated due to the (0, 2pi) definition
    """
    start_zero = (arr[:, 0] == 0)&(arr[:, 1] != 0) 
    end_twopi  = (arr[:, 0] != 0)&(arr[:, 1] == 2*jnp.pi) 
    start_pick = jnp.min(jnp.where(end_twopi, arr[:,0] - 2*jnp.pi, 0.0))
    end_pick   = jnp.max(jnp.where(start_zero, arr[:,1], 0.0))
    merge = jnp.where(end_twopi[:, None], 0.0, arr) # 0 padding 
    merge = jnp.where(start_zero[:, None], jnp.array([start_pick, end_pick]), merge)
    return merge[merge[:, 1].argsort()]

@partial(jit, static_argnames=["bins", "max_cluster", "mode_r"])
def cluster_1d(arr, bins=100, max_cluster=5, mode_r=True):
    # This might cause the gradient error.
    if mode_r:
        bin_min = jnp.min(jnp.where(arr == 0, np.inf, arr))
        bin_max = jnp.max(jnp.where(arr == 0, -np.inf, arr))
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

if(0):
    w_center = jnp.complex128(0.1 - 0.0j)
    #w_center = jnp.complex128(-0.0065 - 0.0j)
    s = 1.0
    q = 0.01
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"a": a, "e1": e1, "q": q, "s": s}
    rho = 5e-2
    Nlimb = 500
    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    image_limb = image_limb.ravel()
    mask_limb = mask_limb.ravel()
    r     = jnp.abs(image_limb * mask_limb)
    theta = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi) * mask_limb
    
    bins=100
    r_mins, r_maxs = cluster_1d(r, bins=bins, max_cluster=5)
    th_mins, th_maxs = cluster_1d(theta, bins, mode_r=False, max_cluster=10)
    print("r map")
    r_map = jnp.array([r_mins, r_maxs]).T
    print(r_map)
    th_map = jnp.array([th_mins, th_maxs]).T
    print(th_map)
    #r_scan, th_scan = merge_final(r_map, th_map)
    #r_scan, th_scan = r_scan[:5], th_scan[:5] 
    #print(r_scan, ".", th_scan)

    r_non = r[r!=0]
    bin_min, bin_max = r_non.min(), r_non.max()
    delta = (bin_max - bin_min) / bins
    bin_edges = jnp.linspace(bin_min - delta, bin_max + delta, bins + 3, endpoint=True)


    #plt.hist(np.array(r), bins=bins)
    plt.hist(np.array(r_non), bins=bins)
    ymin, ymax = plt.ylim()
    plt.vlines(r_mins, ymin=0, ymax=ymax, color="orange", zorder=2)
    plt.vlines(r_maxs, ymin=0, ymax=ymax, color="red", zorder=1)
    plt.vlines(bin_edges, ymin=0, ymax=0.1*ymax, color="gray", zorder=1, alpha=0.5)
    plt.xlim(0.9*r_non.min(), 1.1*r_non.max())
    plt.yscale("log")
    plt.savefig("test_r.pdf")
    plt.close()

    bin_min, bin_max = 0, 2*jnp.pi
    delta = (bin_max - bin_min) / bins
    bin_edges = jnp.linspace(bin_min - delta, bin_max + delta, bins + 3, endpoint=True)
    plt.hist(np.array(theta), bins=jnp.linspace(0, 2*jnp.pi, bins + 1))
    ymin, ymax = plt.ylim()
    plt.vlines(th_mins, ymin=0, ymax=ymax, color="orange", zorder=2)
    plt.vlines(th_maxs, ymin=0, ymax=ymax, color="red", zorder=1)
    plt.vlines(bin_edges, ymin=0, ymax=0.1*ymax, color="gray", zorder=1, alpha=0.5)
    plt.yscale("log")
    plt.savefig("test_th.pdf")
    plt.close()



    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
    plt.scatter(w_limb.real, w_limb.imag, color="blue", s=1, label="source limb")
    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
    plt.scatter(critical_curves.ravel().real, critical_curves.ravel().imag, 
                marker=".", color="green", s=3, label="critical curve")
    plt.scatter(caustic_curves.ravel().real, caustic_curves.ravel().imag, 
                marker=".", color="crimson", s=3, label="caustic")
    plt.plot(-q/(1+q) * s, 0 , "o",c="k")
    plt.plot((1.0)/(1+q) * s, 0 ,"o",c="k")
    plt.scatter(image_limb[mask_limb].real, image_limb[mask_limb].imag, s=1, color="purple")
    plt.axis("equal")
    plt.savefig("test_geo.pdf")
    plt.close()


if(0):
    #w_center = jnp.complex128(-0.05 - 0.0j)
    #w_center = jnp.complex128(-0.0065 - 0.0j)
    w_center = jnp.complex128(-0.045 - 0.0j)
    #w_center = jnp.complex128(-0.14 - 0.1j)
    s = 1.0
    q = 0.01
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"a": a, "e1": e1, "q": q, "s": s}
    rho = 1e-3

    Nlimb = 100

    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, 0.01, 1)
    for r, th in zip(r_scan, th_scan):
        print(r, th)

    image_limb = np.ravel(image_limb)
    mask_limb  = np.ravel(mask_limb)

    cluster_1d(np.abs(image_limb[mask_limb]), bins=20)
    a = np.histogram(np.abs(image_limb[mask_limb]), bins=100)
    #print(a)
    if(1):
        th = np.arctan2(image_limb.imag, image_limb.real)
        plt.hist(np.abs(image_limb[mask_limb]), bins=100)
        #plt.hist(th[mask_limb], bins=100)
        plt.show()
    #r = np.abs(image_limb)
    #theta = np.mod(np.arctan2(image_limb.imag, image_limb.real), 2*np.pi)
    if(1):
        w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
        plt.scatter(w_limb.real, w_limb.imag, color="blue", s=1, label="source limb")
        w_limb2 = w_center + jnp.array(0.5*rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
        image_limb2, mask_limb2 = calc_source_limb(w_center, 0.5*rho, Nlimb, **_params)
        plt.scatter(w_limb2.real, w_limb2.imag, color="lightblue", s=1, label="source limb")

        plt.scatter(critical_curves.ravel().real, 
                    critical_curves.ravel().imag, marker=".", color="green", s=3, label="critical curve")
        plt.scatter(caustic_curves.ravel().real, 
                    caustic_curves.ravel().imag, marker=".", color="crimson", s=3, label="caustic")
        plt.plot(-q/(1+q) * s, 0 , "o",c="k")
        plt.plot((1.0)/(1+q) * s, 0 ,"o",c="k")
        plt.scatter(image_limb[mask_limb].real, image_limb[mask_limb].imag, s=1, color="purple")
        plt.scatter(image_limb2[mask_limb2].real, image_limb2[mask_limb2].imag, s=1, color="magenta")
        plt.axis("equal")
        #plt.hist(theta[mask_limb], bins=100)
        #plt.hist(r[mask_limb], bins=1000)
        plt.show()
    if(0):
        th = np.arctan2(image_limb.imag, image_limb.real)
        plt.hist(np.abs(image_limb[mask_limb]), bins=100)
        #plt.hist(th[mask_limb], bins=100)
        plt.show()