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

@partial(jit, static_argnames=["bins", "max_cluster", "zero_cut"])
def cluster_1d(arr, bins=100, max_cluster=5, zero_cut=True):
    #arr = jnp.where(arr == 0, -1, arr)
    #bin_max = arr.min(), arr.max()
    # This might cause the gradient error.
    if zero_cut:
        bin_min = jnp.min(jnp.where(arr == 0, np.inf, arr))
        bin_max = jnp.max(jnp.where(arr == 0, -np.inf, arr))
    else:
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

def refine_intervals(image_limb, mask_limb, r_starts, r_ends, th_starts, th_ends):
    image_limb_real = image_limb * mask_limb


if(1):
    w_center = jnp.complex128(0.1 - 0.1j)
    s = 1.0
    q = 0.01
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"a": a, "e1": e1, "q": q, "s": s}
    rho = 1e-2
    Nlimb = 300
    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=500, s=s, q=q) 
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    image_limb = image_limb.ravel()
    mask_limb = mask_limb.ravel()
    r     = jnp.abs(image_limb * mask_limb)
    theta = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi) * mask_limb
    
    bins=50
    r_mins, r_maxs = cluster_1d(r, bins=bins)
    th_mins, th_maxs = cluster_1d(theta, bins, zero_cut=False)
    for r_min, r_max in zip(r_mins, r_maxs):
        print(r_min, r_max) 
    for t_min, t_max in zip(th_mins, th_maxs):
        print(t_min, t_max) 

    r_non = r[r!=0]
    bin_min, bin_max = r_non.min(), r_non.max()
    delta = (bin_max - bin_min) / bins
    bin_edges = jnp.linspace(bin_min - delta, bin_max + delta, bins + 3, endpoint=True)


    #plt.hist(np.array(r), bins=bins)
    plt.hist(np.array(r_non), bins=bins)
    ymin, ymax = plt.ylim()
    plt.vlines(r_mins, ymin=0, ymax=ymax, color="orange", zorder=2)
    plt.vlines(r_maxs, ymin=0, ymax=ymax, color="red", zorder=1)
    plt.vlines(bin_edges, ymin=0, ymax=0.1*ymax, color="gray", zorder=1)
    plt.xlim(0.9*r_non.min(), 1.1*r_non.max())
    plt.savefig("test.pdf")
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