"""Adaptive lightcurve computation mixing multipole and inverse-ray methods.

This module computes microlensing lightcurves by defaulting to the fast
hexadecapole approximation and selectively invoking the full inverse-ray
finite-source integration near caustics or when required by accuracy tests.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap 

from microjax.inverse_ray.extended_source import mag_uniform, mag_limb_dark
from microjax.point_source import _images_point_source
from microjax.multipole import _mag_hexadecapole
from microjax.utils import *
from microjax.inverse_ray.cond_extended import _caustics_proximity_test, _planetary_caustic_test
from microjax.inverse_ray.cond_extended import test_full
from typing import Tuple

# Consistent array alias used across modules
Array = jnp.ndarray

@partial(jit,static_argnames=("r_resolution", "th_resolution", "u1", "delta_c", 
                              "bins_r", "bins_th", "margin_r", "margin_th", 
                              "Nlimb", "MAX_FULL_CALLS", "chunk_size"))
def mag_binary(
    w_points: Array,
    rho: float,
    r_resolution: int = 1000,
    th_resolution: int = 1000,
    u1: float = 0.0,
    delta_c: float = 0.01,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 1.0,
    margin_th: float = 1.0,
    MAX_FULL_CALLS: int = 500,
    chunk_size: int = 100,
    **params,
) -> Array:
    """Binary-lens lightcurve with adaptive full-solve selection.

    Parameters follow the underlying inverse-ray integrators. The selection
    uses a proximity test to caustics plus planetary-caustic checks for small
    mass ratios. Returns magnifications per input time/position.
    """
    s = params.get("s", None)
    q = params.get("q", None)
    if s is None or q is None:
        raise ValueError("For nlenses=2, 's' and 'q' must be provided.") 
    
    nlenses = 2
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {**params, "a": a, "e1": e1}
    x_cm = a * (1.0 - q) / (1.0 + q)
    w_points_shifted = w_points - x_cm

    # test whether inverse-ray shooting needed or not. test==False means needed.
    z, z_mask = _images_point_source(w_points_shifted, nlenses=nlenses, a=a, e1=e1)
    mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    test1 = _caustics_proximity_test(w_points_shifted, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params)
    test2 = _planetary_caustic_test(w_points_shifted, rho, **_params)
    test = jnp.where(q < 0.01, test1 & test2, test1)

    if u1 == 0.0:
        def _mag_full(w):
            return mag_uniform(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution = th_resolution,
                               bins_r = bins_r, bins_th = bins_th, margin_r = margin_r, margin_th = margin_th, 
                               Nlimb = Nlimb, **_params)
    else:
        def _mag_full(w):
            return mag_limb_dark(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution= th_resolution,
                                 u1 = u1, delta_c = delta_c, bins_r = bins_r, bins_th = bins_th, margin_r = margin_r,
                                 margin_th = margin_th, Nlimb = Nlimb, **_params)
    
    idx_sorted = jnp.argsort(test)
    idx_full = idx_sorted[:MAX_FULL_CALLS]

    def chunked_vmap(func, data, chunk_size):
        N = data.shape[0]
        pad_len = (-N) % chunk_size
        chunks = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:])
        return lax.map(lambda c: vmap(func)(c), chunks).reshape(-1, *data.shape[2:])[:N]
    
    def chunked_vmap_scan(func, data, chunk_size):
        N = data.shape[0]
        pad_len  = (-N) % chunk_size
        chunks   = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:]) 
        #@jax.checkpoint
        def body(carry, chunk):
            #out = vmap(jax.checkpoint(func))(chunk)             
            out = vmap(func)(chunk)                            
            return carry, out               
        _, outs = lax.scan(body, None, chunks)   # outs → (T, chunk_size, ...)
        return outs.reshape(-1, *data.shape[2:])[:N]

    _mag_full = jax.checkpoint(_mag_full, policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
    mag_extended = chunked_vmap(_mag_full, w_points[idx_full], chunk_size)
    #mag_extended = chunked_vmap_scan(_mag_full, w_points[idx_full], chunk_size)
    mags = mu_multi.at[idx_full].set(mag_extended)
    mags = jnp.where(test, mu_multi, mags)
    return mags 

@partial(jit,static_argnames=("r_resolution", "th_resolution", "u1", "delta_c",
                              "bins_r", "bins_th", "margin_r", "margin_th", 
                              "Nlimb", "MAX_FULL_CALLS", "chunk_size"))
def mag_triple(
    w_points: Array,
    rho: float,
    r_resolution: int = 1000,
    th_resolution: int = 1000,
    u1: float = 0.0,
    delta_c: float = 0.01,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 1.0,
    margin_th: float = 1.0,
    MAX_FULL_CALLS: int = 500,
    chunk_size: int = 50,
    **params,
) -> Array:
    """Triple-lens lightcurve with adaptive full-solve selection."""
    nlenses = 3
    s, q, q3 = params["s"], params["q"], params["q3"]
    a = 0.5 * s
    e1 = q / (1.0 + q + q3) 
    e2 = 1.0/(1.0 + q + q3)
    #r3 = r3 * jnp.exp(1j * psi)
    #_params = {"a": a, "r3": r3, "e1": e1, "e2": e2}
    _params = {**params, "a": a, "e1": e1, "e2": e2}
    #_params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
    x_cm = a * (1.0 - q) / (1.0 + q)
    w_points_shifted = w_points - x_cm
    
    z, z_mask = _images_point_source(w_points_shifted, nlenses=nlenses, **_params) 
    mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    test = jnp.zeros_like(w_points).astype(jnp.bool_)

    if u1 == 0.0:
        def _mag_full(w):
            return mag_uniform(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution = th_resolution,
                               bins_r = bins_r, bins_th = bins_th, margin_r = margin_r, margin_th = margin_th, 
                               Nlimb = Nlimb, **_params)
    else:
        def _mag_full(w):
            return mag_limb_dark(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution= th_resolution,
                                 u1 = u1, delta_c = delta_c, bins_r = bins_r, bins_th = bins_th, margin_r = margin_r,
                                 margin_th = margin_th, Nlimb = Nlimb, **_params)

    idx_sorted = jnp.argsort(test)
    idx_full = idx_sorted[:MAX_FULL_CALLS]

    def chunked_vmap(func, data, chunk_size):
        N = data.shape[0]
        pad_len = (-N) % chunk_size
        chunks = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:])
        return lax.map(lambda c: vmap(func)(c), chunks).reshape(-1, *data.shape[2:])[:N]

    _mag_full = jax.checkpoint(_mag_full)
    mag_extended = chunked_vmap(_mag_full, w_points[idx_full], chunk_size)
    mags = mu_multi.at[idx_full].set(mag_extended)
    mags = jnp.where(test, mu_multi, mags)
    return mags 
    

if __name__ == "__main__":
    import time
    import jax
    jax.config.update("jax_enable_x64", True)
    #jax.config.update("jax_debug_nans", True)
    
    if(1):
        t0, u0, tE = 6.83640951e+03, 2.24211333e-01, 1.33559958e+02 
        s, q, alpha = 9.16157288e-01, 5.87559438e-04, jnp.deg2rad(1.00066409e+02)
        rho, pi_EN, pi_EE = 2.44003713e-03, 1.82341182e-01,9.58542572e-02

    if(0):
        q = 0.01
        s = 1.0
        alpha = jnp.deg2rad(10) 
        tE = 30 
        t0 = 0.0 
        u0 = 0.0 
        rho = 0.02

    nlenses = 2
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"a": a, "e1": e1}
    x_cm = a * (1 - q) / (1 + q)

    num_points = 2000
    t  =  jnp.linspace(-1.0*tE + t0, 1.0*tE + t0, num_points)
    #t  =  jnp.linspace(-0.8*tE, 0.8*tE, num_points)
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    test_params = {"q": q, "s": s}  # Lens parameters

    Nlimb = 500
    r_resolution  = 1000
    th_resolution = 1000
    MAX_FULL_CALLS = 100

    cubic = True
    bins_r = 50
    bins_th = 120
    margin_r = 1.0
    margin_th= 1.0

    import MulensModel as mm
    import VBBinaryLensing
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.a1 = 0.0
    VBBL.RelTol = 1e-5
    params_VBBL = [jnp.log(s), jnp.log(q), u0, alpha - jnp.pi, jnp.log(rho), jnp.log(tE), t0]

    #magn2  = lambda w0: jnp.array([mag_vbbl(w, rho) for w in w0])
    
    print("number of data points: %d"%(num_points))
    from microjax.point_source import mag_point_source, critical_and_caustic_curves
    mag_point_source(w_points, s=s, q=q)
    start = time.time()
    mags_poi = mag_point_source(w_points, s=s, q=q)
    mags_poi.block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms) per points for point-source in microjax"%(end-start, 1000*(end - start)/num_points)) 

    from microjax.multipole import _mag_hexadecapole
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    _, _ = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) 
    start = time.time()
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    mu_multi.block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for hexadecapole in microjax"%(end-start, 1000*(end - start)/num_points)) 


    start = time.time()
    magnifications2, y1, y2 = jnp.array(VBBL.BinaryLightCurve(params_VBBL, t))
    magnifications2.block_until_ready() 
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for VBBinaryLensing"%(end - start,1000*(end - start)/num_points))

    _ = mag_binary(w_points, rho, s=s, q=q, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic, 
                       Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th, margin_r=margin_r, margin_th=margin_th, MAX_FULL_CALLS=MAX_FULL_CALLS)
    #_ = mag_lc_uniform(w_points, rho, s=s, q=q, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic, 
    #                   Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th, margin_r=margin_r, margin_th=margin_th, MAX_FULL_CALLS=MAX_FULL_CALLS)
    print("start computation with mag_lc_uniform, %d full calculation, %d rbin, %d thbin"%(MAX_FULL_CALLS, r_resolution, th_resolution))
    start = time.time()
    magnifications = mag_binary(w_points, rho, s=s, q=q, r_resolution=r_resolution, th_resolution=th_resolution,
                                    cubic=cubic, Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th, 
                                    margin_r=margin_r, margin_th=margin_th, MAX_FULL_CALLS=MAX_FULL_CALLS)
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for mag_lc_uniform in microjax"%(end-start, 1000*(end - start)/num_points))
   
    # Print out the result
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.ticker as ticker
    import seaborn as sns
    sns.set_theme(style="ticks")

    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)

    fig, ax_ = plt.subplots(2,1,figsize=(8,6), sharex=True, gridspec_kw=dict(hspace=0.1, height_ratios=[4,1]))
    ax  = ax_[0]
    ax1 = ax_[1]
    ax_in = inset_axes(ax,
        width="60%", height="60%", 
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(0.35, 0.35, 0.6, 0.6)
    )
    ax_in.set_aspect(1)
    ax_in.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")
    for cc in caustic_curves:
        ax_in.plot(cc.real, cc.imag, color='red', lw=0.7)
    circles = [plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, ec="blue", zorder=2) 
               for xi, yi in zip(w_points.real, w_points.imag)
               ]
    c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.5)
    ax_in.add_collection(c)
    ax_in.set_aspect(1)
    ax_in.set(xlim=(-1., 1.2), ylim=(-1.0, 1.))
    ax_in.plot(-q/(1+q) * s, 0 , ".",c="k")
    ax_in.plot((1.0)/(1+q) * s, 0 ,".",c="k")

    ax.plot(t, magnifications, ".", label="microjax", zorder=1)
    ax.plot(t, magnifications2, "-", label="VBBinaryLensing", zorder=2)
    ylim = ax.get_ylim()
    #ax.plot(t, mags_poi, "--", label="point-source", zorder=-1, color="gray")
    ax.set_title("mag_lc_uniform")
    ax.grid(ls=":")
    ax.set_ylabel("magnification")
    ax.set_ylim(ylim[0], ylim[1])
    ax1.plot(t, jnp.abs(magnifications - magnifications2)/magnifications2, "-", ms=1)
    ax1.grid(ls=":")
    ax1.set_yticks(10**jnp.arange(-4, -2, 1))
    ax1.set_ylabel("relative diff")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0, 10**-2, 10**-4, 10**-6], numticks=10))
    ax1.set_ylim(1e-6, 1e-2)
    ax.legend(loc="upper left")
    ax1.set_xlabel("time (days)")
    
    mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    test1 = _caustics_proximity_test(
        w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params 
    )
    test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

    test = lax.cond(
        q < 0.01, 
        lambda:test1 & test2,
        lambda:test1,
    )
    ax.plot(t[~test], magnifications[~test], ".", color="red", zorder=20)
    print("full num: %d"%jnp.sum(~test))
    plt.show()
    #plt.savefig("z_fig/mag_lc.png", bbox_inches="tight", dpi=300)
    #print("z_fig/mag_lc.png")
    plt.close()
