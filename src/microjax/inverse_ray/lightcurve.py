# -*- coding: utf-8 -*-
"""
Computing the magnification of an extended source at an arbitrary
set of points in the source plane.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap 

#from .extended_source import mag_uniform
from microjax.inverse_ray.extended_source import mag_uniform, mag_binary
from microjax.point_source import _images_point_source
from microjax.multipole import _mag_hexadecapole
from microjax.utils import *
from microjax.inverse_ray.cond_extended import _caustics_proximity_test, _planetary_caustic_test

def mag_lc_vmap(w_points, rho, nlenses=2, batch_size=400,
                r_resolution=1000, th_resolution=4000, Nlimb=1000, u1=0.0, **params):
    if nlenses == 1:
        _params = {}
        x_cm = 0 # miyazaki
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q/(1.0 + q) 
        _params = {"a": a, "e1": e1, "q": q, "s": s}
        x_cm = a*(1.0 - q)/(1.0 + q)
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = 1.0 / (1.0 + q + q3) #miyazaki
        r3 = r3 * jnp.exp(1j * psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
        x_cm = a * (1.0 - q) / (1.0 + q)
    else:
        raise ValueError("nlenses must be <= 3")
    
    # compute quadrupole approximation at every point and a test where it is sufficient 
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    if nlenses==1:
        test = w_points > 2*rho
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) #miyazaki
    elif nlenses==2:
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params #miyazaki
        )
        test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

        test = lax.cond(q < 0.01, lambda:test1 & test2, lambda:test1)
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)
    
    mag_full = lambda w: mag_binary(w, rho, nlenses=nlenses, Nlimb=Nlimb, u1=u1, 
                                     r_resolution=r_resolution, th_resolution=th_resolution, **_params)
    mag_full_vmap = vmap(mag_full, in_axes=(0,))

    map_input = [test, mu_multi, w_points]
    result = lax.map(lambda xs: 
                     lax.cond(xs[0], 
                              lambda _: xs[1], 
                              lambda _: mag_full_vmap(xs[2]), 
                              None), 
                     map_input)
    return result

    #def batched_vmap(w_points, batch_size=400):
    #    results = []
    #    for i in range(0, len(w_points), batch_size):
    #        chunk = w_points[i:i + batch_size]
    #        results.append(vmap(mag_full)(chunk))
    #    return jnp.concatenate(results)
    #
    #return batched_vmap(w_points, batch_size=batch_size)



#@partial(jit,static_argnames=("nlenses","r_resolution", "th_resolution", "Nlimb", "u1"))
def mag_lc(w_points, rho, nlenses=2, r_resolution=500, th_resolution=500, Nlimb=2000, u1=0.0, **params):
    # set parameters for the lens system
    if nlenses == 1:
        _params = {}
        x_cm = 0 # miyazaki
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q/(1.0 + q) 
        _params = {"a": a, "e1": e1, "q": q, "s": s}
        x_cm = a*(1.0 - q)/(1.0 + q)
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = 1.0 / (1.0 + q + q3) #miyazaki
        r3 = r3 * jnp.exp(1j * psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
        x_cm = a * (1.0 - q) / (1.0 + q)
    else:
        raise ValueError("nlenses must be <= 3")

    # compute quadrupole approximation at every point and a test where it is sufficient 
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    if nlenses==1:
        test = w_points > 2*rho
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) #miyazaki
    elif nlenses==2:
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params #miyazaki
        )
        test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

        test = lax.cond(q < 0.01, lambda:test1 & test2, lambda:test1)
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)

    mag_full = lambda w: mag_binary(w, rho, nlenses=nlenses, Nlimb=Nlimb, u1=u1, 
                                     r_resolution=r_resolution, th_resolution=th_resolution, **_params)
    #mag_full_jit = jit(mag_full) 
    #def mag_full_vmap(w_points, batch_size=500):
    #    results = []
    #    for i in range(0, len(w_points), batch_size):
    #        chunk = w_points[i:i + batch_size]
    #        results.append(vmap(mag_full)(chunk))
    #    return jnp.concatenate(results)
    map_input = [test, mu_multi, w_points] 
    return lax.map(lambda xs: 
                        lax.cond(xs[0], 
                                lambda _: xs[1], 
                                lambda _: mag_full(xs[2]), 
                                None), 
                            map_input)

@partial(jit,static_argnames=("nlenses","r_resolution", "th_resolution", 
                              "Nlimb", "MAX_FULL_CALLS", "cubic"))
def mag_lc_uniform(w_points, rho, nlenses=2, r_resolution=500, th_resolution=500, 
                   Nlimb=500, MAX_FULL_CALLS = 100, cubic=True, **params):

    s = params.get("s", None)
    q = params.get("q", None)
    
    if nlenses == 1:
        _params = {}
        x_cm = 0.0
    elif nlenses == 2:
        if s is None or q is None:
            raise ValueError("For nlenses=2, 's' and 'q' must be provided.")
        a = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {"a": a, "e1": e1}
        x_cm = a * (1 - q) / (1 + q)
    elif nlenses == 3:
        q3 = params["q3"]
        r3 = params["r3"]
        psi = params["psi"]
        if s is None or q is None:
            raise ValueError("For nlenses=3, 's' and 'q' must be provided.")
        a = 0.5 * s
        total_mass = 1.0 + q + q3
        e1 = q / total_mass
        e2 = 1.0 / total_mass
        r3 = r3 * jnp.exp(1j * psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2}
        x_cm = a * (1.0 - q) / (1.0 + q)
    else:
        raise ValueError("nlenses must be 1, 2, or 3.")
    
    # Compute point images for a point source
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    if nlenses==1:
        test = w_points > 2*rho
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) #miyazaki
    elif nlenses==2:
        # Compute hexadecapole approximation at every point and a test where it is sufficient
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params #miyazaki
        )
        test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

        test = lax.cond(
            q < 0.01, 
            lambda:test1 & test2,
            lambda:test1,
        )
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)

    _params = {"q": q, "s": s} 
    mag_full = lambda w: mag_uniform(w, rho, 
                                     nlenses=nlenses, 
                                     r_resolution=r_resolution,
                                     th_resolution=th_resolution,
                                     Nlimb=Nlimb,
                                     cubic=cubic, 
                                     **_params)

    mag_full = jit(mag_full)
    if(1): # padding 
        idx_sorted = jnp.argsort(test)
        idx_full = idx_sorted[:MAX_FULL_CALLS]
        mag_extended = jit(vmap(mag_full))(w_points[idx_full])
        #def scan_body(carry, w):
        #    out = mag_full(w)
        #    return carry, out
        #_, mag_extended = lax.scan(scan_body, None, w_points[idx_full])
        mags = mu_multi.at[idx_full].set(mag_extended)
        mags = jnp.where(test, mu_multi, mags)
        return mags
    if(0): # lax.scan and lax.cond
        def scan_body(carry, xs):
            test_i, mu_i, w_i = xs
            out = lax.cond(test_i, lambda _: mu_i, mag_full, w_i)
            return carry, out
        _, result = lax.scan(scan_body, None, (test, mu_multi, w_points))
        return result
    if(0): # lax.map and lax.cond
        return lax.map(lambda xs: lax.cond(xs[0], lambda _: xs[1], mag_full, xs[2],),
                   [test, mu_multi, w_points])


if __name__ == "__main__":
    import time
    import jax
    jax.config.update("jax_enable_x64", True)
    #jax.config.update("jax_debug_nans", True)
    q = 0.05
    s = 1.0
    alpha = jnp.deg2rad(20) 
    tE = 30 
    t0 = 0.0 
    u0 = 0.1 
    rho = 5e-2

    nlenses = 2
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"a": a, "e1": e1}
    x_cm = a * (1 - q) / (1 + q)

    num_points = 500
    t  =  jnp.linspace(-0.5*tE, 0.5*tE, num_points)
    #t  =  jnp.linspace(-0.8*tE, 0.8*tE, num_points)
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    test_params = {"q": q, "s": s}  # Lens parameters

    Nlimb = 500
    r_resolution  = 500
    th_resolution = 500
    MAX_FULL_CALLS = 300

    cubic = True
    bins_r = 50
    bins_th = 120
    margin_r = 0.5
    margin_th= 0.5

    from microjax.caustics.extended_source import mag_extended_source
    import MulensModel as mm
    import VBBinaryLensing
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.a1 = 0.0
    VBBL.RelTol = 1e-4
    params_VBBL = [jnp.log(s), jnp.log(q), u0, alpha - jnp.pi, jnp.log(rho), jnp.log(tE), t0]

    def mag_vbbl(w0, rho, u1=0.0, accuracy=5e-05):
        a  = 0.5 * s
        e1 = 1.0 / (1.0 + q)
        e2 = 1.0 - e1  
        bl = mm.BinaryLens(e1, e2, 2*a)
        return bl.vbbl_magnification(w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)
    
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

    _ = mag_lc_uniform(w_points, rho, s=s, q=q, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic, Nlimb=Nlimb, MAX_FULL_CALLS=MAX_FULL_CALLS)
    print("start computation with mag_lc_uniform")
    start = time.time()
    magnifications = mag_lc_uniform(w_points, rho, s=s, q=q, 
                                    r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic,
                                    Nlimb=Nlimb, MAX_FULL_CALLS=MAX_FULL_CALLS)
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
    plt.savefig("mag_lc.pdf", bbox_inches="tight")
    print("mag_lc.pdf")
    plt.close()