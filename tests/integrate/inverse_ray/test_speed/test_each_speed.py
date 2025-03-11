from jax import config 
config.update('jax_enable_x64', True) 
import pandas as pd
import numpy as np
import time
import jax.numpy as jnp
from jax import grad, jit, vmap, random


import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap, custom_jvp
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
from microjax.inverse_ray.merge_area import calc_source_limb, determine_grid_regions
from microjax.inverse_ray.merge_area_2 import grid_intervals
from microjax.inverse_ray.limb_darkening import Is_limb_1st
from microjax.inverse_ray.boundary import in_source, distance_from_source, calc_facB
from microjax.inverse_ray.boundary import distance_from_source_adaptive
from microjax.inverse_ray.extended_source import mag_uniform
from microjax.multipole import _mag_hexadecapole

def mag_hex(w_points, rho, nlenses=2, **params):
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
    
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    return mu_multi

#@partial(jit, static_argnames=("nlenses", "r_resolution", "th_resolution", "Nlimb", "offset_r", "offset_th", "cubic",))
def test1(w_center, rho, nlenses=2, r_resolution=500, th_resolution=500, Nlimb=2000, offset_r=0.1, offset_th=0.1, cubic=True, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    #r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r, offset_th, nlenses=nlenses)
    return jnp.sum(mask_limb)

#@partial(jit, static_argnames=("nlenses", "r_resolution", "th_resolution", "Nlimb", "offset_r", "offset_th", "cubic",))
def test2(w_center, rho, nlenses=2, r_resolution=500, th_resolution=500, Nlimb=2000, offset_r=0.1, offset_th=0.1, cubic=True, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r, offset_th, nlenses=nlenses)
    return jnp.sum(r_scan + th_scan)

#@partial(jit, static_argnames=("nlenses", "r_resolution", "th_resolution", "Nlimb", "offset_r", "offset_th", "cubic",))
def test3(w_center, rho, nlenses=2, r_resolution=500, th_resolution=500, Nlimb=2000, offset_r=0.1, offset_th=0.1, cubic=True, **_params):
    q, s = _params["q"], _params["s"]
    a  = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"q": q, "s": s, "a": a, "e1": e1}
    
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_scan, th_scan = grid_intervals(image_limb, mask_limb, rho)
    return jnp.sum(r_scan + th_scan)

if __name__ == "__main__":
    import time
    jax.config.update("jax_enable_x64", True)
    #jax.config.update("jax_debug_nans", True)
    q = 0.5
    s = 0.9
    test_params = {"q": q, "s": s}  # Lens parameters
    alpha = jnp.deg2rad(30) # angle between lens axis and source trajectory
    tE = 10 # einstein radius crossing time
    t0 = 0.0 # time of peak magnification
    u0 = 0.1 # impact parameter
    rho = 0.1

    num_points = int(1e+3)
    t  =  jnp.linspace(-0.8*tE, 0.8*tE, num_points)
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

    @jax.jit
    def mag_mj(w):
        return mag_uniform(w, rho, s=s, q=q, Nlimb=2000, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic)

    r_resolution  = 500
    th_resolution = 500
    cubic = True

    from microjax.point_source import mag_point_source, critical_and_caustic_curves
    @jax.jit
    def mag_point_scan(w_points):
        def body_fun(carry, w):
            result = mag_point_source(w, s=s, q=q)
            return carry, result
        _, results = lax.scan(body_fun, None, w_points)
        return results
    
    _ = mag_point_source(w_points, s=s, q=q)
    print("start computation")
    start = time.time()
    #mags_poi = mag_point_scan(w_points)
    mags_poi = mag_point_source(w_points, s=s, q=q)
    mags_poi.block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for point-source in microjax"%(end-start, 1000*(end - start)/num_points)) 

    mags_hex = mag_hex(w_points, rho, s=s, q=q)
    print("start computation")
    start = time.time()
    mags_hex = mag_hex(w_points, rho, s=s, q=q).block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for hexadecapole in microjax"%(end-start, 1000*(end - start)/num_points)) 

    Nlimb = 200

    test1_ = lambda w: test1(w, rho, s=s, q=q, Nlimb=Nlimb) 
    @jax.jit
    def test1_scan(w_points):
        def body_fun(carry, w):
            result = test1_(w)
            return carry, result
        _, results = lax.scan(body_fun, None, w_points)
        return results

    test1_vmap = jit(vmap(test1_))
    _ = test1_vmap(w_points)
    print("start computation")
    start = time.time()
    #_= test1_scan(w_points).block_until_ready()
    _= test1_vmap(w_points).block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for calc_source_limb in microjax"%(end-start, 1000*(end - start)/num_points)) 

    test2_ = lambda w: test2(w, rho, s=s, q=q, Nlimb=Nlimb) 
    @jax.jit
    def test2_scan(w_points):
        def body_fun(carry, w):
            result = test2_(w)
            return carry, result
        _, results = lax.scan(body_fun, None, w_points)
        return results

    test2_vmap = jit(vmap(lambda w: test2(w, rho, s=s, q=q, Nlimb=Nlimb)))
    _ = test2_vmap(w_points)
    print("start computation")
    start = time.time()
    #_ = test2_scan(w_points).block_until_ready()
    _ = test2_vmap(w_points).block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for determine_region in microjax"%(end-start, 1000*(end - start)/num_points))

    #test3_vmap = jit(vmap(lambda w: test3(w, rho, s=s, q=q, Nlimb=Nlimb)))
    #_ = test3_vmap(w_points)
    #print("start computation")
    #start = time.time()
    #_ = test3_vmap(w_points).block_until_ready()
    #end = time.time()
    #rint("computation time: %.3f sec (%.3f ms per points) for grid_interval in microjax"%(end-start, 1000*(end - start)/num_points)) 

    #@jax.jit
    mag_uni = vmap(lambda w: mag_uniform(w, rho, s=s, q=q, Nlimb=Nlimb, r_resolution=r_resolution, th_resolution=th_resolution, cubic=cubic))
    _ = mag_uni(w_points)
    print("start computation")
    start = time.time()
    mags_uni = mag_uni(w_points).block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for mag_uniform in microjax"%(end-start, 1000*(end - start)/num_points)) 

