import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap, custom_jvp
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
from microjax.inverse_ray.merge_area import calc_source_limb, determine_grid_regions
from microjax.inverse_ray.merge_area_2 import grid_intervals, cluster_1d, merge_theta
from microjax.inverse_ray.limb_darkening import Is_limb_1st
from microjax.inverse_ray.boundary import in_source, distance_from_source, calc_facB
from microjax.inverse_ray.boundary import distance_from_source_adaptive
from microjax.inverse_ray.extended_source import mag_uniform

jax.config.update("jax_enable_x64", True)
w_center = jnp.complex128(0 + 0.0j)
q, s = 0.1, 1.0
rho = 0.01
Nlimb = 2000
r_resolution = 500
th_resolution= 500

def check(image_limb, mask_limb, rho, bins=100, max_cluster=5, optimize=False, margin_r=0.5, margin_th=0.1):
    image_limb = image_limb.ravel()
    mask_limb = mask_limb.ravel()
    r     = jnp.abs(image_limb * mask_limb)
    theta = jnp.mod(jnp.arctan2(image_limb.imag, image_limb.real), 2*jnp.pi) * mask_limb
    r_mins, r_maxs   = cluster_1d(r, bins=bins, max_cluster=max_cluster)
    th_mins, th_maxs = cluster_1d(theta, bins, mode_r=False, max_cluster=2*max_cluster)
    r_map = jnp.array([r_mins, r_maxs]).T
    th_map = jnp.array([th_mins, th_maxs]).T
    th_map = merge_theta(th_map)[-max_cluster:]
    return 


a  = 0.5 * s
e1 = q / (1.0 + q)
_params = {"q": q, "s": s, "a": a, "e1": e1}

shifted = 0.5 * s * (1 - q) / (1 + q)  
w_center_shifted = w_center - shifted
image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r=0.1, offset_th=0.1, nlenses=2)
mag_uniform(w_center, rho, nlenses=2, r_resolution=r_resolution, th_resolution=th_resolution,
            Nlimb=Nlimb, offset_r=0.1, offset_th=0.1, cubic=True, **_params)
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    #r_scan, th_scan = determine_grid_regions(image_limb, mask_limb, rho, offset_r=0.1, offset_th=0.1, nlenses=2)
    mag_uniform(w_center, rho, nlenses=2, r_resolution=r_resolution, th_resolution=th_resolution, 
                Nlimb=Nlimb, offset_r=0.1, offset_th=0.1, cubic=True, **_params)