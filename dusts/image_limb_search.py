import jax.numpy as jnp
from jax import tree_util
from jax.tree_util import register_pytree_node_class
import jax
from jax import lax, jit
from functools import partial
from ..src.microjax.point_source import lens_eq, _images_point_source
from ..src.microjax.inverse_ray.image_area0 import image_area0

@partial(jit, static_argnames=("NBIN", "Nlimbs", "nlenses"))
def image_limb_search(w_center, rho, NBIN=10, Nlimbs=1000, nlenses=2, **_params):

    q, s  = _params["q"], _params["s"] 
    incr  = jnp.abs(rho / NBIN)
    a  = 0.5 * s
    e1 = q / (1.0 + q) 
    _params = {"q": q, "s": s, "a": a, "e1": e1}

    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.pi * jnp.linspace(0.0, 2*jnp.pi, Nlimbs)), dtype=complex)
    w_limb_shift = w_limb - 0.5 * s * (1 - q) / (1 + q) # half-axis coordinate
    image, mask  = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
    image_limb   = image + 0.5 * s * (1 - q) / (1 + q)       # center-of-mass coordinate

    yi         = 0
    area_all   = 0.0
    max_iter   = int(1e+6)
    indx       = jnp.zeros((max_iter * 2, 6), dtype=int) # index for checking the overlaps
    Nindx      = jnp.zeros((max_iter * 2), dtype=int)     # Number of images at y_index
    xmin       = jnp.zeros((max_iter * 2))
    xmax       = jnp.zeros((max_iter * 2)) 
    area_x     = jnp.zeros((max_iter * 2)) 
    y          = jnp.zeros((max_iter * 2)) 
    dys        = jnp.zeros((max_iter * 2))

    # set image_limb as the start points of inverse-ray shooting
    area_all    = 0.0
    def scan_fn(carry_scan, i):
        area_i = 0.0
        (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all) = carry_scan
        carry  = (yi, indx, Nindx, xmin, xmax, area_x, y, dys) 
        for j in jnp.arange(6):
            z_init = image_limb[i][j]
            # Positive dy search
            area, carry = image_area0(w_center, rho, z_init, incr, carry, **_params)
            area_i += area
            # Negative dy search
            z_init_neg = z_init + 1j * (-incr)
            area, carry = image_area0(w_center, rho, z_init_neg, -incr, carry, **_params)
            area_i += area
        area_all += area_i
        (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
        carry_scan = (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all)
        return carry_scan, None

    carry_scan = (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all)
    carry_scan, _ = lax.scan(scan_fn, carry_scan, jnp.arange(Nlimbs))
    
    (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all) = carry_scan 
    carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    magnification = area_all / (jnp.pi * NBIN * NBIN) 
    return area_all, magnification, carry


