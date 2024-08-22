import jax.numpy as jnp
from jax import tree_util
from jax.tree_util import register_pytree_node_class
import jax
from jax import lax, jit
from functools import partial
from .point_source import lens_eq, _images_point_source
from .image_area0 import image_area0

@partial(jit, static_argnames=("nlenses", "NBIN"))
def image_area_all(w_center, rho, NBIN=10, nlenses=2, **_params):
    """
    Calculate the total image area and magnification for a given source position and lens configuration.

    Args:
        w_center (complex): Complex coordinate of the source center.
        rho (float): Radius of the source.
        NBIN (int, optional): Number of bins for discretizing the source radius. Default is 20.
        nlenses (int, optional): Number of lenses. Default is 2.
        **_params: Additional parameters for the lens model, including mass ratio (q) and separation (s).

    Returns:
        tuple: Total image area, magnification, and carry data for further processing.
    """
    q, s  = _params["q"], _params["s"] 
    incr  = jnp.abs(rho / NBIN)
    incr2 = incr * 0.5
    a  = 0.5 * s
    e1 = q / (1.0 + q) 
    _params = {"q": q, "s": s, "a": a, "e1": e1}

    w_center_mid = w_center - 0.5 * s * (1 - q) / (1 + q) 
    z_inits_mid, z_mask = _images_point_source(w_center_mid, nlenses=nlenses, **_params)
    z_inits = z_inits_mid + 0.5 * s * (1 - q) / (1 + q)

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

    # seach images from each start points
    area_all    = 0.0
    Nmax_images = 10

    def scan_fn(carry_scan, i):
        def run_search(carry_scan):
            area_i = 0.0
            (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all) = carry_scan
            carry  = (yi, indx, Nindx, xmin, xmax, area_x, y, dys) 
            
            # positive dy search 
            dy = incr
            z_init = z_inits[i]
            area, carry = image_area0(w_center, rho, z_init, dy, carry, **_params)
            area_i += area
            
            # negative dy search
            dy = -incr
            z_init = z_inits[i] + 1j * dy
            area, carry = image_area0(w_center, rho, z_init, dy, carry, **_params)
            area_i += area
            area_all += area_i
            (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
            carry_scan = (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all)

            return carry_scan
        
        def no_search(carry_scan):
            return carry_scan

        carry_scan = lax.cond(z_mask[i], 
                              run_search, 
                              no_search, 
                              carry_scan)
        
        return carry_scan, None
    
    carry_scan = (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all)
    carry_scan, _ = lax.scan(scan_fn, carry_scan, jnp.arange(Nmax_images))
    (yi, indx, Nindx, xmin, xmax, area_x, y, dys, area_all) = carry_scan 
    carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys) 

    # identify the protruding areas that were missed
    xmin_diff = jnp.where(jnp.diff(xmin)==0, jnp.inf, jnp.diff(xmin))
    xmax_diff = jnp.where(jnp.diff(xmax)==0,-jnp.inf, jnp.diff(xmax))
    y_diff    = jnp.where(jnp.diff(y)==0, jnp.inf, jnp.diff(y))
    fac_marg = 2.0
    upper_left  = (xmin_diff < -fac_marg * incr) & (dys[:-1] < 0) & (jnp.abs(y_diff) <= 2.0 * incr) 
    lower_left  = (xmin_diff < -fac_marg * incr) & (dys[:-1] > 0) & (jnp.abs(y_diff) <= 2.0 * incr)
    upper_right = (xmax_diff > fac_marg * incr)  & (dys[:-1] < 0) & (jnp.abs(y_diff) <= 2.0 * incr)
    lower_right = (xmax_diff > fac_marg * incr)  & (dys[:-1] > 0) & (jnp.abs(y_diff) <= 2.0 * incr)

    magnification = area_all / (jnp.pi * NBIN * NBIN) 
    return area_all, magnification, carry

    """ 
    # identify the protruding areas that are missed
    (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
    xmin_diff = jnp.where(jnp.diff(xmin)==0, jnp.inf, jnp.diff(xmin))
    xmax_diff = jnp.where(jnp.diff(xmax)==0,-jnp.inf, jnp.diff(xmax)) 
    fac_marg = 1.1
    upper_left  = (xmin_diff < -fac_marg * incr) & (dys[1:] < 0) 
    lower_left  = (xmin_diff < -fac_marg * incr) & (dys[1:] > 0) 
    upper_right = (xmax_diff > fac_marg * incr)  & (dys[1:] < 0) 
    lower_right = (xmax_diff > fac_marg * incr)  & (dys[1:] > 0) 

    fac = 3.0
    for k in jnp.where(upper_left)[0]:
        offset_factor = fac * jnp.abs((xmin[k + 2] - xmin[k + 1]) / incr).astype(int)
        z_init = jnp.complex128(xmin[k + 1] + offset_factor * incr + 1j * (y[k + 1] + incr))
        yi += 1
        carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
        area, carry = image_area0(w_center, rho, z_init, incr, carry, **_params) 
        (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
        area_all += area
        yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)

    for k in jnp.where(upper_right)[0]:
        #k + 1 is 伸びてる部分のyi
        offset_factor = fac * jnp.abs((xmax[k + 2] - xmax[k + 1]) / incr).astype(int)
        z_init = jnp.complex128(xmax[k + 1] - offset_factor * incr + 1j * (y[k + 1] + incr))
        yi += 1
        carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
        area, carry = image_area0(w_center, rho, z_init, incr, carry, **_params) 
        (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
        area_all += area
        yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)

    for k in jnp.where(lower_left)[0]:
        offset_factor = fac * jnp.abs((xmin[k + 2] - xmin[k + 1]) / incr).astype(int)
        z_init = jnp.complex128(xmin[k + 1] + offset_factor * incr + 1j * (y[k + 1] - incr))
        yi += 1
        carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
        area, carry = image_area0(w_center, rho, z_init, -incr, carry, **_params) 
        (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
        area_all += area
        yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)

    for k in jnp.where(lower_right)[0]:
        offset_factor = fac * jnp.abs((xmax[k + 2] - xmax[k + 1]) / incr).astype(int)
        z_init = jnp.complex128(xmax[k + 1] - offset_factor * incr + 1j * (y[k + 1] - incr))
        yi += 1
        carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
        area, carry = image_area0(w_center, rho, z_init, -incr, carry, **_params) 
        (yi, indx, Nindx, xmin, xmax, area_x, y, dys) = carry
        area_all += area
        yi = jax.lax.cond(area == 0, lambda _: yi - 1, lambda _: yi, None)

    carry = (yi, indx, Nindx, xmin, xmax, area_x, y, dys)
    magnification = area_all / (jnp.pi * NBIN * NBIN) 
    return area_all, magnification, carry
    """
