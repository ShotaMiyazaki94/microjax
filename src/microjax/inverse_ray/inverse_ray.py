import jax.numpy as jnp
from jax import lax, jit
from functools import partial
from ..point_source import _images_point_source
from .image_area0 import image_area0

@partial(jit, static_argnames=("nlenses", "NBIN", "Nlimb"))
def mag_inverse_ray(w_center, rho, NBIN=10, Nlimb=10, nlenses=2, **_params):
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

    # Source center as the start points of inverse-ray
    w_center_mid = w_center - 0.5 * s * (1 - q) / (1 + q) 
    z_inits_mid, z_mask = _images_point_source(w_center_mid, nlenses=nlenses, **_params)
    z_inits = z_inits_mid + 0.5 * s * (1 - q) / (1 + q)
    
    # Source limb as the start points of inverse-ray
    w_limb = w_center + jnp.array(rho * jnp.exp(1.0j * jnp.linspace(0.0, 2*jnp.pi, Nlimb)), dtype=complex)
    w_limb_shift = w_limb - 0.5*s*(1 - q)/(1 + q) # half-axis coordinate
    image, mask = _images_point_source(w_limb_shift, a=a, e1=e1) # half-axis coordinate
    image_limb = image + 0.5*s*(1 - q)/(1 + q)       # center-of-mass coordinate
    
    # combine and attribute it to given grids
    z_inits = jnp.append(z_inits.ravel(),image_limb.ravel())
    z_mask = jnp.append(z_mask.ravel(), mask.ravel())
    x_inits = jnp.int_(z_inits.real / incr) * incr
    y_inits = jnp.int_(z_inits.imag / incr) * incr
    z_inits = x_inits + 1j * y_inits 

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
    Nmax_images = len(z_inits)

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

    magnification = area_all / (jnp.pi * NBIN**2)
    return magnification, carry