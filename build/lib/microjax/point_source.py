from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from .poly_solver import poly_roots
#from .poly_solver import poly_roots_EA_multi as poly_roots
#from .poly_solver import poly_roots_EA_init as poly_roots_init_single
#from .poly_solver import poly_roots_EA_multi_init as poly_roots_init
from .utils import match_points
from .coeffs import _poly_coeffs_binary, _poly_coeffs_triple, _poly_coeffs_critical_triple, _poly_coeffs_critical_binary

def _lens_eq_single(z):
    zbar = jnp.conjugate(z)
    return z - 1 / zbar

def _lens_eq_binary(z, a, e1):
    """
    compute source position given image position with binary-lens

    Args:
        z (array_like): 
            Image position in the complex plane.
        a (float):
            Half of separation between the two lenses.
        e1 (float):
            mass fraction for second mass, e1 = m2 / (m1 + m2)
    Return:
        array_like: 
            The source position evaluated at z. 
    """

    zbar = jnp.conjugate(z)
    return z - e1 / (zbar - a) - (1.0 - e1) / (zbar + a)

def _lens_eq_triple(z, a, r3, e1, e2):
    """
    compute source position given image position with triple-lens

    Args:
        z (array_like): 
            Image position in the complex plane.
        a (float):
            Half of separation between primary and secondary lenses.
        r3 (complex):
            The coordinate of the third mass from the mid-point.
        e1 (float):
            mass fraction for second mass, e1 = m2 / (m1 + m2 + m3)
        e2 (float):
            mass fraction for first mass, e2 = m1 / (m1 + m2 + m3)
    Return:
        array_like: 
            The source position evaluated at z.  
    """
    zbar = jnp.conjugate(z)
    return (
        z - e1 / (zbar - a)
          - e2 / (zbar + a)
          - (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3))
    )

def _det_jac_single(z):
    zbar = jnp.conjugate(z)
    return 1.0 - 1.0 / jnp.abs(zbar**2)

def _det_jac_binary(z, a, e1):
    zbar = jnp.conjugate(z)
    return 1.0 - jnp.abs(e1 / (zbar - a) ** 2 + (1.0 - e1) / (zbar + a) ** 2) ** 2

def _det_jac_triple(z, a, r3, e1, e2):
    zbar = jnp.conjugate(z)
    return (
        1.0 - jnp.abs(
            e1 / (zbar - a) ** 2
            + e2 / (zbar + a) ** 2
            + (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3)) ** 2
        ) ** 2
    )

@jit
def _images_point_source_single(w):
    w_abs_sq = w.real**2 + w.imag**2
    z1 = 0.5 * w * (1.0 + jnp.sqrt(1 + 4 / w_abs_sq))
    z2 = 0.5 * w * (1.0 - jnp.sqrt(1 + 4 / w_abs_sq))
    z = jnp.array([z1, z2])
    z_mask = jnp.ones(z.shape, dtype=jnp.bool_)
    return z, z_mask

@jit
def _images_point_source_binary(w, a, e1):
    """
    compute image positions with binary-lens in the mid-point coordinates

    Args:
        w (array_like): 
            Source position in the complex plane of the mid-point coordinates.
        a (float): 
            Half of separation between the two lenses.
        e1 (float): 
            Mass fraction defined as $e1 = m2/(m1+m2)$.
    Return:
        Tuple (z: array_like, z_mask: array_like): 
            z (array_like): The image position evaluated by w.
            z_mask (array_like): The bool whether the image position meets the lens equation.
    """
    coeffs = _poly_coeffs_binary(w, a, e1)
    z = poly_roots(coeffs)
    z = jnp.moveaxis(z, -1, 0)
    lens_eq_eval = _lens_eq_binary(z, a, e1) - w
    z_mask = jnp.abs(lens_eq_eval) < 1e-6
    return z, z_mask

@jit
def _images_point_source_triple(w, a, r3, e1, e2):
    """
    compute image positions with triple-lens

    Args:
        w (array_like): 
            Source position in the complex plane of the mid-point coordinates.
        a (float): 
            Half of separation between the two lenses.
        r3 (complex): 
            Complex position of the third lens from the mid-point.
        e1 (float): 
            Mass fraction defined as $e1 = m2 / (m1+m2+m3)$.
        e2 (float): 
            Mass fraction defined as $e2 = m1 / (m1+m2+m3)$.
    Return:
        Tuple (z: array_like, z_mask: array_like): 
            z (array_like): The image position evaluated by w.
            z_mask (array_like): The bool whether the image position meets the lens equation.
    """
    coeffs = _poly_coeffs_triple(w, a, r3, e1, e2)
    z = poly_roots(coeffs)
    z = jnp.moveaxis(z, -1, 0)
    lens_eq_eval = _lens_eq_triple(z, a, r3, e1, e2) - w
    z_mask = jnp.abs(lens_eq_eval) < 1e-6
    return z, z_mask

def mag_point_source_single(w):
    """
    compute point-source magnification with single-lens

    Args:
        w (array_like): 
            Source position in the complex plane.
    Return:
        array_like: 
            The point source magnification evaluated at w. 
    """
    z, z_mask = _images_point_source_single(w)
    det = _det_jac_single(z)
    mag = (1.0 / jnp.abs(det)) * z_mask
    return mag.sum(axis=0).reshape(w.shape)

@jit
def mag_point_source_binary(w, s, q):
    """
    compute point-source magnification with binary-lens

    Args:
        w (array_like): 
            Source position in the complex plane of center-of-mass coordinates.
        s (float): 
            Separation between the two lenses. The first lens is located 
            at $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$ on the real line.
        q (float): 
            Mass ratio defined as $m_2/m_1$.
    Return:
        array_like: 
            The point source magnification evaluated at w. 
    """
    a = 0.5 * s
    e1 = q / (1 + q)
    # center-of-mass -> midpoint
    x_cm = a*(1 - q)/(1 + q)
    w -= x_cm   
    # calculate image positions z
    z, z_mask = _images_point_source_binary(w, a, e1)
    det = _det_jac_binary(z, a, e1)
    mag = (1.0 / jnp.abs(det)) * z_mask
    return mag.sum(axis=0).reshape(w.shape)

def mag_point_source_triple(w, s, q, q3, r3, psi):
    """
    compute point-source magnification with triple-lens

    Args:
        w (array_like): 
            Source position in the complex plane of center-of-mass coordinates.
        s (float): 
            Separation between the two lenses. The first lens is located 
            at $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$ on the real line.
        q (float): 
            Mass ratio defined as $m_2/m_1$.
        q3 (float): 
            Mass ratio defined as $m_3/m_1$.
        r3 (float): 
            Magnitude of the complex position of the third lens.
        psi (float): 
            Phase angle of the complex position of the third lens.
    Return:
        array_like: 
            The point source magnification evaluated at w. 
    """
    a = 0.5 * s
    e1 = q / (1.0 + q + q3) #miyazaki
    e2 = 1.0 / (1.0 + q + q3)#miyazaki
    r3 = r3*jnp.exp(1j*psi)
    # Shift w by x_cm
    x_cm = a*(1.0 - q)/(1.0 + q)
    w -= x_cm 
    # calculate image positions z 
    z, z_mask = _images_point_source_triple(w, a, r3, e1, e2)
    det = _det_jac_triple(z, a, r3, e1, e2)
    mag = (1.0 / jnp.abs(det)) * z_mask
    return mag.sum(axis=0).reshape(w.shape)

@partial(jit, static_argnames=("npts"))
def critical_and_caustic_curves_binary(npts=1000, s=1.0, q=1.0):
    """
    Compute critical and caustic curves for visualization purposes.

    Args:
        npts (int): 
            Number of points to when computing the critical curves.
        s (float): 
            Separation between the two lenses. The first lens is located 
            at $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$ on the real line.
        q (float): 
            Mass ratio defined as $m_2/m_1$.
    Returns:
        tuple: 
            Tuple (critical_curves, caustic_curves) where both elements are
            arrays with shape (`npts`) containing continuous segments of 
            the critical curves and caustics.
    """
    def apply_match_points(carry, z):
        idcs = match_points(carry, z)
        return z[idcs], z[idcs]

    phi = jnp.linspace(-np.pi, np.pi, npts)
    a = 0.5 * s
    e1 = q / (1.0 + q)
    coeffs = jnp.moveaxis(_poly_coeffs_critical_binary(phi, a, e1), 0, -1)
    z_cr = poly_roots(coeffs)

    # creating sequential curves
    init = z_cr[0, :]
    _, z_cr = lax.scan(apply_match_points, init, z_cr)
    z_cr = z_cr.T

    z_ca = _lens_eq_binary(z_cr, a, e1)
    x_cm = a * (1.0 - q) / (1.0 + q)
    z_cr, z_ca = z_cr + x_cm, z_ca + x_cm
    return z_cr, z_ca

@partial(jit, static_argnames=("npts"))
def critical_and_caustic_curves_triple(npts=1000, s=1.0, q=1.0, q3=1.0, r3=1.0, psi=np.pi):
    """
    Compute critical and caustic curves for visualization purposes.

    Args:
        npts (int): 
            Number of points to when computing the critical curves.
        s (float): 
            Separation between the two lenses. The first lens is located 
            at $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$ on the real line.
        q (float): 
            Mass ratio defined as $m_2/m_1$.
        q3 (float): 
            Mass ratio defined as $m_3/m_1$.
        r3 (float): 
            Magnitude of the complex position of the third lens.
        psi (float): 
            Phase angle of the complex position of the third lens.
    Returns:
        tuple: 
            Tuple (critical_curves, caustic_curves) where both elements are
            arrays with shape (`npts`) containing continuous segments of 
            the critical curves and caustics.
    """
    def apply_match_points(carry, z):
        idcs = match_points(carry, z)
        return z[idcs], z[idcs]
    phi = jnp.linspace(-np.pi, np.pi, npts)
    
    a = 0.5 * s
    e1 = q / (1.0 + q + q3)
    e2 = 1.0 / (1.0 + q + q3)
    r3 = r3 * jnp.exp(1j * psi)
    coeffs = jnp.moveaxis(_poly_coeffs_critical_triple(phi, a, r3, e1, e2), 0, -1)
    z_cr = poly_roots(coeffs)

    # creating sequential curves
    init = z_cr[0, :]
    _, z_cr = lax.scan(apply_match_points, init, z_cr)
    z_cr = z_cr.T

    z_ca = _lens_eq_triple(z_cr, a, r3, e1, e2)
    x_cm = a * (1.0 - q) / (1.0 + q)
    z_cr, z_ca = z_cr + x_cm, z_ca + x_cm
    return z_cr, z_ca

########################################################
def _images_point_source(w, nlenses=2, custom_init=False, z_init=None, **params):
    if nlenses == 2:
        a, e1 = params["a"], params["e1"]
        coeffs = _poly_coeffs_binary(w, a, e1)
    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        coeffs = _poly_coeffs_triple(w, a, r3, e1, e2)

    if custom_init:
        z = poly_roots(coeffs, custom_init=True, roots_init=z_init)
    else:
        z = poly_roots(coeffs)
    z = jnp.moveaxis(z, -1, 0)

    lens_eq_eval = lens_eq(z, nlenses=nlenses, **params) - w
    z_mask = jnp.abs(lens_eq_eval) < 1e-6

    return z, z_mask 

def lens_eq(z, nlenses=2, **params):
    
    zbar = jnp.conjugate(z)
    if nlenses == 1:
        return z - 1 / zbar

    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        return z - e1 / (zbar - a) - (1.0 - e1) / (zbar + a)

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        return (
            z
            - e1 / (zbar - a)
            - e2 / (zbar + a)
            - (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3))
        )

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.") 

def lens_eq_det_jac(z, nlenses=2, **params):
    zbar = jnp.conjugate(z)

    if nlenses == 1:
        return 1.0 - 1.0 / jnp.abs(zbar**2)

    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        return 1.0 - jnp.abs(e1 / (zbar - a) ** 2 + (1.0 - e1) / (zbar + a) ** 2) ** 2

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        return (
            1.0
            - jnp.abs(
                e1 / (zbar - a) ** 2
                + e2 / (zbar + a) ** 2
                + (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3)) ** 2
            )
            ** 2
        )
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")




'''
@jit
def _images_point_source_binary_init(w, a, e1, z_init):
    coeffs = _poly_coeffs_binary(w, a, e1)
    #z_init = z_init.ravel()
    z = poly_roots_init(coeffs[None, :], z_init.T)
    #z = poly_roots_init(coeffs[None, :], z_init.T)
    z = jnp.moveaxis(z, -1, 0)
    lens_eq_eval = _lens_eq_binary(z, a, e1) - w
    z_mask = jnp.abs(lens_eq_eval) < 1e-6
    return z, z_mask

def _images_point_source_binary_sequential(w, a, e1):
    """
    Same as `images_point_source` except w is a 1D arrray and the images 
    are computed sequentially using `lax.scan` such that the first set 
    of images is initialized using the default initialization and the 
    subsequent images are initialized using the previous images as a starting
    point.
    """
    def fn_init(w, z_init):
        z, z_mask = _images_point_source_binary_init(w, a, e1, z_init)
        return z, z_mask
    def fn(w):
        z, z_mask = _images_point_source_binary(w, a, e1)
        return z, z_mask

    z_first, z_mask_first = _images_point_source_binary(jnp.array([w[0]]), a, e1)
    print("first:", z_first.shape, z_mask_first.shape)

    def body_fn(z_prev, w):
        z, z_mask = fn_init(w, z_init=z_prev)
        return z, (z, z_mask)
    
    _, xs = lax.scan(body_fn, z_first, w[1:])
    z, z_mask = xs

    # Append to the initial point
    z = jnp.concatenate([z_first[None, :], z])
    z_mask = jnp.concatenate([z_mask_first[None, :], z_mask])

    return z.T, z_mask.T
'''