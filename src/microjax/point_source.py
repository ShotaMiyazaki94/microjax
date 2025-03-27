from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from .poly_solver import poly_roots
from .utils import match_points
from .coeffs import _poly_coeffs_binary, _poly_coeffs_triple 
from .coeffs import _poly_coeffs_critical_triple, _poly_coeffs_critical_binary

#@partial(jit, static_argnames=("nlenses"))
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
    
#@partial(jit, static_argnames=("nlenses"))
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
    
@partial(jit, static_argnames=("npts", "nlenses"))
def critical_and_caustic_curves(npts=200, nlenses=2, **params):
    phi = jnp.linspace(-np.pi, np.pi, npts)

    def apply_match_points(carry, z):
        idcs = match_points(carry, z)
        return z[idcs], z[idcs]

    if nlenses == 1:  
        return jnp.exp(-1j * phi), jnp.zeros(npts).astype(jnp.complex128)

    if nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q / (1.0 + q) 
        _params = {"a": a, "e1": e1}
        coeffs = jnp.moveaxis(_poly_coeffs_critical_binary(phi, a, e1), 0, -1)

    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a  = 0.5 * s
        e1 = q / (1.0 + q + q3)  
        e2 = 1.0 / (1.0 + q + q3) 
        r3 = r3*jnp.exp(1j*psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2}
        coeffs = jnp.moveaxis(_poly_coeffs_critical_triple(phi, a, r3, e1, e2), 0, -1)

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    # Compute roots
    z_cr = poly_roots(coeffs)
    # Permute roots so that they form contiguous curves
    init = z_cr[0, :]
    _, z_cr = lax.scan(apply_match_points, init, z_cr)
    z_cr = z_cr.T
    # Caustics are critical curves mapped by the lens equation
    z_ca = lens_eq(z_cr, nlenses=nlenses, **_params)

    # Shift from mid-point to center-of-mass
    x_cm = 0.5 * s * (1.0 - q) / (1.0 + q)
    z_cr, z_ca = z_cr + x_cm, z_ca + x_cm 

    return z_cr, z_ca

@partial(jit, static_argnames=("nlenses", "custom_init"))
def _images_point_source(w, nlenses=2, custom_init=False, z_init=None, **params):
    """
    Computes the image positions for a point source under the influence of multiple lenses
    in the mid-point coordinate system. Handles configurations with one, two, or three lenses.

    Args:
        w (complex or array_like of complex):
            Source position(s) in the complex plane, representing the coordinates in the mid-point system.
        nlenses (int, optional):
            Number of lenses affecting the source light. Valid values are 1, 2, or 3. Defaults to 2.
        custom_init (bool, optional):
            Flag to indicate whether a custom initial guess for the root finding should be used.
            Defaults to False.
        z_init (array_like, optional):
            Initial guess for the root positions if custom_init is True.
        **params:
            Additional parameters specific to the lens configuration, such as:
            - a (float): Separation between primary and secondary lens (used when nlenses > 1).
            - e1, e2 (floats): Mass ratios or other relevant parameters.
            - r3 (float): Position or relevant parameter for the third lens (used when nlenses == 3).

    Returns:
        Tuple (z, z_mask):
            z (array_like):
                The calculated image positions in the complex plane.
            z_mask (array_like):
                Boolean array indicating whether each position satisfies the lens equation to within a tolerance.
    """
    if nlenses == 1:
        w_abs_sq = w.real**2 + w.imag**2
        #  Compute the image locations using the quadratic formula
        z1 = 0.5 * w * (1.0 + jnp.sqrt(1 + 4 / w_abs_sq))
        z2 = 0.5 * w * (1.0 - jnp.sqrt(1 + 4 / w_abs_sq))
        z = jnp.stack(jnp.array([z1, z2]))
        return z, jnp.ones(z.shape).astype(jnp.bool_)
    
    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        coeffs = _poly_coeffs_binary(w, a, e1)
    
    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        coeffs = _poly_coeffs_triple(w, a, r3, e1, e2)

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")
    
    if custom_init:
        z = poly_roots(coeffs, custom_init=True, roots_init=z_init)
    else:
        z = poly_roots(coeffs)
    
    z = jnp.moveaxis(z, -1, 0)
    # Evaluate the lens equation at the roots
    lens_eq_eval = lens_eq(z, nlenses=nlenses, **params) - w
    # Mask out roots which don't satisfy the lens equation
    z_mask = jnp.abs(lens_eq_eval) < 1e-6
    
    return z, z_mask 

@partial(jit, static_argnames=("nlenses"))
def _images_point_source_sequential(w, nlenses=2, **params):
    
    def fn(w, z_init=None, custom_init=False):
        if custom_init:
            z, z_mask = _images_point_source(w, nlenses=nlenses, custom_init=True, 
                                             z_init=z_init,**params)
        else:
            z, z_mask = _images_point_source(w, nlenses=nlenses, **params)
        return z, z_mask

    z_first, z_mask_first = fn(w[0])
    
    def body_fn(z_prev, w):
        z, z_mask = fn(w, z_init=z_prev, custom_init=True)
        return z, (z, z_mask)

    _, xs = lax.scan(body_fn, z_first, w[1:])
    z, z_mask = xs

    # Append to the initial point
    z = jnp.concatenate([z_first[None, :], z])
    z_mask = jnp.concatenate([z_mask_first[None, :], z_mask])

    return z.T, z_mask.T 

@partial(jit, static_argnames=("nlenses"))
def mag_point_source(w, nlenses=2, **params):
    if nlenses == 1:
        _params = {}
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q / (1.0 + q) 
        _params = {**params, "a": a, "e1": e1}
        x_cm = a * (1.0 - q) / (1.0 + q)
        w   -= x_cm 
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5 * s
        e1 = q / (1.0 + q + q3) 
        e2 = 1.0/(1.0 + q + q3)
        r3 = r3 * jnp.exp(1j * psi)
        _params = {**params, "a": a, "r3": r3, "e1": e1, "e2": e2}
        x_cm = a * (1.0 - q) / (1.0 + q)
        w   -= x_cm
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")
    
    z, z_mask = _images_point_source(w, nlenses=nlenses, **_params)
    det = lens_eq_det_jac(z, nlenses=nlenses, **_params)
    mag = (1.0 / jnp.abs(det)) * z_mask 
    return mag.sum(axis=0).reshape(w.shape)