from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, lax

from microjax.poly_solver import poly_roots_EA_multi as poly_roots
#from .ehrlich_aberth_primitive import poly_roots
from .utils import match_points
from microjax.coeffs import _poly_coeffs_binary, _poly_coeffs_triple, _poly_coeffs_critical_triple, _poly_coeffs_critical_binary

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

@partial(jit, static_argnames=("npts", "nlenses"))
def critical_and_caustic_curves(npts=200, nlenses=2, **params):
    """
    Compute critical and caustic curves for visualization purposes.

    If `nlenses` is 2 only the parameters `s` and `q` should be specified. If 
    `nlenses` is 3, the parameters `s`, `q`, `q3`, `r3` and `psi` should be 
    specified.

    Args:
        npts (int): Number of points to when computing the critical curves.
        nlenses (int): Number of lenses in the system.
        s (float): Separation between the two lenses. The first lens is located 
            at $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$ on the real line.
        q (float): Mass ratio defined as $m_2/m_1$.
        q3 (float): Mass ratio defined as $m_3/m_1$.
        r3 (float): Magnitude of the complex position of the third lens.
        psi (float): Phase angle of the complex position of the third lens.

    Returns:
        tuple: Tuple (critical_curves, caustic_curves) where both elements are
            arrays with shape (`nlenses`, `npts`) containing continuous segments of 
            the critical curves and caustics.
    """
    phi = jnp.linspace(-np.pi, np.pi, npts)

    def apply_match_points(carry, z):
        idcs = match_points(carry, z)
        return z[idcs], z[idcs]

    if nlenses == 1:  # trivial
        return jnp.exp(-1j * phi), jnp.zeros(npts).astype(jnp.complex128)

    if nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5*s
        e1 = q/(1 + q) #miyazaki
        #e1 = 1/(1 + q)
        _params = {"a": a, "e1": e1}
        coeffs = jnp.moveaxis(_poly_coeffs_critical_binary(phi, a, e1), 0, -1)

    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5*s
        e1 = q/(1 + q + q3)  #miyazaki
        #e1 = q/(1 + q + q3)
        e2 = 1/(1 + q + q3) #miyazaki
        #e2 = q*e1
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

    # Shift by the centre of mass
    x_cm = 0.5*s*(1 - q)/(1 + q)
    z_cr, z_ca = z_cr + x_cm, z_ca + x_cm #miyazaki
    #z_cr, z_ca = z_cr - x_cm, z_ca - x_cm

    return z_cr, z_ca


@partial(
    jit, static_argnames=("nlenses",)
)
def _images_point_source(
    w,
    nlenses=2,
    **params
):
    """
    Compute the image positions z and whether they meets the lens equation z_mask
    from the given source positions w and lens components.
    Args:
        w (array_like): Source position in the complex plane.
            please note that the origin should be the mid-point between 1st and 2nd lens, not center of mass.
    Return:

    """
    if nlenses == 1:
        w_abs_sq = w.real**2 + w.imag**2
        #  Compute the image locations using the quadratic formula
        z1 = 0.5 * w * (1.0 + jnp.sqrt(1 + 4 / w_abs_sq))
        z2 = 0.5 * w * (1.0 - jnp.sqrt(1 + 4 / w_abs_sq))
        z = jnp.stack(jnp.array([z1, z2]))

        return z, jnp.ones(z.shape).astype(jnp.bool_)

    elif nlenses == 2:
        #s, q = params["s"], params["q"]
        #a = 0.5*s
        #e1 = q/(1 + q) #miyazaki
        #_params = {"a": a, "e1": e1}
        a, e1 = params["a"], params["e1"]
        # Compute complex polynomial coefficients for each element of w
        coeffs = _poly_coeffs_binary(w, a, e1)

    elif nlenses == 3:
        #, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        #a = 0.5*s
        #e1 = q/(1 + q + q3)  #miyazaki
        #e1 = q/(1 + q + q3)
        #e2 = 1/(1 + q + q3) #miyazaki
        #e2 = q*e1
        #r3 = r3*jnp.exp(1j*psi)
        #_params = {"a": a, "r3": r3, "e1": e1, "e2": e2}
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        # Compute complex polynomial coefficients for each element of w
        coeffs = _poly_coeffs_triple(w, a, r3, e1, e2)

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    # Compute roots
    z = poly_roots(coeffs)
    z = jnp.moveaxis(z, -1, 0)

    # Evaluate the lens equation at the roots
    lens_eq_eval = lens_eq(z, nlenses=nlenses, **params) - w

    # Mask out roots which don't satisfy the lens equation
    z_mask = jnp.abs(lens_eq_eval) < 1e-6

    return z, z_mask 

def _images_point_source_sequential(w, nlenses=2, **params,):
    """
    Same as `images_point_source` except w is a 1D arrray and the images 
    are computed sequentially using `lax.scan` such that the first set 
    of images is initialized using the default initialization and the 
    subsequent images are initialized using the previous images as a starting
    point.
    """
    def fn(w):
        z, z_mask = _images_point_source(w, nlenses=nlenses, **params,)
        return z, z_mask
    
    z_first, z_mask_first = fn(w[0])

    def body_fn(z_prev, w):
        z, z_mask = fn(w)
        return z, (z, z_mask)

    _, xs = lax.scan(body_fn, z_first, w[1:])
    z, z_mask = xs

    # Append to the initial point
    z = jnp.concatenate([z_first[None, :], z])
    z_mask = jnp.concatenate([z_mask_first[None, :], z_mask])

    return z.T, z_mask.T  


@partial(jit, static_argnames=("nlenses"))
def mag_point_source(w, nlenses=2, **params):
    """
    Compute the magnification of a point source for a system with `nlenses`
    lenses. If `nlenses` is 2 (binary lens) or 3 (triple lens), the coordinate
    system is set up such that the the origin is at the center of mass of the 
    first two lenses which are both located on the real line. The location of 
    the first lens is $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$. The 
    optional third lens is located at an arbitrary position in the complex plane 
    $r_3e^{-i\psi}$. For a single lens lens the magnification is computed 
    analytically. For binary and triple lenses computing the magnification 
    involves solving for the roots of a complex polynomial with degree 
    (`nlenses`**2 + 1) using the Elrich-Aberth algorithm.

    If `nlenses` is 2 only the parameters `s` and `q` should be specified. If 
    `nlenses` is 3, the parameters `s`, `q`, `q3`, `r3` and `psi` should be 
    specified.

    Args:
        w (array_like): Source position in the complex plane.
        nlenses (int): Number of lenses in the system.
        s (float): Separation between the two lenses. The first lens is located 
            at $-sq/(1 + q)$ and the second lens is at $s/(1 + q)$ on the real line.
        q (float): Mass ratio defined as $m_2/m_1$.
        q3 (float): Mass ratio defined as $m_3/m_1$.
        r3 (float): Magnitude of the complex position of the third lens.
        psi (float): Phase angle of the complex position of the third lens.
        roots_itmax (int, optional): Number of iterations for the root solver.
        roots_compensated (bool, optional): Whether to use the compensated
            arithmetic version of the Ehrlich-Aberth root solver.

    Returns:
        array_like: The point source magnification evaluated at w.
    """
    if nlenses == 1:
        _params = {}
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5*s
        e1 = q/(1 + q) #miyazaki
        #e1 = 1/(1 + q)
        _params = {"a": a, "e1": e1}

        # Shift w by x_cm
        x_cm = a*(1 - q)/(1 + q)
        w -= x_cm #miyazaki the origin shift is center-of-mass -> midpoint
        #w += x_cm
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5*s
        e1 = q/(1 + q + q3) #miyazaki
        #e1 = q/(1 + q + q3)
        e2 = 1/(1 + q + q3)#miyazaki
        #e2 = q*e1
        r3 = r3*jnp.exp(1j*psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2}

        # Shift w by x_cm
        x_cm = a*(1 - q)/(1 + q)
        w -= x_cm #miyazaki
        #w += x_cm
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    z, z_mask = _images_point_source(w, nlenses=nlenses, **_params)
    det = lens_eq_det_jac(z, nlenses=nlenses, **_params)
    mag = (1.0 / jnp.abs(det)) * z_mask 
    return mag.sum(axis=0).reshape(w.shape)