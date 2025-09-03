# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Point-source microlensing utilities (JAX).

This module provides JAX-friendly implementations of core operations for
point-source gravitational microlensing by up to three point-mass lenses:

- `lens_eq`: complex lens equation mapping image-plane coordinates `z` to
  source-plane coordinates `w`.
- `lens_eq_det_jac`: determinant of the Jacobian of the lens mapping, used to
  compute magnification through `|det J|^{-1}`.
- `critical_and_caustic_curves`: critical curves in the image plane and their
  mapped caustics in the source plane.
- `_images_point_source`: image positions for given source position(s) by
  solving the corresponding complex polynomial.
- `_images_point_source_sequential`: helper that tracks images across a
  sequence of sources by reusing the previous roots as initialization.
- `mag_point_source`: total point-source magnification (sum over images).

All functions are written to be compatible with `jax.jit` and vectorized
evaluation over batches of input where appropriate. Complex numbers encode 2D
coordinates with `x + i y` convention.

Coordinate conventions
- Binary/triple lens configurations use the mid-point coordinate system for
  coefficient construction; `mag_point_source` shifts the input source-plane
  coordinates to the center-of-mass when required and handles the inverse shift
  internally where necessary.

References
- Bartolic, F. "caustics" (MIT License) — original inspiration and some
  coefficient-generation routines, modified here for JAX and extended triple
  lens support.
"""

from functools import partial
from typing import Tuple

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from .poly_solver import poly_roots
from .utils import match_points
from .coeffs import _poly_coeffs_binary, _poly_coeffs_critical_binary
from .coeffs import _poly_coeffs_critical_triple, _poly_coeffs_triple_CM

#@partial(jit, static_argnames=("nlenses"))
@partial(jit, static_argnames=("nlenses",))
def lens_eq(z: jax.Array, nlenses: int = 2, **params) -> jax.Array:
    """Lens equation mapping image-plane `z` to source-plane `w`.

    Parameters
    - z: complex scalar or array-like; image-plane coordinate(s).
    - nlenses: number of point-mass lenses; supported values are 1, 2, or 3.
    - params: lens parameters, depending on `nlenses`:
      - nlenses=1: no additional parameters.
      - nlenses=2: `a` (half-separation), `e1` (mass fraction for lens at `+a`),
        with the second lens having fraction `1 - e1` at `-a`.
      - nlenses=3: `a` (half-separation of the binary on real axis), `r3`
        (distance of third lens), `psi` (angle of the third lens w.r.t. x-axis),
        `e1`, `e2` (mass fractions at `+a` and `-a` respectively). The third
        lens fraction is `1 - e1 - e2` at position `r3 * exp(i psi)`.

    Returns
    - Complex scalar/array with the same shape as `z`, giving `w = w(z)`.

    Notes
    - All arithmetic is performed in complex form; gradients propagate through
      JAX as expected.
    """
    zbar = jnp.conjugate(z)

    if nlenses == 1:
        return z - 1.0 / zbar

    if nlenses == 2:
        a, e1 = params["a"], params["e1"]
        return z - e1 / (zbar - a) - (1.0 - e1) / (zbar + a)

    if nlenses == 3:
        a, r3, psi, e1, e2 = (
            params["a"],
            params["r3"],
            params["psi"],
            params["e1"],
            params["e2"],
        )
        r3_complex = r3 * jnp.exp(1j * psi)
        return (
            z
            - e1 / (zbar - a)
            - e2 / (zbar + a)
            - (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3_complex))
        )

    raise ValueError("`nlenses` has to be set to be <= 3.")
    
#@partial(jit, static_argnames=("nlenses"))
@partial(jit, static_argnames=("nlenses",))
def lens_eq_det_jac(z: jax.Array, nlenses: int = 2, **params) -> jax.Array:
    """Determinant of the Jacobian of the lens mapping at `z`.

    Parameters
    - z: complex scalar or array-like; image-plane coordinate(s).
    - nlenses: number of lenses (1, 2, or 3).
    - params: same as for `lens_eq` for the corresponding configuration.

    Returns
    - Real scalar/array of the same shape as `z` with `det J(z)`.

    Notes
    - The point-source magnification is `|det J|^{-1}` for each image.
    """
    zbar = jnp.conjugate(z)

    if nlenses == 1:
        return 1.0 - 1.0 / jnp.abs(zbar**2)

    if nlenses == 2:
        a, e1 = params["a"], params["e1"]
        return 1.0 - jnp.abs(
            e1 / (zbar - a) ** 2 + (1.0 - e1) / (zbar + a) ** 2
        ) ** 2

    if nlenses == 3:
        a, r3, psi, e1, e2 = (
            params["a"],
            params["r3"],
            params["psi"],
            params["e1"],
            params["e2"],
        )
        r3_complex = r3 * jnp.exp(1j * psi)
        return (
            1.0
            - jnp.abs(
                e1 / (zbar - a) ** 2
                + e2 / (zbar + a) ** 2
                + (1.0 - e1 - e2) / (zbar - jnp.conjugate(r3_complex)) ** 2
            )
            ** 2
        )

    raise ValueError("`nlenses` has to be set to be <= 3.")
    
@partial(jit, static_argnames=("npts", "nlenses"))
def critical_and_caustic_curves(
    npts: int = 200,
    nlenses: int = 2,
    **params,
):
    """Compute critical curves and mapped caustics.

    Parameters
    - npts: number of sampling points on the unit circle used to construct the
      polynomial for the critical curves.
    - nlenses: number of lenses (1, 2, or 3).
    - params: lens configuration parameters:
      - nlenses=1: no additional parameters.
      - nlenses=2: `s` (separation), `q` (mass ratio m2/m1). Internally, we use
        `a = s/2` and `e1 = q/(1+q)`.
      - nlenses=3: `s`, `q`, `q3` (third mass ratio), `r3` (radius), `psi`
        (angle). Internally `a = s/2`, `e1 = q/(1+q+q3)`, `e2 = 1/(1+q+q3)`.

    Returns
    - z_cr: complex array of shape `(Nbranches, npts)` with critical curves in
      the image plane; branches are ordered to form contiguous curves.
    - z_ca: complex array of shape `(Nbranches, npts)` with mapped caustics in
      the source plane via the lens equation.

    Notes
    - For `nlenses=1`, the critical curve is the unit circle and the caustic is
      a single point at the origin.
    - Output is shifted from mid-point to center-of-mass for consistency with
      the rest of the library.
    """
    phi = jnp.linspace(-np.pi, np.pi, npts)

    def apply_match_points(carry, z):
        idcs = match_points(carry, z)
        return z[idcs], z[idcs]

    if nlenses == 1:
        return jnp.exp(-1j * phi), jnp.zeros(npts, dtype=jnp.complex128)

    if nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {"a": a, "e1": e1}
        coeffs = jnp.moveaxis(_poly_coeffs_critical_binary(phi, a, e1), 0, -1)

    elif nlenses == 3:
        s, q, q3, r3, psi = (
            params["s"],
            params["q"],
            params["q3"],
            params["r3"],
            params["psi"],
        )
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = 1.0 / (1.0 + q + q3)
        r3_complex = r3 * jnp.exp(1j * psi)
        _params = {**params, "a": a, "e1": e1, "e2": e2, "r3": r3, "psi": psi}
        coeffs = jnp.moveaxis(
            _poly_coeffs_critical_triple(phi, a, r3_complex, e1, e2), 0, -1
        )

    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    # Compute roots along the sampling circle
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
def _images_point_source(
    w: jax.Array,
    nlenses: int = 2,
    custom_init: bool = False,
    z_init: jax.Array | None = None,
    **params,
):
    """Solve for image positions for a point source.

    Parameters
    - w: complex scalar or array-like; source-plane coordinate(s). Broadcasting
      over arrays is supported.
    - nlenses: number of lenses (1, 2, or 3).
    - custom_init: if True, use `z_init` as initial guesses for the polynomial
      root solver.
    - z_init: complex array of initial roots; must match the trailing shape of
      the polynomial degree for the given configuration when `custom_init=True`.
    - params: lens parameters in the mid-point coordinate system:
      - nlenses=1: no additional parameters.
      - nlenses=2: `a`, `e1` as in `lens_eq`.
      - nlenses=3: `a`, `r3`, `psi`, `e1`, `e2` as in `lens_eq`.

    Returns
    - z: complex array with shape `(Nimages, ...)` where `...` matches `w`'s
      shape; `Nimages` is 2, 5, or 10 for 1-, 2-, or 3-lens cases respectively.
    - z_mask: boolean array with the same shape as `z` indicating which roots
      satisfy the lens equation within a tight tolerance.

    Notes
    - For the triple lens, coefficients are constructed in center-of-mass
      coordinates and shifted back using the value returned from
      `_poly_coeffs_triple_CM`.
    - The mask threshold is `1e-6` (`1e-3` for the triple path); callers should
      treat masked-out roots as non-physical.
    """
    if nlenses == 1:
        w_abs_sq = w.real**2 + w.imag**2
        # Compute the image locations using the quadratic formula
        z1 = 0.5 * w * (1.0 + jnp.sqrt(1 + 4 / w_abs_sq))
        z2 = 0.5 * w * (1.0 - jnp.sqrt(1 + 4 / w_abs_sq))
        z = jnp.stack([z1, z2])
        return z, jnp.ones(z.shape, dtype=jnp.bool_)
    
    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        coeffs = _poly_coeffs_binary(w, a, e1)
    
    elif nlenses == 3:
        a, r3, psi, e1, e2 = params["a"], params["r3"], params["psi"], params["e1"], params["e2"]
        r3_complex = r3 * jnp.exp(1j * psi)
        coeffs, shift_cm = _poly_coeffs_triple_CM(w, a, r3_complex, e1, e2)
        #coeffs = _poly_coeffs_triple(w, a, r3_complex, e1, e2)
        z = poly_roots(coeffs)
        z += shift_cm
        z = jnp.moveaxis(z, -1, 0)
        lens_eq_eval = lens_eq(z, nlenses=3, **params) - w
        z_mask = jnp.abs(lens_eq_eval) < 1e-3
        return z, z_mask 

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
    """Sequential image tracking across a 1D path in `w`.

    Parameters
    - w: 1D complex array of source positions. The solver computes images for
      the first element and then reuses each solution as the initial guess for
      the next element, which helps maintain branch continuity.
    - nlenses: number of lenses (1, 2, or 3).
    - params: parameters forwarded to `_images_point_source`.

    Returns
    - z: complex array of shape `(Nimages, w.size)` with tracked images.
    - z_mask: boolean array with the same shape as `z`.

    Notes
    - This is primarily useful for drawing image tracks while a source moves
      along a curve; it is not a generic batched solver.
    """
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
    """Total point-source magnification for 1–3 lens configurations.

    Parameters
    - w: complex scalar or array-like; source-plane coordinate(s).
    - nlenses: number of lenses (1, 2, or 3).
    - params: lens parameters by configuration:
      - nlenses=1: none required.
      - nlenses=2: `s` (separation), `q` (mass ratio m2/m1). Internally
        `a = s/2` and `e1 = q/(1+q)`; the source is shifted to the
        center-of-mass for consistency with the polynomial construction.
      - nlenses=3: `s`, `q`, `q3`, `r3`, `psi`. Internally `a = s/2`,
        `e1 = q/(1+q+q3)`, `e2 = 1/(1+q+q3)`; a consistent shift to the COM is
        applied as required by the coefficient builder.

    Returns
    - Real scalar/array of the same shape as `w` with the total magnification.

    Notes
    - Magnification is computed as the sum of `|det J|^{-1}` over valid image
      branches returned by `_images_point_source`.
    """
    if nlenses == 1:
        _params = {}
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {**params, "a": a, "e1": e1}
        x_cm = a * (1.0 - q) / (1.0 + q)
        w -= x_cm
    elif nlenses == 3:
        s, q, q3, r3, psi = (
            params["s"],
            params["q"],
            params["q3"],
            params["r3"],
            params["psi"],
        )
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = 1.0 / (1.0 + q + q3)
        _params = {**params, "a": a, "e1": e1, "e2": e2, "r3": r3, "psi": psi}
        x_cm = a * (1.0 - q) / (1.0 + q)
        w -= x_cm
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    z, z_mask = _images_point_source(w, nlenses=nlenses, **_params)
    det = lens_eq_det_jac(z, nlenses=nlenses, **_params)
    mag = (1.0 / jnp.abs(det)) * z_mask
    return mag.sum(axis=0).reshape(w.shape)
