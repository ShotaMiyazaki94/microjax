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
point-source gravitational microlensing by up to three point-mass lenses.

Functions
---------

- ``lens_eq``: complex lens equation mapping image-plane coordinates ``z`` to
  source-plane coordinates ``w``.
- ``lens_eq_det_jac``: determinant of the Jacobian of the lens mapping, used to
  compute magnification through ``|det J|^{-1}``.
- ``critical_and_caustic_curves``: critical curves in the image plane and their
  mapped caustics in the source plane.
- ``_images_point_source``: image positions for given source position(s) by
  solving the corresponding complex polynomial.
- ``_images_point_source_sequential``: helper that tracks images across a
  sequence of sources by reusing previous roots as initialization.
- ``mag_point_source``: total point-source magnification (sum over images).

All functions are compatible with ``jax.jit`` and support batched evaluation
where appropriate. Complex numbers encode 2D coordinates using the ``x + i y``
convention.

Coordinate conventions
----------------------

- Binary/triple lens configurations use the mid-point coordinate system for
  coefficient construction. ``mag_point_source`` shifts input source-plane
  coordinates to the center of mass when required and handles the inverse shift
  internally.

References
----------

- Bartolic, F. ``caustics`` (MIT License) — original inspiration and some
  coefficient-generation routines, modified here for JAX and extended triple
  lens support.
"""

__all__ = [
    "lens_eq",
    "lens_eq_det_jac",
    "critical_and_caustic_curves",
    "mag_point_source",
]

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
    """Lens equation mapping image-plane ``z`` to source-plane ``w``.

    Parameters
    ----------
    z : jax.Array
        Complex scalar or array of image-plane coordinates.
    nlenses : int, optional
        Number of point-mass lenses (1, 2, or 3). Defaults to 2.
    params : dict
        Lens configuration parameters. For ``nlenses = 2`` expect ``a`` (half
        separation) and ``e1`` (mass fraction at ``+a``). For ``nlenses = 3``
        expect ``a``, ``r3``, ``psi``, ``e1`` and ``e2``.

    Returns
    -------
    jax.Array
        Source-plane coordinates with the same shape as ``z``.

    Notes
    -----
    All arithmetic is performed in complex form; gradients propagate through
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
    """Determinant of the Jacobian of the lens mapping at ``z``.

    Parameters
    ----------
    z : jax.Array
        Complex scalar or array of image-plane coordinates.
    nlenses : int, optional
        Number of point-mass lenses (1, 2, or 3). Defaults to 2.
    params : dict
        Lens configuration parameters matching those accepted by
        :func:`lens_eq`.

    Returns
    -------
    jax.Array
        Real array with the same shape as ``z`` storing ``det J(z)``.

    Notes
    -----
    Point-source magnification is ``|det J|^{-1}`` for each image.
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
    ----------
    npts : int, optional
        Number of sampling points on the unit circle used to construct the
        critical-curve polynomial. Defaults to 200.
    nlenses : int, optional
        Number of point-mass lenses (1, 2, or 3). Defaults to 2.
    params : dict
        Lens configuration parameters:

        - ``nlenses = 1``: no additional parameters.
        - ``nlenses = 2``: ``s`` (separation) and ``q`` (mass ratio ``m2/m1``).
          Internally we set ``a = s / 2`` and ``e1 = q / (1 + q)``.
        - ``nlenses = 3``: ``s``, ``q``, ``q3`` (third mass ratio), ``r3``
          (radius), ``psi`` (angle). Internally ``a = s / 2``, ``e1 = q / (1 +
          q + q3)``, ``e2 = 1 / (1 + q + q3)``.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(z_cr, z_ca)`` where ``z_cr`` contains critical curves and ``z_ca``
        the mapped caustics. Both arrays have shape ``(N_branches, npts)``.

    Notes
    -----
    - For ``nlenses = 1`` the critical curve is the unit circle and the caustic
      collapses to the origin.
    - Output is shifted from mid-point to center of mass for consistency with
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
    ----------
    w : jax.Array
        Complex scalar or array of source-plane coordinate(s). Broadcasting
        over arrays is supported.
    nlenses : int, optional
        Number of point-mass lenses (1, 2, or 3). Defaults to 2.
    custom_init : bool, optional
        When ``True``, reuse ``z_init`` as the initial guess for the root
        solver. Defaults to ``False``.
    z_init : jax.Array or None, optional
        Complex initial guesses with shape matching the trailing dimension of
        the polynomial degree. Required when ``custom_init`` is ``True``.
    **params
        Lens parameters expressed in the mid-point coordinate frame:

        - ``nlenses = 1``: no additional parameters.
        - ``nlenses = 2``: ``a`` (half-separation) and ``e1`` (mass fraction at
          ``+a``) as in :func:`lens_eq`.
        - ``nlenses = 3``: ``a``, ``r3``, ``psi``, ``e1`` and ``e2`` as in
          :func:`lens_eq`.

    Returns
    -------
    z : jax.Array
        Image locations with shape ``(N_images, ...)`` where ``...`` matches
        ``w``. ``N_images`` is 2, 5, or 10 for 1-, 2-, or 3-lens systems.
    z_mask : jax.Array
        Boolean mask with the same shape as ``z`` indicating which roots
        satisfy the tolerance criteria.

    Notes
    -----
    - For triple lenses the polynomial is constructed in center-of-mass
      coordinates using :func:`_poly_coeffs_triple_CM`; the resulting images are
      shifted back before being returned.
    - The mask threshold is ``1e-6`` (``1e-3`` for the triple branch); masked
      roots should be treated as non-physical solutions.
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
    """Sequentially track images along a 1D source trajectory.

    Parameters
    ----------
    w : jax.Array
        One-dimensional complex array of source positions. The solver computes
        images for ``w[0]`` and reuses each solution as the initial guess for
        the following element, promoting branch continuity.
    nlenses : int, optional
        Number of lenses (1, 2, or 3). Defaults to 2.
    **params
        Additional keyword arguments forwarded to
        :func:`_images_point_source`.

    Returns
    -------
    z : jax.Array
        Complex array of shape ``(N_images, w.size)`` storing the tracked image
        positions.
    z_mask : jax.Array
        Boolean array with the same shape as ``z`` storing the validity mask.

    Notes
    -----
    This helper is primarily intended for visualising image tracks along a
    trajectory rather than as a general batched solver.
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
    ----------
    w : jax.Array
        Complex scalar or array of source-plane coordinates.
    nlenses : int, optional
        Number of point-mass lenses (1, 2, or 3). Defaults to 2.
    params : dict
        Lens parameters depend on ``nlenses``:

        - ``nlenses = 1``: no additional parameters required.
        - ``nlenses = 2``: ``s`` (separation) and ``q`` (mass ratio ``m2/m1``).
          Internally ``a = s / 2`` and ``e1 = q / (1 + q)``; the source is
          shifted to the center of mass for the polynomial construction.
        - ``nlenses = 3``: ``s``, ``q``, ``q3``, ``r3`` and ``psi``. Internally
          ``a = s / 2``, ``e1 = q / (1 + q + q3)``, ``e2 = 1 / (1 + q + q3)``,
          and the same center-of-mass shift is applied.

    Returns
    -------
    jax.Array
        Real-valued magnification with the same shape as ``w``.

    Notes
    -----
    Magnification is computed as the sum of ``|det J|^{-1}`` over valid image
    branches returned by ``_images_point_source``.
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
