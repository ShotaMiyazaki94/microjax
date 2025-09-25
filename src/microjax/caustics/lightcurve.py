# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Hybrid light-curve evaluation adapted from the `caustics` package.

This module retains the contour-selection heuristics of Fran Bartolic's
``caustics`` project while porting the implementation to JAX and the
microJAX coordinate conventions.  The main entry point,
:func:`magnifications`, mixes the fast hexadecapole approximation with full
finite-source contour integrations to deliver accurate light curves across
binary and triple microlensing configurations.

Design highlights
-----------------

- **JAX-native execution**: relies on :mod:`jax`, :mod:`jax.numpy`, and
  :func:`jax.jit` to make the control flow differentiable and accelerator
  friendly.
- **Center-of-mass bookkeeping**: transparently shifts source positions to the
  midpoint frame required by the polynomial image solver and restores them
  before returning magnifications.
- **Selective refinement**: reuses the proximity and ghost-image tests from
  ``caustics`` to decide when the multipole estimate suffices.

External dependencies
---------------------

- :mod:`jax` (``jax.numpy`` arrays and ``jit``/``lax`` primitives).
- :mod:`functools.partial` for annotating static JIT parameters.

Internal collaborators
----------------------

- :mod:`microjax.multipole` — hexadecapole approximation.
- :mod:`microjax.caustics.extended_source` — contour integration routines.
- :mod:`microjax.point_source` — polynomial image finder used in the accuracy
  tests.
"""

__all__ = [
    "magnifications",
]

from functools import partial

import jax.numpy as jnp
from jax import jit, lax 

from .extended_source import mag_extended_source
from ..point_source import _images_point_source

from ..multipole import mag_hexadecapole

from ..utils import *


@partial(jit, static_argnames=("nlenses"))
def _caustics_proximity_test(
    w, z, z_mask, rho, delta_mu_multi, nlenses=2, c_m=1e-02, gamma=0.02, c_f=4., rho_min=1e-03, **params
):
    if nlenses == 2:
        a, e1 = params["a"], params["e1"]
        e2 = 1.0 - e1
        # Derivatives
        f = lambda z: - e1 / (z - a) - e2 / (z + a)
        f_p = lambda z: e1 / (z - a) ** 2 + e2 / (z + a) ** 2
        f_pp = lambda z: 2 * (e1 / (a - z) ** 3 - e2 / (a + z) ** 3)

    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        # Derivatives
        f = lambda z: - e1 / (z - a) - e2 / (z + a) - (1 - e1 - e2) / (z + r3)
        f_p = (
            lambda z: e1 / (z - a) ** 2
            + e2 / (z + a) ** 2
            + (1 - e1 - e2) / (z + r3) ** 2
        )
        f_pp = (
            lambda z: 2 * (e1 / (a - z) ** 3 - e2 / (a + z) ** 3)
            + (1 - e1 - e2) / (z + r3) ** 3
        )
    zbar = jnp.conjugate(z)
    zhat = jnp.conjugate(w) - f(z)

    # Derivatives
    fp_z     = f_p(z)
    fpp_z    = f_pp(z)
    fp_zbar  = f_p(zbar)
    fp_zhat  = f_p(zhat)
    fpp_zbar = f_pp(zbar)
    J        = 1.0 - jnp.abs(fp_z * fp_zbar)

    # Multipole test and cusp test
    mu_cusp = 6 * jnp.imag(3 * fp_zbar**3.0 * fpp_z**2.0) / J**5 * (rho + rho_min)**2
    mu_cusp = jnp.sum(jnp.abs(mu_cusp) * z_mask, axis=0)
    test_multipole_and_cusp = gamma * mu_cusp + delta_mu_multi < c_m

    # False images test
    Jhat = 1 - jnp.abs(fp_z * fp_zhat)
    factor = jnp.abs(J * Jhat**2 / 
                     (Jhat*fpp_zbar*fp_z - jnp.conjugate(Jhat)  * fpp_z * fp_zbar * fp_zhat)
                     )
    test_false_images = 0.5 * (~z_mask * factor).sum(axis=0) > c_f * (rho + rho_min)
    test_false_images = jnp.where((~z_mask).sum(axis=0)==0, 
                                  jnp.ones_like(test_false_images, dtype=jnp.bool_), 
                                  test_false_images
                                  )
    return test_false_images & test_multipole_and_cusp


def _planetary_caustic_test(w, rho, c_p=2., **params):
    e1, a = params["e1"], params["a"]
    s = 2 * a
    q = e1 / (1.0 - e1)
    x_cm = (2*e1 - 1)*a
    w_pc = -1/s 
    delta_pc = 3*jnp.sqrt(q)/s
    return (w_pc - w).real**2 + (w_pc - w).imag**2 > c_p*(rho**2 + delta_pc**2)


@partial(
    jit,
    static_argnames=(
        "nlenses",
        "npts_limb",
        "limb_darkening",
        "npts_ld",
    ),
)
def magnifications(
    w_points,
    rho,
    nlenses=2,
    npts_limb=200,
    limb_darkening=False,
    u1=0.0,
    npts_ld=100,
    **params
):
    """Finite-source magnification samples along a caustic light curve.

    The routine evaluates the magnification for each complex source position in
    ``w_points`` by reusing the ``caustics`` decision logic, adapted to
    microJAX's center-of-mass conventions and JAX-native solvers.  Depending on
    the local configuration it either applies
    :func:`microjax.multipole.mag_hexadecapole` or falls back to the full
    contour-integration path implemented in
    :func:`microjax.caustics.extended_source.mag_extended_source`.

    Parameters
    ----------
    w_points : jax.Array
        Source-plane positions expressed in the center-of-mass frame of the
        first two lenses (or the lone lens when ``nlenses == 1``).
    rho : float
        Angular radius of the source in Einstein units.
    nlenses : int, optional
        Number of point-mass lenses (1, 2, or 3). Defaults to 2.
    npts_limb : int, optional
        Baseline number of uniformly spaced samples placed on the source limb
        before adaptive refinement during contour integrations. Defaults to
        200.
    limb_darkening : bool, optional
        When ``True``, evaluate linear limb-darkening with coefficient ``u1``.
        Defaults to ``False``.
    u1 : float, optional
        Linear limb-darkening coefficient used when ``limb_darkening`` is
        enabled. Defaults to ``0.0``.
    npts_ld : int, optional
        Number of quadrature nodes for the Dominik (1998) ``P``/``Q`` integrals
        used in the limb-darkened branch. Defaults to 100.
    **params : Any
        Lens parameters forwarded to the multipole and contour solvers.  The
        expected keywords depend on ``nlenses``:

        - ``nlenses == 1``: none required.
        - ``nlenses == 2``: ``s`` (separation) and ``q`` (mass ratio ``m2/m1``).
        - ``nlenses == 3``: ``s``, ``q``, ``q3`` (``m3/m1``), ``r3`` (modulus of
          the third lens position), and ``psi`` (azimuth of the third lens).

        Additional keyword arguments recognised by
        :func:`microjax.multipole.mag_hexadecapole`,
        :func:`microjax.caustics.extended_source.mag_extended_source`, or
        :func:`microjax.point_source._images_point_source` (for example custom
        root-solver settings) are propagated unchanged.

    Returns
    -------
    jax.Array
        Magnification evaluated at each entry of ``w_points``.

    Notes
    -----
    The routine temporarily shifts coordinates into the midpoint frame required
    by the polynomial root solver and shifts the images back before applying
    the multipole or contour integrations.  For ``nlenses == 1`` the multipole
    approximation is exact; for ``nlenses == 2`` proximity and planetary tests
    decide whether it is used; for ``nlenses == 3`` the algorithm always falls
    back to contour integration because the ghost-image test has not yet been
    generalised.  Enabling limb darkening can increase runtime by up to an
    order of magnitude due to the extra quadrature.
    """
    if nlenses == 1:
        _params = {}
        x_cm = 0 # miyazaki
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q / (1.0 + q) 
        _params = {"a": a, "e1": e1}
        x_cm = a*(1 - q)/(1 + q)

    # Trigger the full calculation everywhere because I haven't figured out 
    # how to implement the ghost image test for nlenses > 2 yet
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = 1.0 / (1.0 + q + q3) #miyazaki
        r3 = r3*jnp.exp(1j * psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2}
        x_cm = a * (1.0 - q) / (1.0 + q)

    else:
        raise ValueError("nlenses must be <= 3")


    # Compute point images for a point source
    z, z_mask = _images_point_source(
        w_points - x_cm, #miyazaki
        #w_points + x_cm,
        nlenses=nlenses,
        **_params
    )

    if nlenses==1:
        test = w_points > 2*rho
        mu_multi, delta_mu_multi = mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)  # miyazaki
    elif nlenses==2:
        # Compute hexadecapole approximation at every point and a test where it is
        # sufficient
        mu_multi, delta_mu_multi = mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params #miyazaki
            #w_points + x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params
        )
        test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)
        #test2 = _planetary_caustic_test(w_points + x_cm, rho, **_params)

        test = lax.cond(
            q < 0.01, 
            lambda:test1 & test2,
            lambda:test1,
        )
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)
    
    mag_full = lambda w: mag_extended_source(
        w,
        rho,
        nlenses=nlenses,
        npts_limb=npts_limb,
        limb_darkening=limb_darkening,
        u1=u1,
        npts_ld=npts_ld,
        **params,
    )

    # Iterate over w_points and execute either the hexadecapole  approximation
    # or the full extended source calculation. `vmap` cannot be used here because
    # `lax.cond` executes both branches within vmap.
    return lax.map(
        lambda xs: lax.cond(
            xs[0],
            lambda _: xs[1],
            mag_full,
            xs[2],
        ),
        [test, mu_multi, w_points],
        #        jnp.stack([mask_test, mu_approx,  w_points]).T,
    )
