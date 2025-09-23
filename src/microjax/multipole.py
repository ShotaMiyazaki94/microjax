# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Multipole approximations for finite-source magnifications."""

__all__ = ["mag_hexadecapole"]

import jax
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import gammaln

#@jax.checkpoint
def _mag_hexadecapole_cassan(W, rho, u1=0.0):
    """Evaluate Cassan-style point/quadrupole/hexadecapole contributions.

    Parameters
    ----------
    W : jax.Array
        Stack of complex coefficients ``[W2, W3, W4, W5, W6]`` evaluated at the
        image locations.
    rho : float
        Source radius in Einstein units.
    u1 : float, optional
        Linear limb-darkening coefficient. Defaults to ``0``.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Point-source magnification and the quadrupole/hexadecapole corrections
        for each image.
    """
    W2, W3, W4, W5, W6 = W

    # Gamma LD coefficients is related to u
    Gamma = 2 * u1 / (3.0 - u1)

    # Compute a(p-n, n) factors from Q(p-n, n)
    akl = lambda mu, Qkl: mu * (jnp.conjugate(Qkl) + jnp.conjugate(W2) * Qkl)

    # Monopole magnification
    mu0 = 1.0 / (1.0 - jnp.abs(W2) ** 2)

    # Order p = 1 (monopole)
    a10 = mu0 * (1.0 + jnp.conjugate(W2))
    a01 = mu0 * 1j * (1.0 - jnp.conjugate(W2))

    # Order p = 2
    Q20 = W3 * a10 * a10
    Q11 = W3 * a10 * a01
    Q02 = W3 * a01 * a01
    a20 = akl(mu0, Q20)
    a11 = akl(mu0, Q11)
    a02 = akl(mu0, Q02)

    # Order p = 3 (quadrupole)
    Q30 = W3 * 3.0 * a20 * a10 + W4 * a10 * a10 * a10
    Q21 = W3 * (2.0 * a11 * a10 + a01 * a20) + W4 * a01 * a10 * a10
    Q12 = W3 * (2.0 * a11 * a01 + a10 * a02) + W4 * a10 * a01 * a01
    Q03 = W3 * 3.0 * a02 * a01 + W4 * a01 * a01 * a01
    a30 = akl(mu0, Q30)
    a21 = akl(mu0, Q21)
    a12 = akl(mu0, Q12)
    a03 = akl(mu0, Q03)

    # Order p = 4
    Q40 = (
        W3 * (4.0 * a30 * a10 + 3.0 * a20 * a20)
        + W4 * 6.0 * a20 * a10 * a10
        + W5 * a10 * a10 * a10 * a10
    )
    Q31 = (
        W3 * (3.0 * a21 * a10 + 3.0 * a11 * a20 + a01 * a30)
        + W4 * 3.0 * (a11 * a10 * a10 + a01 * a20 * a10)
        + W5 * a01 * a10 * a10 * a10
    )
    Q22 = (
        W3 * (2.0 * a12 * a10 + 2.0 * a11 * a11 + a02 * a20 + 2.0 * a21 * a01)
        + W4 * (a02 * a10 * a10 + 4.0 * a11 * a01 * a10 + a01 * a01 * a20)
        + W5 * a01 * a01 * a10 * a10
    )
    Q13 = (
        W3 * (3.0 * a12 * a01 + 3.0 * a11 * a02 + a10 * a03)
        + W4 * 3.0 * (a11 * a01 * a01 + a10 * a02 * a01)
        + W5 * a10 * a01 * a01 * a01
    )
    Q04 = (
        W3 * (4.0 * a03 * a01 + 3.0 * a02 * a02)
        + W4 * 6.0 * a02 * a01 * a01
        + W5 * a01 * a01 * a01 * a01
    )
    a40 = akl(mu0, Q40)
    a31 = akl(mu0, Q31)
    a22 = akl(mu0, Q22)
    a13 = akl(mu0, Q13)
    a04 = akl(mu0, Q04)

    # Order p = 5 (hexadecapole)
    Q50 = (
        W3 * (5.0 * a40 * a10 + 10.0 * a20 * a30)
        + W4 * (10.0 * a30 * a10 * a10 + 15.0 * a20 * a20 * a10)
        + W5 * 10.0 * a20 * a10 * a10 * a10
        + W6 * a10 * a10 * a10 * a10 * a10
    )
    Q41 = (
        W3 * (4.0 * a31 * a10 + 4.0 * a11 * a30 + 6.0 * a21 * a20 + a01 * a40)
        + W4
        * (
            6.0 * a21 * a10 * a10
            + 12.0 * a11 * a20 * a10
            + 3.0 * a01 * a20 * a20
            + 4.0 * a01 * a10 * a30
        )
        + W5 * (4.0 * a11 * a10 * a10 * a10 + 6.0 * a01 * a10 * a10 * a20)
        + W6 * a01 * a10 * a10 * a10 * a10
    )
    Q32 = (
        W3
        * (
            3.0 * a22 * a10
            + 6.0 * a11 * a21
            + 3.0 * a12 * a20
            + a02 * a30
            + 2.0 * a31 * a01
        )
        + W4
        * (
            3.0 * a12 * a10 * a10
            + 6.0 * a11 * a11 * a10
            + 3.0 * a02 * a20 * a10
            + 6.0 * a01 * a11 * a20
            + 6.0 * a01 * a21 * a10
            + a01 * a01 * a30
        )
        + W5
        * (
            a02 * a10 * a10 * a10
            + 6.0 * a11 * a01 * a10 * a10
            + 3.0 * a01 * a01 * a10 * a20
        )
        + W6 * a01 * a01 * a10 * a10 * a10
    )
    Q23 = (
        W3
        * (
            3.0 * a22 * a01
            + 6.0 * a11 * a12
            + 3.0 * a21 * a02
            + a20 * a03
            + 2.0 * a13 * a10
        )
        + W4
        * (
            3.0 * a21 * a01 * a01
            + 6.0 * a11 * a11 * a01
            + 3.0 * a20 * a02 * a01
            + 6.0 * a10 * a11 * a02
            + 6.0 * a10 * a12 * a01
            + a10 * a10 * a03
        )
        + W5
        * (
            a20 * a01 * a01 * a01
            + 6.0 * a11 * a10 * a01 * a01
            + 3.0 * a10 * a10 * a01 * a02
        )
        + W6 * a10 * a10 * a01 * a01 * a01
    )
    Q14 = (
        W3 * (4.0 * a13 * a01 + 4.0 * a11 * a03 + 6.0 * a12 * a02 + a10 * a04)
        + W4
        * (
            6.0 * a12 * a01 * a01
            + 12.0 * a11 * a02 * a01
            + 3.0 * a10 * a02 * a02
            + 4.0 * a10 * a01 * a03
        )
        + W5 * (4.0 * a11 * a01 * a01 * a01 + 6.0 * a10 * a01 * a01 * a02)
        + W6 * a10 * a01 * a01 * a01 * a01
    )
    Q05 = (
        W3 * (5.0 * a04 * a01 + 10.0 * a02 * a03)
        + W4 * (10.0 * a03 * a01 * a01 + 15.0 * a02 * a02 * a01)
        + W5 * 10.0 * a02 * a01 * a01 * a01
        + W6 * a01 * a01 * a01 * a01 * a01
    )
    a50 = akl(mu0, Q50)
    a41 = akl(mu0, Q41)
    a32 = akl(mu0, Q32)
    a23 = akl(mu0, Q23)
    a14 = akl(mu0, Q14)
    a05 = akl(mu0, Q05)

    # Compute hexadecapole and quadrupole mu and A
    mu2 = (
        1.0
        / 4.0
        * jnp.imag(
            a01 * jnp.conjugate(a12 + a30)
            + jnp.conjugate(a10) * (a03 + a21)
            + 2.0 * a02 * jnp.conjugate(a11)
            + 2.0 * a11 * jnp.conjugate(a20)
        )
    )
    mu4 = (
        1.0
        / 8.0
        * jnp.imag(
            (a05 + 2.0 * a23 + a41) * jnp.conjugate(a10)
            + 4.0 * a04 * jnp.conjugate(a11)
            + 4 * (a13 + a31) * jnp.conjugate(a20)
            + 6.0 * a12 * jnp.conjugate(a21)
            + 6.0 * a21 * jnp.conjugate(a30)
            + a03 * (6.0 * jnp.conjugate(a12) + 2.0 * jnp.conjugate(a30))
            + 4.0 * a02 * jnp.conjugate(a13 + a31)
            + 4.0 * a11 * jnp.conjugate(a40)
            + a01 * jnp.conjugate(a14 + 2.0 * a32 + a50)
        )
    )

    mu_ps = mu0
    delta_mu_quad = 1.0 / 2.0 * mu2 * (1.0 - 1.0 / 5.0 * Gamma) * rho**2
    delta_mu_hex  = 1.0 / 24.0 * mu4 * (1.0 - 11.0 / 35.0 * Gamma) * rho**4
    
    return mu_ps, delta_mu_quad, delta_mu_hex

def mag_hexadecapole(
    z: jax.Array,
    z_mask: jax.Array,
    rho: float,
    u1: float = 0.0,
    nlenses: int = 2,
    **params,
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate the hexadecapole approximation for finite-source magnification.

    Parameters
    ----------
    z : jax.Array
        Complex image locations with shape ``(N_images, ...)``.
    z_mask : jax.Array
        Boolean mask with the same shape as ``z`` selecting valid image
        branches (typically output of :func:`microjax.point_source._images_point_source`).
    rho : float
        Source radius in Einstein units.
    u1 : float, optional
        Linear limb-darkening coefficient. Defaults to ``0`` (uniform source).
    nlenses : int, optional
        Number of lenses (1, 2, or 3). Defaults to 2.
    **params
        Additional lens parameters in the mid-point coordinate system. For
        binaries this expects ``a`` and ``e1``; for triples ``a``, ``r3``,
        ``e1`` and ``e2`` in addition to any other configuration details.

    Returns
    -------
    mu : jax.Array
        Finite-source magnification obtained by summing the point-source term
        with quadrupole and hexadecapole corrections over all valid images.
    residual : jax.Array
        Diagnostic amplitude (absolute quadrupole + hexadecapole contribution)
        that can be used to gauge when the approximation may be insufficient.

    Notes
    -----
    The implementation follows Cassan et al. (2017), computing the multipole
    moments via the complex quantities ``W_k`` and applying limb-darkening
    corrections through the ``Gamma`` parameter.
    """

    factorial = lambda n: jnp.exp(gammaln(n + 1))

    if nlenses == 1:
        W = lambda k: (-1) ** (k - 1) * factorial(k - 1) * 1.0 / z**k
    elif nlenses == 2:
        a, e1 = params["a"], params["e1"]
        W = (
            lambda k: (-1) ** (k - 1)
            * factorial(k - 1)
            * (e1 / (z - a) ** k + (1.0 - e1) / (z + a) ** k)
        )
    elif nlenses == 3:
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        W = (
            lambda k: (-1) ** (k - 1)
            * factorial(k - 1)
            * (
                e1 / (z - a) ** k
                + e2 / (z + a) ** k
                + (1.0 - e1 - e2) / (z - r3) ** k
            )
        )
    else:
        raise ValueError("`nlenses` has to be set to be <= 3.")

    Ws = vmap(W)(jnp.arange(2, 7))

    mu_ps, delta_mu_quad, delta_mu_hex = jax.checkpoint(
        lambda Ws_: _mag_hexadecapole_cassan(Ws_, rho, u1=u1)
    )(Ws)

    mu_multi = jnp.sum(z_mask * jnp.abs(mu_ps + delta_mu_quad + delta_mu_hex), axis=0)
    mu_quad_abs = jnp.sum(z_mask * jnp.abs(delta_mu_quad), axis=0)
    mu_hex_abs = jnp.sum(z_mask * jnp.abs(delta_mu_hex), axis=0)

    return mu_multi, mu_quad_abs + mu_hex_abs


# Backwards compatibility alias (to be removed in a future release)
_mag_hexadecapole = mag_hexadecapole
