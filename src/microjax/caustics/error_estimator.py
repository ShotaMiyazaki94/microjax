# This file vendors pieces of the `microlux` error estimator so we can
# drive adaptive sampling with a gradient-aware metric.
# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp


def dot_product(a, b):
    return jnp.real(a) * jnp.real(b) + jnp.imag(a) * jnp.imag(b)


def basic_partial(z, theta, rho, q, s, caustic_crossing):
    """
    Analytic derivatives of the binary-lens equation used by microlux.

    Parameters
    ----------
    z : array_like
        Image positions with shape ``(n_samples, n_images)``.
    theta : array_like
        Limb angles matching the first dimension of ``z``.
    rho : float
        Source radius (Einstein units).
    q, s : float
        Mass ratio and separation of the binary lens.
    caustic_crossing : bool
        If True, also compute the derivative of the parabolic correction term
        (needed for the gradient error term ``e4``).

    Returns
    -------
    tuple of array_like
        ``(deXProde2X, de_z, de_deXPro_de2X)`` matching the shape of ``z``.
    """
    theta = theta if theta.ndim > 1 else theta[:, None]
    z_c = jnp.conj(z)
    parZetaConZ = 1 / (1 + q) * (1 / (z_c - s) ** 2 + q / z_c**2)
    par2ConZetaZ = -2 / (1 + q) * (1 / (z - s) ** 3 + q / z**3)
    de_zeta = 1j * rho * jnp.exp(1j * theta)
    detJ = 1 - jnp.abs(parZetaConZ) ** 2
    de_z = (de_zeta - parZetaConZ * jnp.conj(de_zeta)) / detJ
    deXProde2X = (rho**2 + jnp.imag(de_z**2 * de_zeta * par2ConZetaZ)) / detJ

    def get_de_deXPro_de2X(_):
        de2_zeta = -rho * jnp.exp(1j * theta)
        de2_zetaConj = -rho * jnp.exp(-1j * theta)
        par3ConZetaZ = 6 / (1 + q) * (1 / (z - s) ** 4 + q / (z) ** 4)
        de2_z = (
            de2_zeta
            - jnp.conj(par2ConZetaZ) * jnp.conj(de_z) ** 2
            - parZetaConZ * (de2_zetaConj - par2ConZetaZ * de_z**2)
        ) / detJ
        return (
            1
            / detJ**2
            * jnp.imag(
                detJ
                * (
                    de2_zeta * par2ConZetaZ * de_z**2
                    + de_zeta * par3ConZetaZ * de_z**3
                    + de_zeta * par2ConZetaZ * 2 * de_z * de2_z
                )
                + (
                    jnp.conj(par2ConZetaZ) * jnp.conj(de_z) * jnp.conj(parZetaConZ)
                    + parZetaConZ * par2ConZetaZ * de_z
                )
                * de_zeta
                * par2ConZetaZ
                * de_z**2
            )
        )

    de_deXPro_de2X = jax.lax.cond(
        caustic_crossing, get_de_deXPro_de2X, lambda _: jnp.zeros_like(deXProde2X), None
    )
    return deXProde2X, de_z, de_deXPro_de2X


def error_ordinary(deXProde2X, de_z, delta_theta, z, parity, de_deXPro_de2X):
    """
    Microlux Eq.18 error estimator for ordinary segments (no creation/destruction).

    Parameters mirror microlux.error_estimator.error_ordinary.
    Shapes:
        - deXProde2X, de_z, z, parity: (n_samples, n_images)
        - delta_theta: (n_samples-1,)
    Returns
    -------
    tuple (e_tot, dAp) each with shape (n_samples-1, n_images)
    """
    delta_theta = delta_theta if delta_theta.ndim > 1 else delta_theta[:, None]
    dAp_1 = 0.5 * (deXProde2X[:-1] + deXProde2X[1:]) * (delta_theta**2) / 12.0
    delta_theta_wave = jnp.abs(z[:-1] - z[1:]) ** 2 / jnp.abs(dot_product(de_z[:-1], de_z[1:]))

    dAp_v1 = dAp_1 * parity[:-1]
    dAp_v2 = (
        1
        / 12
        * (
            (z.real[1:] - z.real[:-1]) * (de_z.imag[1:] - de_z.imag[:-1])
            - (z.imag[1:] - z.imag[:-1]) * (de_z.real[1:] - de_z.real[:-1])
        )
        * delta_theta
        * parity[:-1]
    )
    dAp = 0.5 * (dAp_v1 + dAp_v2)

    e1 = 0.5 * jnp.abs(dAp_v1 - dAp_v2)
    e2 = 1.5 * jnp.abs(dAp_1 * (delta_theta_wave - delta_theta**2))
    e3 = 0.1 * jnp.abs(dAp) * delta_theta**2

    de_dAp = (
        (de_deXPro_de2X[:-1] - de_deXPro_de2X[1:])
        * (delta_theta**3)
        * parity[:-1]
        / 24.0
    )
    e4 = 0.1 * jnp.abs(de_dAp)

    e_tot = e1 + e2 + e3 + e4
    return e_tot, dAp
