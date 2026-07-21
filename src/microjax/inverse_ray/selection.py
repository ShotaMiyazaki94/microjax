"""Select the multipole approximation or full boundary integration.

This module provides tests that determine whether the hexadecapole approximation
is sufficient for a given source position or whether to fall back to the full
inverse-ray integration. The logic is designed for binary and triple lenses.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

# Consistent array alias used across inverse_ray and trajectory modules
Array = jnp.ndarray


@partial(jit, static_argnames=("nlenses"))
def _caustics_proximity_test(
    w: Array,
    z: Array,
    z_mask: Array,
    rho: float,
    delta_mu_multi: Array,
    nlenses: int = 2,
    c_m: float = 1e-02,
    gamma: float = 0.02,
    c_f: float = 4.0,
    rho_min: float = 1e-03,
    **params,
) -> Array:
    """Multipole accuracy and cusp proximity tests near caustics.

    Combines a magnitude threshold on the hexadecapole correction and a cusp
    proximity metric. Also checks for false images and filters them with a
    scale set by ``c_f``.
    """
    if nlenses == 2:
        a, e1 = params["a"], params["e1"]
        lens_positions = jnp.asarray([a, -a])
        lens_masses = jnp.asarray([e1, 1.0 - e1])

    elif nlenses == 3:
        a, e1, e2 = params["a"], params["e1"], params["e2"]
        if "r3_complex" in params:
            r3_complex = params["r3_complex"]
        else:
            r3_complex = params["r3"] * jnp.exp(1j * params["psi"])
        lens_positions = jnp.asarray([a, -a, r3_complex])
        lens_masses = jnp.asarray([e1, e2, 1.0 - e1 - e2])
    else:
        raise ValueError("nlenses must be 2 or 3")

    def derivative_sum(value, power):
        expansion = (1,) * value.ndim
        positions = lens_positions.reshape((-1,) + expansion)
        masses = lens_masses.reshape((-1,) + expansion)
        return jnp.sum(
            masses / (value[None, ...] - positions) ** power,
            axis=0,
        )

    def f(value):
        return -derivative_sum(value, 1)

    def f_p(value):
        return derivative_sum(value, 2)

    def f_pp(value):
        return -2.0 * derivative_sum(value, 3)

    zbar = jnp.conjugate(z)
    zhat = jnp.conjugate(w) - f(z)

    # Derivatives
    fp_z = f_p(z)
    fpp_z = f_pp(z)
    fp_zbar = f_p(zbar)
    fp_zhat = f_p(zhat)
    fpp_zbar = f_pp(zbar)
    J = 1.0 - jnp.abs(fp_z * fp_zbar)

    # Multipole test and cusp test
    mu_cusp = 6 * jnp.imag(3 * fp_zbar**3.0 * fpp_z**2.0) / J**5 * (rho + rho_min) ** 2
    mu_cusp = jnp.sum(jnp.abs(mu_cusp) * z_mask, axis=0)
    test_multipole_and_cusp = gamma * mu_cusp + delta_mu_multi < c_m

    # False images test
    Jhat = 1 - jnp.abs(fp_z * fp_zhat)
    factor = jnp.abs(J * Jhat**2 / (Jhat * fpp_zbar * fp_z - jnp.conjugate(Jhat) * fpp_z * fp_zbar * fp_zhat))
    n_false = (~z_mask).sum(axis=0)
    if nlenses == 2:
        # A binary polynomial has one conjugate ghost pair, so the historical
        # half-sum estimates that pair's caustic distance.
        false_image_scale = 0.5 * (~z_mask * factor).sum(axis=0)
    else:
        # A triple polynomial can carry several unrelated ghost pairs. Summing
        # all six false-root scales lets distant pairs hide the one that is
        # about to become physical. Every pair must therefore clear the guard.
        false_image_scale = jnp.min(jnp.where(~z_mask, factor, jnp.inf), axis=0)
    test_false_images = false_image_scale > c_f * (rho + rho_min)
    test_false_images = jnp.where(
        n_false == 0,
        jnp.ones_like(test_false_images, dtype=jnp.bool_),
        test_false_images,
    )
    return test_false_images & test_multipole_and_cusp


def _planetary_caustic_test(w: Array, rho: float, c_p: float = 2.0, **params) -> Array:
    """Exclude regions too close to planetary caustics for small mass ratios."""
    e1, a = params["e1"], params["a"]
    s = 2 * a
    q = e1 / (1.0 - e1)
    w_pc = -1 / s
    delta_pc = 3 * jnp.sqrt(q) / s
    return (w_pc - w).real ** 2 + (w_pc - w).imag ** 2 > c_p * (rho**2 + delta_pc**2)
