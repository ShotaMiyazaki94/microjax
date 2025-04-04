from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap 
from microjax.point_source import _images_point_source
from microjax.multipole import _mag_hexadecapole


#from .extended_source import mag_uniform
@partial(jit, static_argnames=("nlenses"))
def test_full(w_points_shifted, rho, nlenses=2, **_params):
    q = _params["q"]
    if nlenses==2:
        z, z_mask = _images_point_source(w_points_shifted, nlenses=nlenses, **_params)
        # Compute hexadecapole approximation at every point and a test where it is sufficient
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(w_points_shifted, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params)
        test2 = _planetary_caustic_test(w_points_shifted, rho, **_params)
        test = jnp.where(q < 0.01, test1 & test2, test1)
        #not_full = lax.cond(q < 0.01, lambda:test1 & test2, lambda:test1,)
    else:
        test = jnp.zeros_like(w_points_shifted).astype(jnp.bool_)
    # test==True means no needs for inverse-ray shooting
    return test

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