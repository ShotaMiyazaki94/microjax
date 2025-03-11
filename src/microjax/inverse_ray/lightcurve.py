# -*- coding: utf-8 -*-
"""
Computing the magnification of an extended source at an arbitrary
set of points in the source plane.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap 

#from .extended_source import mag_uniform
from microjax.inverse_ray.extended_source import mag_uniform, mag_binary
from microjax.point_source import _images_point_source
from microjax.multipole import _mag_hexadecapole
from microjax.utils import *


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


def mag_lc_vmap(w_points, rho, nlenses=2, batch_size=400,
                r_resolution=1000, th_resolution=4000, Nlimb=1000, u1=0.0, **params):
    if nlenses == 1:
        _params = {}
        x_cm = 0 # miyazaki
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q/(1.0 + q) 
        _params = {"a": a, "e1": e1, "q": q, "s": s}
        x_cm = a*(1.0 - q)/(1.0 + q)
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = 1.0 / (1.0 + q + q3) #miyazaki
        r3 = r3 * jnp.exp(1j * psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
        x_cm = a * (1.0 - q) / (1.0 + q)
    else:
        raise ValueError("nlenses must be <= 3")
    
    # compute quadrupole approximation at every point and a test where it is sufficient 
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    if nlenses==1:
        test = w_points > 2*rho
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) #miyazaki
    elif nlenses==2:
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params #miyazaki
        )
        test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

        test = lax.cond(q < 0.01, lambda:test1 & test2, lambda:test1)
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)
    
    mag_full = lambda w: mag_binary(w, rho, nlenses=nlenses, Nlimb=Nlimb, u1=u1, 
                                     r_resolution=r_resolution, th_resolution=th_resolution, **_params)
    mag_full_vmap = vmap(mag_full, in_axes=(0,))

    map_input = [test, mu_multi, w_points]
    result = lax.map(lambda xs: 
                     lax.cond(xs[0], 
                              lambda _: xs[1], 
                              lambda _: mag_full_vmap(xs[2]), 
                              None), 
                     map_input)
    return result

    #def batched_vmap(w_points, batch_size=400):
    #    results = []
    #    for i in range(0, len(w_points), batch_size):
    #        chunk = w_points[i:i + batch_size]
    #        results.append(vmap(mag_full)(chunk))
    #    return jnp.concatenate(results)
    #
    #return batched_vmap(w_points, batch_size=batch_size)



#@partial(jit,static_argnames=("nlenses","r_resolution", "th_resolution", "Nlimb", "u1"))
def mag_lc(w_points, rho, nlenses=2, r_resolution=500, th_resolution=500, Nlimb=2000, u1=0.0, **params):
    # set parameters for the lens system
    if nlenses == 1:
        _params = {}
        x_cm = 0 # miyazaki
    elif nlenses == 2:
        s, q = params["s"], params["q"]
        a = 0.5 * s
        e1 = q/(1.0 + q) 
        _params = {"a": a, "e1": e1, "q": q, "s": s}
        x_cm = a*(1.0 - q)/(1.0 + q)
    elif nlenses == 3:
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a = 0.5 * s
        e1 = q / (1.0 + q + q3)
        e2 = 1.0 / (1.0 + q + q3) #miyazaki
        r3 = r3 * jnp.exp(1j * psi)
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
        x_cm = a * (1.0 - q) / (1.0 + q)
    else:
        raise ValueError("nlenses must be <= 3")

    # compute quadrupole approximation at every point and a test where it is sufficient 
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    if nlenses==1:
        test = w_points > 2*rho
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) #miyazaki
    elif nlenses==2:
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params #miyazaki
        )
        test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

        test = lax.cond(q < 0.01, lambda:test1 & test2, lambda:test1)
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)

    mag_full = lambda w: mag_binary(w, rho, nlenses=nlenses, Nlimb=Nlimb, u1=u1, 
                                     r_resolution=r_resolution, th_resolution=th_resolution, **_params)
    #mag_full_jit = jit(mag_full) 
    #def mag_full_vmap(w_points, batch_size=500):
    #    results = []
    #    for i in range(0, len(w_points), batch_size):
    #        chunk = w_points[i:i + batch_size]
    #        results.append(vmap(mag_full)(chunk))
    #    return jnp.concatenate(results)
    map_input = [test, mu_multi, w_points] 
    return lax.map(lambda xs: 
                        lax.cond(xs[0], 
                                lambda _: xs[1], 
                                lambda _: mag_full(xs[2]), 
                                None), 
                            map_input)

@partial(jit,static_argnames=("nlenses","r_resolution", "th_resolution", "Nlimb", "cubic"))
def mag_lc_uniform(w_points, rho, nlenses=2, r_resolution=500, th_resolution=500, Nlimb=2000, cubic=True, **params):
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
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    if nlenses==1:
        test = w_points > 2*rho
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) #miyazaki
    elif nlenses==2:
        # Compute hexadecapole approximation at every point and a test where it is sufficient
        mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
        test1 = _caustics_proximity_test(
            w_points - x_cm, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params #miyazaki
        )
        test2 = _planetary_caustic_test(w_points - x_cm, rho, **_params)

        test = lax.cond(
            q < 0.01, 
            lambda:test1 & test2,
            lambda:test1,
        )
    elif nlenses == 3:
        test = jnp.zeros_like(w_points).astype(jnp.bool_)

    _params = {"q": q, "s": s} 
    mag_full = lambda w: mag_uniform(w, rho, 
                                     nlenses=nlenses, 
                                     r_resolution=r_resolution,
                                     th_resolution=th_resolution,
                                     Nlimb=Nlimb,
                                     cubic=cubic, 
                                     **_params)

    # Iterate over w_points and execute either the hexadecapole  approximation
    # or the full extended source calculation. `vmap` cannot be used here because
    # `lax.cond` executes both branches within vmap.
    #jnp.stack([mask_test, mu_approx,  w_points]).T,
    return lax.map(lambda xs: lax.cond(xs[0], lambda _: xs[1], mag_full, xs[2],),
                   [test, mu_multi, w_points])
