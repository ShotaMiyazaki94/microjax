"""Adaptive light-curve computation combining multipole and inverse-ray solvers.

This module mixes the fast hexadecapole approximation with selective inverse-ray
finite-source integrations to deliver accurate binary- and triple-lens light
curves while keeping GPU/CPU cost manageable.  The key routines exported here
are :func:`mag_binary` and :func:`mag_triple`.

Design highlights
-----------------

- **Fast-first evaluation**: start from the hexadecapole estimate everywhere
  and upgrade only the samples that fail accuracy heuristics.
- **Hybrid triggers**: combine caustic-proximity and planetary-caustic tests to
  decide when a full inverse-ray solve is required.
- **Chunked batching**: evaluate expensive inverse-ray calls in configurable
  chunks to balance memory usage and accelerator occupancy.
- **Limb-darkening aware**: switch seamlessly between uniform and linear
  limb-darkened profiles by toggling the ``u1`` parameter.
- **Shared infrastructure**: binary and triple lenses reuse the same chunking
  and integration utilities, so configuration knobs behave consistently.

Workflow outline
----------------

1. Build a complex source-plane trajectory ``w_points``.
2. Call :func:`mag_binary` or :func:`mag_triple` with lens parameters and
   integration settings.
3. Feed the returned magnifications into downstream likelihoods (see
   :mod:`microjax.likelihood`).

References
----------

- Miyazaki & Kawahara (in prep.) — description of the adaptive microJAX
  solver stack (forthcoming).
"""

__all__ = [
    "mag_binary",
    "mag_triple",
]

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from microjax.inverse_ray.cond_extended import (
    _caustics_proximity_test,
    _planetary_caustic_test,
    test_full,
)
from microjax.inverse_ray.extended_source import mag_limb_dark, mag_uniform
from microjax.multipole import mag_hexadecapole
from microjax.point_source import _images_point_source

# Consistent array alias used across modules
Array = jnp.ndarray

@partial(jit,static_argnames=("r_resolution", "th_resolution", "u1", "delta_c", 
                              "bins_r", "bins_th", "margin_r", "margin_th", 
                              "Nlimb", "MAX_FULL_CALLS", "chunk_size"))
def mag_binary(
    w_points: Array,
    rho: float,
    r_resolution: int = 1000,
    th_resolution: int = 1000,
    u1: float = 0.0,
    delta_c: float = 0.01,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 1.0,
    margin_th: float = 1.0,
    MAX_FULL_CALLS: int = 500,
    chunk_size: int = 100,
    **params,
) -> Array:
    """Compute a binary-lens light curve with adaptive inverse-ray fallback.

    The routine evaluates the fast hexadecapole approximation everywhere and
    selectively replaces samples with full inverse-ray finite-source results
    when accuracy tests trigger.  All arguments mirror the lower-level
    integrators in :mod:`microjax.inverse_ray.extended_source`.

    Parameters
    ----------
    w_points : Array
        Complex array of source-plane coordinates (``x + 1j*y``) along the
        trajectory.  The trailing shape is preserved in the output.
    rho : float
        Angular source radius in Einstein units.
    r_resolution : int, optional
        Number of radial quadrature panels for the inverse-ray solver.
    th_resolution : int, optional
        Number of azimuthal panels for the inverse-ray solver.
    u1 : float, optional
        Linear limb-darkening coefficient. Set to ``0`` for a uniform source.
    delta_c : float, optional
        Radial contraction factor used by the limb-darkened integrator.
    Nlimb : int, optional
        Number of rings in the limb-darkening interpolation table.
    bins_r : int, optional
        Radial bin count for the polar mesh acceleration structure.
    bins_th : int, optional
        Azimuthal bin count for the polar mesh acceleration structure.
    margin_r : float, optional
        Additional margin (in source radii) added to the radial domain.
    margin_th : float, optional
        Additional margin (in radians) added to the azimuthal domain.
    MAX_FULL_CALLS : int, optional
        Maximum number of samples upgraded to the full inverse-ray solve.
    chunk_size : int, optional
        Batch size fed to ``vmap`` during inverse-ray evaluation.
    **params
        Lens configuration parameters.  ``s`` (separation) and ``q`` (mass
        ratio) are required, and any additional keyword arguments are forwarded
        to the integrators.

    Returns
    -------
    Array
        Magnification values with the same shape as ``w_points``.

    Notes
    -----
    The adaptive trigger combines a caustic proximity test with an additional
    planetary-caustic guard for very small mass ratios.  When a trigger fires,
    the corresponding sample is recomputed with :func:`mag_uniform` or
    :func:`mag_limb_dark`, depending on ``u1``.  The returned array is lazily
    evaluated; call ``.block_until_ready()`` if you need synchronisation in a
    JAX-asynchronous context.
    """
    s = params.get("s", None)
    q = params.get("q", None)
    if s is None or q is None:
        raise ValueError("For nlenses=2, 's' and 'q' must be provided.") 
    
    nlenses = 2
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {**params, "a": a, "e1": e1}
    x_cm = a * (1.0 - q) / (1.0 + q)
    w_points_shifted = w_points - x_cm

    # test whether inverse-ray shooting needed or not. test==False means needed.
    z, z_mask = _images_point_source(w_points_shifted, nlenses=nlenses, a=a, e1=e1)
    mu_multi, delta_mu_multi = mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    test1 = _caustics_proximity_test(w_points_shifted, z, z_mask, rho, delta_mu_multi, nlenses=nlenses,  **_params)
    test2 = _planetary_caustic_test(w_points_shifted, rho, **_params)
    test = jnp.where(q < 0.01, test1 & test2, test1)

    if u1 == 0.0:
        def _mag_full(w):
            return mag_uniform(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution = th_resolution,
                               bins_r = bins_r, bins_th = bins_th, margin_r = margin_r, margin_th = margin_th, 
                               Nlimb = Nlimb, **_params)
    else:
        def _mag_full(w):
            return mag_limb_dark(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution= th_resolution,
                                 u1 = u1, delta_c = delta_c, bins_r = bins_r, bins_th = bins_th, margin_r = margin_r,
                                 margin_th = margin_th, Nlimb = Nlimb, **_params)
    
    idx_sorted = jnp.argsort(test)
    idx_full = idx_sorted[:MAX_FULL_CALLS]

    def chunked_vmap(func, data, chunk_size):
        N = data.shape[0]
        pad_len = (-N) % chunk_size
        chunks = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:])
        return lax.map(lambda c: vmap(func)(c), chunks).reshape(-1, *data.shape[2:])[:N]
    
    def chunked_vmap_scan(func, data, chunk_size):
        N = data.shape[0]
        pad_len  = (-N) % chunk_size
        chunks   = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:]) 
        #@jax.checkpoint
        def body(carry, chunk):
            #out = vmap(jax.checkpoint(func))(chunk)             
            out = vmap(func)(chunk)                            
            return carry, out               
        _, outs = lax.scan(body, None, chunks)   # outs → (T, chunk_size, ...)
        return outs.reshape(-1, *data.shape[2:])[:N]

    _mag_full = jax.checkpoint(_mag_full, policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
    mag_extended = chunked_vmap(_mag_full, w_points[idx_full], chunk_size)
    #mag_extended = chunked_vmap_scan(_mag_full, w_points[idx_full], chunk_size)
    mags = mu_multi.at[idx_full].set(mag_extended)
    mags = jnp.where(test, mu_multi, mags)
    return mags 

@partial(jit,static_argnames=("r_resolution", "th_resolution", "u1", "delta_c",
                              "bins_r", "bins_th", "margin_r", "margin_th", 
                              "Nlimb", "MAX_FULL_CALLS", "chunk_size"))
def mag_triple(
    w_points: Array,
    rho: float,
    r_resolution: int = 1000,
    th_resolution: int = 1000,
    u1: float = 0.0,
    delta_c: float = 0.01,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 1.0,
    margin_th: float = 1.0,
    MAX_FULL_CALLS: int = 500,
    chunk_size: int = 50,
    **params,
) -> Array:
    """Compute a triple-lens light curve with optional inverse-ray fallback.

    The structure mirrors :func:`mag_binary`, but the trigger logic currently
    upgrades only the first ``MAX_FULL_CALLS`` samples (the caustic tests have
    not yet been specialised for triple lenses).

    Parameters
    ----------
    w_points : Array
        Complex array of source-plane coordinates (``x + 1j*y``).
    rho : float
        Angular source radius in Einstein units.
    r_resolution : int, optional
        Number of radial quadrature panels for the inverse-ray solver.
    th_resolution : int, optional
        Number of azimuthal panels for the inverse-ray solver.
    u1 : float, optional
        Linear limb-darkening coefficient (``0`` selects a uniform source).
    delta_c : float, optional
        Radial contraction factor for the limb-darkened integrator.
    Nlimb : int, optional
        Number of rings in the limb-darkening interpolation table.
    bins_r : int, optional
        Radial bin count for the polar mesh acceleration structure.
    bins_th : int, optional
        Azimuthal bin count for the polar mesh acceleration structure.
    margin_r : float, optional
        Additional radial margin (in source radii) for the integration domain.
    margin_th : float, optional
        Additional azimuthal margin (in radians) for the integration domain.
    MAX_FULL_CALLS : int, optional
        Maximum number of samples that receive the inverse-ray upgrade.
    chunk_size : int, optional
        Batch size fed to ``vmap`` during inverse-ray evaluation.
    **params
        Triple-lens configuration parameters.  ``s``, ``q``, ``q3``, ``r3`` and
        ``psi`` must be supplied; any additional keyword arguments are forwarded
        to the integrators.

    Returns
    -------
    Array
        Magnification values with the same shape as ``w_points``.

    Notes
    -----
    The method always evaluates the hexadecapole approximation and patches in
    full inverse-ray results for the selected samples using
    :func:`mag_uniform` or :func:`mag_limb_dark` depending on ``u1``.  Triple
    lenses near cusps or resonant caustics may need higher ``MAX_FULL_CALLS``
    to reach convergence.
    """
    nlenses = 3
    s, q, q3 = params["s"], params["q"], params["q3"]
    a = 0.5 * s
    e1 = q / (1.0 + q + q3) 
    e2 = 1.0/(1.0 + q + q3)
    #r3 = r3 * jnp.exp(1j * psi)
    #_params = {"a": a, "r3": r3, "e1": e1, "e2": e2}
    _params = {**params, "a": a, "e1": e1, "e2": e2}
    #_params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
    x_cm = a * (1.0 - q) / (1.0 + q)
    w_points_shifted = w_points - x_cm
    
    z, z_mask = _images_point_source(w_points_shifted, nlenses=nlenses, **_params) 
    mu_multi, delta_mu_multi = mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    test = jnp.zeros_like(w_points).astype(jnp.bool_)

    if u1 == 0.0:
        def _mag_full(w):
            return mag_uniform(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution = th_resolution,
                               bins_r = bins_r, bins_th = bins_th, margin_r = margin_r, margin_th = margin_th, 
                               Nlimb = Nlimb, **_params)
    else:
        def _mag_full(w):
            return mag_limb_dark(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution= th_resolution,
                                 u1 = u1, delta_c = delta_c, bins_r = bins_r, bins_th = bins_th, margin_r = margin_r,
                                 margin_th = margin_th, Nlimb = Nlimb, **_params)

    idx_sorted = jnp.argsort(test)
    idx_full = idx_sorted[:MAX_FULL_CALLS]

    def chunked_vmap(func, data, chunk_size):
        N = data.shape[0]
        pad_len = (-N) % chunk_size
        chunks = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:])
        return lax.map(lambda c: vmap(func)(c), chunks).reshape(-1, *data.shape[2:])[:N]

    _mag_full = jax.checkpoint(_mag_full)
    mag_extended = chunked_vmap(_mag_full, w_points[idx_full], chunk_size)
    mags = mu_multi.at[idx_full].set(mag_extended)
    mags = jnp.where(test, mu_multi, mags)
    return mags 
    
