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
    """Binary-lens finite-source magnification with adaptive inverse-ray fallback.

    The routine expects a 1-D complex trajectory ``w_points`` (Einstein-radius
    units). Internally it (i) derives the centre-of-mass shift from the supplied
    separation ``s`` and mass ratio ``q`` and recentres ``w_points``; (ii)
    evaluates the point-source images with
    :func:`microjax.point_source._images_point_source` and the corresponding
    hexadecapole estimate ``mu_multi`` via
    :func:`microjax.multipole.mag_hexadecapole`; and (iii) applies
    :func:`microjax.inverse_ray.cond_extended._caustics_proximity_test`—and, for
    ``q < 0.01``, the additional
    :func:`microjax.inverse_ray.cond_extended._planetary_caustic_test`—to form a
    boolean mask ``test`` where ``True`` means the multipole value is trusted.
    The indices are sorted so that ``False`` entries precede ``True`` ones, up to
    ``MAX_FULL_CALLS`` of those ``False`` entries are recomputed using
    :func:`microjax.inverse_ray.extended_source.mag_uniform` (``u1 == 0``) or
    :func:`microjax.inverse_ray.extended_source.mag_limb_dark`, and the selected
    samples are spliced back into ``mu_multi``. Any remaining ``False`` entries
    beyond the ``MAX_FULL_CALLS`` budget stay on the hexadecapole baseline.

    Parameters
    ----------
    w_points : Array
        One-dimensional complex ``jax.Array`` of source-plane coordinates
        (``x + 1j*y``) sampled along the trajectory. The returned
        magnifications track this ordering element by element.
    rho : float
        Angular source radius in Einstein units.
    r_resolution : int, optional
        Number of uniformly spaced radial samples per polar cell used by the
        inverse-ray integrator (``>= 2``).
    th_resolution : int, optional
        Number of uniformly spaced angular samples per polar cell used by the
        inverse-ray integrator (``>= 2``).
    u1 : float, optional
        Linear limb-darkening coefficient. Use ``0`` for a uniform surface
        brightness.
    delta_c : float, optional
        Dimensionless smoothing threshold supplied to
        :func:`microjax.inverse_ray.boundary.calc_facB` in the limb-darkened
        integrator.
    Nlimb : int, optional
        Number of source-limb samples traced through the lens to seed the polar
        region construction.
    bins_r : int, optional
        Number of histogram bins used when clustering limb radii into polar
        subregions (higher values capture finer radial structure).
    bins_th : int, optional
        Number of histogram bins used when clustering limb angles into polar
        subregions.
    margin_r : float, optional
        Radial margin applied to each subregion in units of ``rho``.
    margin_th : float, optional
        Angular margin applied to each subregion, expressed in degrees (converted
        to radians internally).
    MAX_FULL_CALLS : int, optional
        Maximum number of samples replaced by the inverse-ray finite-source
        solver. Set to ``0`` to disable the fallback.
    chunk_size : int, optional
        Number of upgraded samples evaluated per :func:`jax.vmap` batch when the
        inverse-ray solver is invoked.
    **params
        Lens-configuration keywords forwarded to the point-source and
        inverse-ray solvers. ``s`` (projected separation) and ``q``
        (companion-to-host mass ratio) are mandatory; any additional entries
        understood by the low-level routines are forwarded unchanged.

    Returns
    -------
    Array
        Real-valued magnification array with the same shape as ``w_points``.

    Notes
    -----
    - ``test`` is ``True`` where the multipole answer is accepted; the indices of
      ``False`` entries are ordered so those positions are considered for an
      upgrade first.
    - ``MAX_FULL_CALLS`` limits the number of inverse-ray evaluations; set it to
      ``0`` to obtain a pure hexadecapole light curve.
    - Source positions are processed in the lens centre-of-mass frame
      internally, but the public API consumes unshifted coordinates.
    - Results are produced lazily under JAX; call ``mags.block_until_ready()`` if
      immediate synchronisation is required.
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
    """Triple-lens magnification with a multipole baseline and limited refinement.

    The control flow mirrors :func:`mag_binary`: ``w_points`` is shifted to the
    binary centre of mass implied by ``s`` and ``q``, point-source images and a
    hexadecapole baseline are evaluated with ``nlenses=3``, and selected samples
    can be upgraded with the finite-source solver. A dedicated caustic trigger
    for triple lenses is not yet available, so ``test`` is ``False`` everywhere.
    Consequently ``jnp.argsort(test)`` produces the canonical ordering
    ``[0, 1, ..., N-1]`` and the first ``MAX_FULL_CALLS`` samples in that order
    are recomputed using :func:`microjax.inverse_ray.extended_source.mag_uniform`
    (``u1 == 0``) or
    :func:`microjax.inverse_ray.extended_source.mag_limb_dark`. The remaining
    samples retain the hexadecapole magnification.

    Parameters
    ----------
    w_points : Array
        One-dimensional complex ``jax.Array`` of source-plane coordinates
        (``x + 1j*y``) sampled along the trajectory. The returned magnifications
        track this ordering element by element.
    rho : float
        Angular source radius in Einstein units.
    r_resolution : int, optional
        Number of uniformly spaced radial samples per polar cell used by the
        inverse-ray integrator (``>= 2``).
    th_resolution : int, optional
        Number of uniformly spaced angular samples per polar cell used by the
        inverse-ray integrator (``>= 2``).
    u1 : float, optional
        Linear limb-darkening coefficient. Use ``0`` for a uniform surface
        brightness.
    delta_c : float, optional
        Dimensionless smoothing threshold supplied to
        :func:`microjax.inverse_ray.boundary.calc_facB` in the limb-darkened
        integrator.
    Nlimb : int, optional
        Number of source-limb samples traced through the lens to seed the polar
        region construction.
    bins_r : int, optional
        Number of histogram bins used when clustering limb radii into polar
        subregions (higher values capture finer radial structure).
    bins_th : int, optional
        Number of histogram bins used when clustering limb angles into polar
        subregions.
    margin_r : float, optional
        Radial margin applied to each subregion in units of ``rho``.
    margin_th : float, optional
        Angular margin applied to each subregion, expressed in degrees (converted
        to radians internally).
    MAX_FULL_CALLS : int, optional
        Maximum number of samples replaced by the inverse-ray finite-source
        solver. Set to ``0`` to stay on the hexadecapole baseline.
    chunk_size : int, optional
        Number of upgraded samples evaluated per :func:`jax.vmap` batch when the
        inverse-ray solver is invoked.
    **params
        Triple-lens configuration keywords forwarded to the low-level solvers.
        Required keys are ``s`` (lens 1–2 separation), ``q`` (lens 2 to lens 1
        mass ratio), ``q3`` (lens 3 to lens 1 mass ratio), ``r3`` (lens 1–3
        separation in Einstein units), and ``psi`` (polar angle of lens 3
        measured counter-clockwise from the lens 1–2 axis). Additional keywords
        are passed through untouched.

    Returns
    -------
    Array
        Real-valued magnification array with the same shape as ``w_points``.

    Notes
    -----
    - Source positions are shifted internally to the centre of mass defined by
      ``s`` and ``q`` before invoking
      :func:`microjax.point_source._images_point_source`; the public API uses
      unshifted coordinates.
    - Because ``test`` is all ``False``, the upgrade set degenerates to the
      leading ``min(MAX_FULL_CALLS, w_points.size)`` samples. Increase
      ``MAX_FULL_CALLS`` if more triple-lens samples need the finite-source
      treatment.
    - ``u1`` selects between
      :func:`microjax.inverse_ray.extended_source.mag_uniform` (``u1 == 0``) and
      :func:`microjax.inverse_ray.extended_source.mag_limb_dark`.
    - ``chunk_size`` controls how many upgraded samples each :func:`jax.vmap`
      invocation processes.
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
    
