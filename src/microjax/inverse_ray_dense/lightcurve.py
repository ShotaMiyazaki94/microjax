"""Legacy dense polar-grid inverse-ray light-curve solvers.

This module couples the hexadecapole approximation with selective inverse-ray
finite-source integrations to provide explicit binary and triple-lens dense
compatibility paths. The current boundary solver does not depend on this
package.

Design highlights
-----------------

- **Hexadecapole-first evaluation**: start from the multipole estimate and
  upgrade only samples that fail the accuracy heuristics.
- **Hybrid triggers**: combine caustic-proximity and planetary-caustic tests to
  decide when a full inverse-ray solve is required.
- **Chunked batching**: evaluate inverse-ray calls in configurable chunks to
  balance memory usage and accelerator occupancy.
- **Limb-darkening aware**: support both uniform and linear limb-darkened
  profiles through the ``u1`` parameter.
- **Shared infrastructure**: binary and triple lenses reuse the same chunking
  and integration utilities, so configuration parameters have consistent
  effects.

Workflow outline
----------------

1. Build a complex source-plane trajectory ``w_points``.
2. Call :func:`mag_binary_dense` or :func:`mag_triple` with lens parameters and
   dense-grid settings.
3. Feed the returned magnifications into downstream likelihoods (see
   :mod:`microjax.likelihood`).

References
----------

- Miyazaki & Kawahara (in prep.) — description of the adaptive microJAX
  solver stack (forthcoming).
"""

__all__ = ["mag_binary_dense", "mag_triple"]

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from microjax.lens_geometry import triple_lens_geometry
from .cond_extended import (
    _caustics_proximity_test,
    _planetary_caustic_test,
)
from .extended_source import (
    mag_limb_dark,
    mag_uniform,
)
from microjax.multipole import mag_hexadecapole
from microjax.point_source import _images_point_source

# Consistent array alias used across modules
Array = jnp.ndarray


def _chunked_vmap_active_scalar(func, data, n_active, chunk_size):
    """Map a scalar-output function over only the active prefix of ``data``.

    ``data`` keeps a static leading dimension for JIT compilation, while
    ``n_active`` is a dynamic scalar.  Entire inactive chunks are skipped with
    one ``lax.cond`` per chunk.  A partially active final chunk is evaluated in
    full, so the amount of deliberate over-computation is bounded by
    ``chunk_size - 1`` rather than by ``MAX_FULL_CALLS``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    n_items = data.shape[0]
    output_dtype = data.real.dtype
    if n_items == 0:
        return jnp.zeros((0,), dtype=output_dtype)

    pad_len = (-n_items) % chunk_size
    padded = jnp.concatenate(
        (data, jnp.repeat(data[:1], pad_len, axis=0)), axis=0
    )
    chunks = padded.reshape(-1, chunk_size, *data.shape[1:])
    starts = jnp.arange(chunks.shape[0], dtype=jnp.int32) * chunk_size

    def evaluate_chunk(inputs):
        start, chunk = inputs
        return lax.cond(
            start < n_active,
            lambda values: vmap(func)(values),
            lambda values: jnp.zeros((chunk_size,), dtype=output_dtype),
            chunk,
        )

    values = lax.map(evaluate_chunk, (starts, chunks))
    return values.reshape(-1)[:n_items]



def _binary_prefilter(
    w_points: Array,
    rho: float,
    u1: float,
    s: float,
    q: float,
) -> tuple[Array, Array, dict]:
    """Return the multipole baseline, acceptance mask, and lens parameters."""

    a = 0.5 * s
    e1 = q / (1.0 + q)
    lens_params = {"s": s, "q": q, "a": a, "e1": e1}
    x_cm = a * (1.0 - q) / (1.0 + q)
    w_points_shifted = w_points - x_cm

    z, z_mask = _images_point_source(
        w_points_shifted, nlenses=2, a=a, e1=e1
    )
    mu_multi, delta_mu_multi = mag_hexadecapole(
        z,
        z_mask,
        rho,
        nlenses=2,
        u1=u1,
        **lens_params,
    )
    test1 = _caustics_proximity_test(
        w_points_shifted,
        z,
        z_mask,
        rho,
        delta_mu_multi,
        nlenses=2,
        **lens_params,
    )
    test2 = _planetary_caustic_test(
        w_points_shifted, rho, **lens_params
    )
    accepted = jnp.where(q < 0.01, test1 & test2, test1)
    return mu_multi, accepted, lens_params


def _select_binary_full_points(
    w_points: Array,
    accepted: Array,
    max_full_calls: int,
) -> tuple[Array, Array, Array]:
    """Compact rejected trajectory samples into a static selection buffer."""

    sentinel = w_points.shape[0]
    n_required = jnp.sum(~accepted, dtype=jnp.int32)
    n_active = jnp.minimum(n_required, jnp.int32(max_full_calls))
    indices = jnp.nonzero(
        ~accepted,
        size=max_full_calls,
        fill_value=sentinel,
    )[0]
    # Duplicate the first rejected point into inactive buffer slots.  Using
    # trajectory point zero made padding cost depend on an unrelated source
    # position and produced a 5x A100 slowdown in one-active-lane batches.
    fallback_index = jnp.where(n_active > 0, indices[0], 0)
    safe_indices = jnp.where(indices < sentinel, indices, fallback_index)
    return w_points[safe_indices], indices, n_active


def _scatter_binary_full_values(
    multipole: Array,
    accepted: Array,
    indices: Array,
    full_values: Array,
) -> Array:
    """Scatter compact full solves without letting sentinels hit index zero."""

    with_sentinel = jnp.concatenate((multipole, jnp.zeros_like(multipole[:1])))
    with_sentinel = with_sentinel.at[indices].set(full_values)
    combined = with_sentinel[:-1]
    return jnp.where(accepted, multipole, combined)



@partial(
    jit,
    static_argnames=(
        "r_resolution",
        "th_resolution",
        "u1",
        "delta_c",
        "Nlimb",
        "bins_r",
        "bins_th",
        "margin_r",
        "margin_th",
        "MAX_FULL_CALLS",
        "chunk_size",
        "grid_fp32",
    ),
)
def _mag_binary_dense_impl(
    w_points: Array,
    rho: float,
    s: float,
    q: float,
    r_resolution: int = 1000,
    th_resolution: int = 1000,
    u1: float = 0.0,
    delta_c: float = 0.01,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 1.0,
    margin_th: float = 1.0,
    MAX_FULL_CALLS: int | None = None,
    chunk_size: int | None = None,
    grid_fp32: bool = False,
) -> Array:
    """Binary light curve using only the legacy dense polar-grid backend.

    This compatibility function owns all dense-grid resolution and mixed-
    precision controls. It never calls the boundary-root integrator. Ordinary
    calculations should use ``microjax.inverse_ray.mag_binary``; call this
    function explicitly for legacy reproduction or dense-grid comparisons.
    """

    if MAX_FULL_CALLS is not None and MAX_FULL_CALLS < 0:
        raise ValueError("MAX_FULL_CALLS must be non-negative or None.")
    multipole, accepted, lens_params = _binary_prefilter(
        w_points, rho, u1, s, q
    )

    def dense64(w):
        if u1 == 0.0:
            return mag_uniform(
                w,
                rho,
                nlenses=2,
                r_resolution=r_resolution,
                th_resolution=th_resolution,
                bins_r=bins_r,
                bins_th=bins_th,
                margin_r=margin_r,
                margin_th=margin_th,
                Nlimb=Nlimb,
                grid_fp32=False,
                **lens_params,
            )
        return mag_limb_dark(
            w,
            rho,
            nlenses=2,
            r_resolution=r_resolution,
            th_resolution=th_resolution,
            u1=u1,
            delta_c=delta_c,
            bins_r=bins_r,
            bins_th=bins_th,
            margin_r=margin_r,
            margin_th=margin_th,
            Nlimb=Nlimb,
            grid_fp32=False,
            **lens_params,
        )

    def dense_mixed(w):
        if u1 == 0.0:
            return mag_uniform(
                w,
                rho,
                nlenses=2,
                r_resolution=r_resolution,
                th_resolution=th_resolution,
                bins_r=bins_r,
                bins_th=bins_th,
                margin_r=margin_r,
                margin_th=margin_th,
                Nlimb=Nlimb,
                grid_fp32=True,
                **lens_params,
            )
        return mag_limb_dark(
            w,
            rho,
            nlenses=2,
            r_resolution=r_resolution,
            th_resolution=th_resolution,
            u1=u1,
            delta_c=delta_c,
            bins_r=bins_r,
            bins_th=bins_th,
            margin_r=margin_r,
            margin_th=margin_th,
            Nlimb=Nlimb,
            grid_fp32=True,
            **lens_params,
        )

    max_full_calls = (
        w_points.shape[0]
        if MAX_FULL_CALLS is None
        else min(MAX_FULL_CALLS, w_points.shape[0])
    )
    if max_full_calls == 0:
        return multipole
    effective_chunk_size = (
        min(100, max_full_calls) if chunk_size is None else chunk_size
    )
    full_points, indices, n_active = _select_binary_full_points(
        w_points, accepted, max_full_calls
    )
    dense64 = jax.checkpoint(
        dense64,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=False,
    )
    if grid_fp32:
        dense_mixed = jax.checkpoint(
            dense_mixed,
            policy=jax.checkpoint_policies.nothing_saveable,
            prevent_cse=False,
        )
        mixed_values = _chunked_vmap_active_scalar(
            dense_mixed,
            full_points,
            n_active,
            effective_chunk_size,
        )
        full_values = lax.cond(
            jnp.all(jnp.isfinite(mixed_values)),
            lambda _: mixed_values,
            lambda _: _chunked_vmap_active_scalar(
                dense64,
                full_points,
                n_active,
                effective_chunk_size,
            ),
            operand=None,
        )
    else:
        full_values = _chunked_vmap_active_scalar(
            dense64,
            full_points,
            n_active,
            effective_chunk_size,
        )
    return _scatter_binary_full_values(
        multipole, accepted, indices, full_values
    )


def mag_binary_dense(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    r_resolution: int = 1000,
    th_resolution: int = 1000,
    u1: float = 0.0,
    delta_c: float = 0.01,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 1.0,
    margin_th: float = 1.0,
    MAX_FULL_CALLS: int | None = None,
    chunk_size: int | None = None,
    grid_fp32: bool = False,
) -> Array:
    """Validate and dispatch the explicit legacy dense binary API."""

    return _mag_binary_dense_impl(
        w_points,
        rho,
        s=s,
        q=q,
        r_resolution=r_resolution,
        th_resolution=th_resolution,
        u1=u1,
        delta_c=delta_c,
        Nlimb=Nlimb,
        bins_r=bins_r,
        bins_th=bins_th,
        margin_r=margin_r,
        margin_th=margin_th,
        MAX_FULL_CALLS=MAX_FULL_CALLS,
        chunk_size=chunk_size,
        grid_fp32=grid_fp32,
    )


mag_binary_dense.__doc__ = _mag_binary_dense_impl.__doc__


@partial(jit,static_argnames=("r_resolution", "th_resolution", "u1", "delta_c",
                              "bins_r", "bins_th", "margin_r", "margin_th",
                              "Nlimb", "MAX_FULL_CALLS", "chunk_size", "grid_fp32"))
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
    grid_fp32: bool = False,
    **params,
) -> Array:
    """Triple-lens magnification with a multipole baseline and limited refinement.

    The procedure is analogous to :func:`mag_binary_dense`. The trajectory is shifted
    to the centre of mass implied by ``s`` and ``q``, point-source images are
    evaluated with ``nlenses = 3``, and the hexadecapole approximation provides
    the baseline magnification. Because specialised accuracy tests for triple
    lenses are not yet available, the boolean mask ``test`` is ``False`` at all
    entries. Consequently ``jnp.argsort(test)`` produces the original ordering
    and the first ``MAX_FULL_CALLS`` elements are recomputed with
    :func:`microjax.inverse_ray_dense.extended_source.mag_uniform` for ``u1 = 0`` or
    :func:`microjax.inverse_ray_dense.extended_source.mag_limb_dark`` otherwise. The
    remaining points retain their hexadecapole magnifications.

    Parameters
    ----------
    w_points : Array
        One-dimensional complex ``jax.Array`` of source-plane coordinates
        (``x + 1j*y``) sampled along the trajectory. The returned magnification
        array preserves the same ordering.
    rho : float
        Angular source radius in Einstein units.
    r_resolution : int, optional
        Number of uniformly spaced radial samples per polar cell used by the
        inverse-ray integrator.
    th_resolution : int, optional
        Number of uniformly spaced angular samples per polar cell used by the
        inverse-ray integrator.
    u1 : float, optional
        Linear limb-darkening coefficient. Use ``0`` for a uniform surface
        brightness.
    delta_c : float, optional
        Dimensionless smoothing threshold supplied to
        :func:`microjax.inverse_ray_dense.boundary.calc_facB` in the limb-darkened
        integrator.
    Nlimb : int, optional
        Number of source-limb samples traced through the lens to seed the polar
        region construction.
    bins_r : int, optional
        Number of histogram bins used when clustering limb radii into polar
        subregions; larger values resolve smaller radial features.
    bins_th : int, optional
        Number of histogram bins used when clustering limb angles into polar
        subregions.
    margin_r : float, optional
        Radial margin applied to each subregion in units of ``rho``.
    margin_th : float, optional
        Angular margin applied to each subregion, expressed in degrees (converted
        to radians internally).
    MAX_FULL_CALLS : int, optional
        Maximum number of points replaced by the inverse-ray finite-source
        solver. Setting ``MAX_FULL_CALLS = 0`` leaves the hexadecapole baseline
        unchanged.
    chunk_size : int, optional
        Number of refined points evaluated per :func:`jax.vmap` batch when the
        inverse-ray solver is invoked.
    grid_fp32 : bool, optional
        When ``True``, evaluate the dense inverse-ray polar grid in ``float32``/
        ``complex64`` while keeping the image finding and region construction in
        the ambient precision. This preserves the baseline path when ``False``.
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
    - Because the ``test`` mask evaluates to ``False`` for every position, the
      refinement stage processes the leading
      ``min(MAX_FULL_CALLS, w_points.size)`` entries. Increasing
      ``MAX_FULL_CALLS`` expands this subset.
    - ``u1`` selects between
      :func:`microjax.inverse_ray_dense.extended_source.mag_uniform` (``u1 == 0``) and
      :func:`microjax.inverse_ray_dense.extended_source.mag_limb_dark`.
    - ``chunk_size`` specifies the number of refined points handled by each
      :func:`jax.vmap` invocation.
    """
    nlenses = 3
    s, q, q3 = params["s"], params["q"], params["q3"]
    geometry = triple_lens_geometry(
        s, q, q3, params["r3"], params["psi"]
    )
    _params = {
        **params,
        "a": geometry.a,
        "e1": geometry.e1,
        "e2": geometry.e2,
    }
    w_points_shifted = w_points - geometry.shifted

    z, z_mask = _images_point_source(w_points_shifted, nlenses=nlenses, **_params)
    mu_multi, delta_mu_multi = mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, u1=u1, **_params)
    test = jnp.zeros_like(w_points).astype(jnp.bool_)

    if u1 == 0.0:
        def _mag_full_baseline(w):
            return mag_uniform(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution = th_resolution,
                               bins_r = bins_r, bins_th = bins_th, margin_r = margin_r, margin_th = margin_th,
                               Nlimb = Nlimb, grid_fp32=False, **_params)
    else:
        def _mag_full_baseline(w):
            return mag_limb_dark(w, rho, nlenses = nlenses, r_resolution = r_resolution, th_resolution= th_resolution,
                                 u1 = u1, delta_c = delta_c, bins_r = bins_r, bins_th = bins_th, margin_r = margin_r,
                                 margin_th = margin_th, Nlimb = Nlimb, grid_fp32=False, **_params)

    idx_sorted = jnp.argsort(test)
    idx_full = idx_sorted[:MAX_FULL_CALLS]

    def chunked_vmap(func, data, chunk_size):
        N = data.shape[0]
        pad_len = (-N) % chunk_size
        chunks = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)).reshape(-1, chunk_size, *data.shape[1:])
        return lax.map(lambda c: vmap(func)(c), chunks).reshape(-1, *data.shape[2:])[:N]

    if grid_fp32:
        def _mag_full_mixed(w):
            return mag_limb_dark(w, rho, nlenses=nlenses, r_resolution=r_resolution, th_resolution=th_resolution,
                                 u1=u1, delta_c=delta_c, bins_r=bins_r, bins_th=bins_th, margin_r=margin_r,
                                 margin_th=margin_th, Nlimb=Nlimb, grid_fp32=True, **_params) if u1 != 0.0 else \
                   mag_uniform(w, rho, nlenses=nlenses, r_resolution=r_resolution, th_resolution=th_resolution,
                               bins_r=bins_r, bins_th=bins_th, margin_r=margin_r, margin_th=margin_th,
                               Nlimb=Nlimb, grid_fp32=True, **_params)

        _mag_full_mixed = jax.checkpoint(_mag_full_mixed)
        _mag_full_baseline = jax.checkpoint(_mag_full_baseline)
        mag_extended_mixed = chunked_vmap(_mag_full_mixed, w_points[idx_full], chunk_size)
        mag_extended = lax.cond(
            jnp.all(jnp.isfinite(mag_extended_mixed)),
            lambda _: mag_extended_mixed,
            lambda _: chunked_vmap(_mag_full_baseline, w_points[idx_full], chunk_size),
            operand=None,
        )
    else:
        _mag_full_baseline = jax.checkpoint(_mag_full_baseline)
        mag_extended = chunked_vmap(_mag_full_baseline, w_points[idx_full], chunk_size)
    mags = mu_multi.at[idx_full].set(mag_extended)
    mags = jnp.where(test, mu_multi, mags)
    return mags
