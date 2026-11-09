"""Finite-source binary- and triple-lens light curves.

Use :func:`microjax.mag_binary` and :func:`microjax.mag_triple` with a complex source trajectory.
For source positions sufficiently far from caustics, microJAX uses a fast
finite-source approximation. Where a full calculation is needed, it traces
the lensed images of the circular source boundary and integrates the enclosed
brightness.

Both functions support a uniform source (``u1=0``) and linear limb darkening
(``u1>0``). They are designed for JAX compilation, vectorization, and
forward-mode automatic differentiation.

The CPU implementation can return diagnostic status flags with each value.
The accelerator implementation returns ``NaN`` for detected calculation
failures.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, lax, vmap

from .config import (
    DEFAULT_BINARY_CONFIG,
    DEFAULT_TRIPLE_CONFIG,
    BinaryMagConfig,
    TripleMagConfig,
)
from .geometry.lens import binary_geometry
from microjax.lens_geometry import triple_lens_geometry
from .selection import (
    _caustics_proximity_test,
    _planetary_caustic_test,
)
from .extended_source import (
    mag_limb_dark_boundary,
    mag_uniform_boundary,
    mag_uniform_triple_boundary,
)
from .roots.angular import (
    ANGULAR_CAPACITY,
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
)
from .geometry.topology import RADIAL_CAPACITY, RADIAL_TOPOLOGY
from microjax.multipole import mag_hexadecapole
from microjax.point_source import _images_point_source
from .cpu.lightcurve import (
    mag_cpu_hybrid_lightcurve,
)

from .cpu.lens import binary_lens, triple_lens, source_polynomial
from .cpu.support import polish_images, physical_image_mask
from .cpu.result import CpuMagnificationResult, INVALID_ROOTS, NONFINITE
from microjax.poly_solver import poly_roots

# Consistent array alias used across modules
Array = jnp.ndarray
# Geometry padding and error diagnostics are implementation details of the
# public one-pass path. They are fixed here so users do not mistake them for
# accuracy guarantees or physical model parameters.
_BOUNDARY_MARGIN_R = 0.5
_BOUNDARY_ABSOLUTE_TOLERANCE = 1.0e-5
# The high-level scheduler expresses radial-region batching through each
# configuration's radial_chunk_size, leaving the redundant compatibility flag
# off.
_PARALLEL_REGIONS = False
# The public one-pass path does not guarantee or adapt to this empirical
# radial-error threshold. Keep it as an internal diagnostic setting rather
# than presenting it as a user-controlled accuracy knob.
_BOUNDARY_RELATIVE_TOLERANCE = 1.0e-4
_FAST_RADIAL_INTERVAL_CAPACITY = 40


def _tiled_vmap_active_scalar(func, data, n_active, tile_size):
    """Map a scalar-output function over only the active prefix of ``data``.

    ``data`` keeps a static leading dimension for JIT compilation, while
    ``n_active`` is a dynamic scalar. Entire inactive tiles are skipped with
    one ``lax.cond`` per tile. A partially active final tile is evaluated in
    full, so the amount of deliberate over-computation is bounded by
    ``tile_size - 1`` rather than by the complete compacted buffer.
    """

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    n_items = data.shape[0]
    output_dtype = data.real.dtype
    if n_items == 0:
        return jnp.zeros((0,), dtype=output_dtype)

    pad_len = (-n_items) % tile_size
    padded = jnp.concatenate((data, jnp.repeat(data[:1], pad_len, axis=0)), axis=0)
    tiles = padded.reshape(-1, tile_size, *data.shape[1:])
    starts = jnp.arange(tiles.shape[0], dtype=jnp.int32) * tile_size

    def evaluate_tile(inputs):
        start, tile = inputs
        return lax.cond(
            start < n_active,
            lambda values: vmap(func)(values),
            lambda _: jnp.zeros((tile_size,), dtype=output_dtype),
            tile,
        )

    values = lax.map(evaluate_tile, (starts, tiles))
    return values.reshape(-1)[:n_items]


def _binary_prefilter(
    w_points: Array,
    rho: float,
    u1: float,
    s: float,
    q: float,
    c_m: float = 1.0e-2,
    c_f: float = 5.0,
    gamma: float = 2.0e-2,
) -> tuple[Array, Array, Array, Array]:
    """Return the profile value and profile-independent trigger diagnostics."""

    lens = binary_geometry(s, q)
    w_points_shifted = w_points - lens.shifted

    z, z_mask = _images_point_source(w_points_shifted, nlenses=2, a=lens.a, e1=lens.e1)
    # One geometric expansion supplies both the requested coefficient and
    # the coefficient-independent (u1=0) shortcut gate. The singleton image
    # axis broadcasts only the final profile weights, including when u1=0.
    magnification, correction = mag_hexadecapole(
        z[..., None],
        z_mask[..., None],
        rho,
        nlenses=2,
        u1=jnp.stack((u1, jnp.zeros_like(u1))),
        s=s,
        q=q,
        a=lens.a,
        e1=lens.e1,
    )
    mu_multi = magnification[..., 0]
    trigger_mu_multi = magnification[..., 1]
    trigger_delta_mu_multi = correction[..., 1]
    test1 = _caustics_proximity_test(
        w_points_shifted,
        z,
        z_mask,
        rho,
        trigger_delta_mu_multi,
        nlenses=2,
        s=s,
        q=q,
        a=lens.a,
        e1=lens.e1,
        c_m=c_m,
        c_f=c_f,
        gamma=gamma,
    )
    accepted = lax.cond(
        q < 0.01,
        lambda _: (test1 & _planetary_caustic_test(w_points_shifted, rho, s=s, q=q, a=lens.a, e1=lens.e1)),
        lambda _: test1,
        operand=None,
    )
    trigger_scale = jnp.maximum(jnp.abs(trigger_mu_multi), 1.0)
    return mu_multi, accepted, trigger_delta_mu_multi, trigger_scale


@jit
def _mag_binary_cpu_impl(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float,
):
    """Fuse the CPU multipole prefilter and selected one-shot scheduler."""

    # Full-solve triggering is deliberately profile-independent: uniform and
    # limb-darkened sources use the same geometric multipole selector.  A
    # formerly relaxed uniform-only gate admitted non-convergent
    # hexadecapole series for source discs intersecting a close-binary
    # caustic, while the identical LD geometry correctly entered full ICRS.
    multipole, accepted, multipole_error, multipole_trigger_scale = _binary_prefilter(
        w_points,
        rho,
        u1,
        s,
        q,
    )
    return mag_cpu_hybrid_lightcurve(
        w_points,
        multipole,
        accepted,
        multipole_error,
        multipole_trigger_scale,
        rho,
        lens=binary_lens(s, q),
        u1=u1,
    )


def _triple_prefilter(
    w_points: Array,
    rho: float,
    u1: float,
    s: float,
    q: float,
    q3: float,
    r3: float,
    psi: float,
) -> tuple[Array, Array]:
    """Return the triple-lens multipole baseline and acceptance mask."""

    lens = triple_lens_geometry(s, q, q3, r3, psi)
    w_points_shifted = w_points - lens.shifted
    lens_params = {
        "a": lens.a,
        "e1": lens.e1,
        "e2": lens.e2,
        "r3": r3,
        "psi": psi,
        "r3_complex": lens.r3_complex,
    }
    z, z_mask = _images_point_source(w_points_shifted, nlenses=3, **lens_params)
    mu_multi, delta_mu_multi = mag_hexadecapole(
        z,
        z_mask,
        rho,
        nlenses=3,
        u1=u1,
        **lens_params,
    )
    accepted = _caustics_proximity_test(
        w_points_shifted,
        z,
        z_mask,
        rho,
        delta_mu_multi,
        nlenses=3,
        **lens_params,
    )
    return mu_multi, accepted


@jit
def _mag_triple_cpu_impl(w_points, rho, *, s, q, q3, r3, psi, u1):
    """Use CPU image roots, a profile-independent gate, and the shared integrator."""
    w_points = jnp.asarray(w_points)
    if w_points.shape[0] == 0:
        return CpuMagnificationResult(jnp.zeros_like(w_points.real), jnp.zeros(w_points.shape, dtype=jnp.int32))
    lens = triple_lens(s, q, q3, r3, psi)
    geometry = triple_lens_geometry(s, q, q3, r3, psi)
    distances = jnp.abs(lens.positions[:, None] - lens.positions[None, :])
    valid_lens = (
        jnp.all(jnp.isfinite(lens.positions))
        & jnp.all(jnp.isfinite(lens.masses) & (lens.masses > 0))
        & jnp.all((distances > 0) | jnp.eye(3, dtype=bool))
        & (s > 0)
        & (rho > 0)
    )

    def evaluate(_):
        coefficients, image_shift = vmap(lambda w: source_polynomial(w, lens))(w_points)
        images = poly_roots(coefficients) + image_shift[:, None]
        images = vmap(lambda z, w: polish_images(z, w, lens))(images, w_points)
        mask = vmap(lambda z, w: physical_image_mask(z, w, lens))(images, w_points)
        # Multipole utilities use the internal midpoint frame and image-first axes.
        z = (images - geometry.shifted).T
        z_mask = mask.T
        parameters = dict(a=geometry.a, e1=geometry.e1, e2=geometry.e2, r3=r3, psi=psi, r3_complex=geometry.r3_complex)
        values, corrections = mag_hexadecapole(
            z[..., None],
            z_mask[..., None],
            rho,
            nlenses=3,
            u1=jnp.stack((u1, jnp.zeros_like(u1))),
            **parameters,
        )
        accepted = _caustics_proximity_test(
            w_points - geometry.shifted,
            z,
            z_mask,
            rho,
            corrections[..., 1],
            nlenses=3,
            **parameters,
        )
        count = jnp.sum(mask, axis=-1)
        accepted = accepted & (count >= 4) & (count % 2 == 0)
        return mag_cpu_hybrid_lightcurve(
            w_points,
            values[..., 0],
            accepted,
            corrections[..., 1],
            jnp.maximum(jnp.abs(values[..., 1]), 1.0),
            rho,
            lens=lens,
            u1=u1,
        )

    def invalid(_):
        return CpuMagnificationResult(
            jnp.full(w_points.shape, jnp.nan, dtype=w_points.real.dtype),
            jnp.full(w_points.shape, INVALID_ROOTS | NONFINITE, dtype=jnp.int32),
        )

    return lax.cond(valid_lens, evaluate, invalid, operand=None)


def _select_full_points(w_points: Array, accepted: Array) -> tuple[Array, Array, Array]:
    """Compact rejected trajectory samples into a static selection buffer."""

    sentinel = w_points.shape[0]
    n_active = jnp.sum(~accepted, dtype=jnp.int32)
    indices = jnp.nonzero(
        ~accepted,
        size=w_points.shape[0],
        fill_value=sentinel,
    )[0]
    # Duplicate the first rejected point into inactive buffer slots.  Using
    # trajectory point zero made padding cost depend on an unrelated source
    # position and produced a 5x A100 slowdown in one-active-lane batches.
    fallback_index = jnp.where(n_active > 0, indices[0], 0)
    safe_indices = jnp.where(indices < sentinel, indices, fallback_index)
    return w_points[safe_indices], indices, n_active


def _scatter_full_values(
    multipole: Array,
    indices: Array,
    full_values: Array,
) -> Array:
    """Scatter compact full solves without letting sentinels hit index zero."""

    with_sentinel = jnp.concatenate((multipole, jnp.zeros_like(multipole[:1])))
    with_sentinel = with_sentinel.at[indices].set(full_values)
    return with_sentinel[:-1]


def _validated_magnification(result):
    """Keep best-effort values unless a structural boundary check failed."""

    structural_failure = (
        result.status
        & (ANGULAR_CAPACITY | ANGULAR_DEGENERATE | ANGULAR_ROOT_FAILURE | RADIAL_CAPACITY | RADIAL_TOPOLOGY)
    ) != 0
    valid = jnp.isfinite(result.magnification) & ~structural_failure
    return jnp.where(valid, result.magnification, jnp.nan)


@partial(
    jit,
    static_argnames=("u1", "config"),
)
def _mag_binary_single_pass_impl(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
    config: BinaryMagConfig = DEFAULT_BINARY_CONFIG,
) -> Array:
    """Compute finite-source magnification for a binary lens.

    Parameters
    ----------
    w_points
        Complex source positions. The real and imaginary parts are the two
        source-plane coordinates in Einstein-radius units.
    rho
        Angular source radius in Einstein-radius units.
    s
        Projected binary-lens separation.
    q
        Mass ratio of the second lens to the first.
    u1
        Linear limb-darkening coefficient. Use zero for a uniform source.
    config
        Source-boundary sampling and accelerator scheduling configuration. The
        default is recommended for normal use. Scheduler settings change static
        JAX shapes and therefore trigger separate compilation.

    Returns
    -------
    Array
        Magnification at each input source position. A value is ``NaN`` when
        microJAX cannot construct a valid image boundary or integration region.

    Notes
    -----
    The function uses a fast approximation away from caustics and a full
    image-boundary integration where needed. The full calculation uses a fixed
    amount of work and is not automatically repeated with more expensive
    settings. Returned finite values do not carry a guaranteed error bound.
    """

    multipole, accepted, _, _ = _binary_prefilter(w_points, rho, u1, s, q)

    def make_boundary():
        if u1 == 0.0:

            def boundary(w):
                result = mag_uniform_boundary(
                    w,
                    rho,
                    margin_r=_BOUNDARY_MARGIN_R,
                    Nlimb=config.n_limb,
                    angular_atol=_BOUNDARY_ABSOLUTE_TOLERANCE,
                    relative_tolerance=_BOUNDARY_RELATIVE_TOLERANCE,
                    parallel_regions=_PARALLEL_REGIONS,
                    max_radial_subdivisions=1,
                    fixed_radial_order=19,
                    robust_roots=False,
                    certify_topology=False,
                    radial_strategy="fixed",
                    radial_chunk_size=config.radial_chunk_size,
                    return_info=True,
                    _planetary_local_chart=True,
                    _radial_interval_capacity=_FAST_RADIAL_INTERVAL_CAPACITY,
                    s=s,
                    q=q,
                )
                return _validated_magnification(result)

        else:

            def boundary(w):
                result = mag_limb_dark_boundary(
                    w,
                    rho,
                    u1=u1,
                    margin_r=_BOUNDARY_MARGIN_R,
                    Nlimb=config.n_limb,
                    angular_atol=_BOUNDARY_ABSOLUTE_TOLERANCE,
                    relative_tolerance=_BOUNDARY_RELATIVE_TOLERANCE,
                    parallel_regions=_PARALLEL_REGIONS,
                    max_radial_subdivisions=1,
                    robust_roots=False,
                    deep_topology_sampling=False,
                    radial_strategy="fixed",
                    certify_topology=False,
                    fixed_radial_order=19,
                    radial_chunk_size=config.radial_chunk_size,
                    angular_profile_subdivisions=1,
                    return_info=True,
                    _planetary_local_chart=True,
                    _radial_interval_capacity=_FAST_RADIAL_INTERVAL_CAPACITY,
                    s=s,
                    q=q,
                )
                return _validated_magnification(result)

        return boundary

    if w_points.shape[0] == 0:
        return multipole
    full_points, indices, n_active = _select_full_points(w_points, accepted)
    tile_size = min(config.source_tile_size, w_points.shape[0])

    def solve(boundary):
        full_values = _tiled_vmap_active_scalar(boundary, full_points, n_active, tile_size)
        return _scatter_full_values(multipole, indices, full_values)

    return solve(make_boundary())


@partial(
    jit,
    static_argnames=("u1", "config"),
)
def _mag_triple_single_pass_impl(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    q3: float,
    r3: float,
    psi: float,
    u1: float = 0.0,
    config: TripleMagConfig = DEFAULT_TRIPLE_CONFIG,
) -> Array:
    """Compute finite-source magnification for a triple lens.

    Parameters
    ----------
    w_points
        Complex source positions in Einstein-radius units.
    rho
        Angular source radius in Einstein-radius units.
    s, q
        Separation and mass ratio of the first two lenses.
    q3
        Mass of the third lens relative to the first.
    r3, psi
        Distance and position angle of the third lens.
    u1
        Linear limb-darkening coefficient. Use zero for a uniform source.
    config
        Source-boundary sampling and accelerator scheduling configuration. The
        default is recommended for normal use. Scheduler settings change static
        JAX shapes and therefore trigger separate compilation.

    Returns
    -------
    Array
        Magnification at each input source position. A value is ``NaN`` when
        microJAX cannot construct a valid image boundary or integration region.

    Notes
    -----
    The function uses a fast approximation away from caustics and a full
    image-boundary integration where needed. Small isolated images are handled
    in coordinates centred near those images to avoid loss of angular
    resolution. Returned finite values do not carry a guaranteed error bound.
    """

    multipole, accepted = _triple_prefilter(w_points, rho, u1, s, q, q3, r3, psi)

    if u1 == 0.0:

        def boundary(w):
            result = mag_uniform_triple_boundary(
                w,
                rho,
                s=s,
                q=q,
                q3=q3,
                r3=r3,
                psi=psi,
                Nlimb=config.n_limb,
                margin_r=_BOUNDARY_MARGIN_R,
                angular_atol=_BOUNDARY_ABSOLUTE_TOLERANCE,
                relative_tolerance=_BOUNDARY_RELATIVE_TOLERANCE,
                parallel_regions=_PARALLEL_REGIONS,
                max_radial_subdivisions=1,
                radial_strategy="fixed",
                radial_chunk_size=config.radial_chunk_size,
                fixed_radial_order=31,
                _compact_local_chart=True,
                return_info=True,
            )
            return _validated_magnification(result)

    else:

        def boundary(w):
            result = mag_limb_dark_boundary(
                w,
                rho,
                s=s,
                q=q,
                q3=q3,
                r3=r3,
                psi=psi,
                nlenses=3,
                u1=u1,
                Nlimb=config.n_limb,
                margin_r=_BOUNDARY_MARGIN_R,
                angular_atol=_BOUNDARY_ABSOLUTE_TOLERANCE,
                relative_tolerance=_BOUNDARY_RELATIVE_TOLERANCE,
                parallel_regions=_PARALLEL_REGIONS,
                max_radial_subdivisions=1,
                radial_strategy="fixed",
                certify_topology=False,
                radial_chunk_size=config.radial_chunk_size,
                angular_profile_subdivisions=1,
                _compact_local_chart=True,
                return_info=True,
            )
            return _validated_magnification(result)

    if w_points.shape[0] == 0:
        return multipole
    full_points, indices, n_active = _select_full_points(w_points, accepted)
    tile_size = min(config.source_tile_size, w_points.shape[0])
    full_values = _tiled_vmap_active_scalar(boundary, full_points, n_active, tile_size)
    return _scatter_full_values(multipole, indices, full_values)
