"""Finite-source binary- and triple-lens light curves.

Use :func:`mag_binary` and :func:`mag_triple` with a complex source trajectory.
For source positions sufficiently far from caustics, microJAX uses a fast
finite-source approximation. Where a full calculation is needed, it traces
the lensed images of the circular source boundary and integrates the enclosed
brightness.

Both functions support a uniform source (``u1=0``) and linear limb darkening
(``u1>0``). They are designed for JAX compilation, vectorization, and
forward-mode automatic differentiation.

The returned values are numerical estimates without a guaranteed error bound.
If a valid image boundary or integration region cannot be constructed, the
corresponding result is ``NaN``.
"""

__all__ = ["mag_binary", "mag_triple"]

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
    _CPU_MULTIPOLE_GATE,
    mag_binary_cpu_hybrid_lightcurve,
    mag_binary_cpu_one_shot_hybrid_lightcurve,
)

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
    mu_multi, profile_delta_mu_multi = mag_hexadecapole(
        z,
        z_mask,
        rho,
        nlenses=2,
        u1=u1,
        s=s,
        q=q,
        a=lens.a,
        e1=lens.e1,
    )
    # The fast/full decision must describe lens/source geometry, not the
    # requested brightness profile.  Linear limb darkening rescales the
    # quadrupole and hexadecapole terms and can otherwise move an identical
    # source disc across the selector threshold.  Always use the uniform
    # value and correction as the shared trigger diagnostics, while retaining
    # the requested-profile multipole value above for accepted points.
    if u1 == 0.0:
        trigger_mu_multi = mu_multi
        trigger_delta_mu_multi = profile_delta_mu_multi
    else:
        trigger_mu_multi, trigger_delta_mu_multi = mag_hexadecapole(
            z,
            z_mask,
            rho,
            nlenses=2,
            u1=0.0,
            s=s,
            q=q,
            a=lens.a,
            e1=lens.e1,
        )
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
        lambda _: (
            test1
            & _planetary_caustic_test(
                w_points_shifted, rho, s=s, q=q, a=lens.a, e1=lens.e1
            )
        ),
        lambda _: test1,
        operand=None,
    )
    trigger_scale = jnp.maximum(jnp.abs(trigger_mu_multi), 1.0)
    return mu_multi, accepted, trigger_delta_mu_multi, trigger_scale


@partial(
    jit,
    static_argnames=("u1", "adaptive"),
)
def _mag_binary_cpu_impl(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float,
    adaptive: bool,
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
    if adaptive:
        return mag_binary_cpu_hybrid_lightcurve(
            w_points,
            multipole,
            accepted,
            multipole_error,
            multipole_trigger_scale,
            rho,
            s=s,
            q=q,
            u1=u1,
            rtol=_CPU_MULTIPOLE_GATE,
        )
    return mag_binary_cpu_one_shot_hybrid_lightcurve(
        w_points,
        multipole,
        accepted,
        multipole_error,
        multipole_trigger_scale,
        rho,
        s=s,
        q=q,
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
        & (
            ANGULAR_CAPACITY
            | ANGULAR_DEGENERATE
            | ANGULAR_ROOT_FAILURE
            | RADIAL_CAPACITY
            | RADIAL_TOPOLOGY
        )
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
        full_values = _tiled_vmap_active_scalar(
            boundary, full_points, n_active, tile_size
        )
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


def mag_binary(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
    config: BinaryMagConfig = DEFAULT_BINARY_CONFIG,
    backend: str = "accelerator",
    return_info: bool = False,
) -> Array:
    """Calculate a binary-lens finite-source light curve.

    ``backend="accelerator"`` preserves the established one-pass GPU-oriented
    scheduler. ``backend="cpu"`` selects the differentiable one-shot CPU
    scheduler: after the multipole prefilter, it traces the source limb once,
    selects one fixed high-order Cartesian or polar rule from the traced image
    state, and never retries. ``cpu-one-shot`` is a
    compatibility alias for the same default CPU path. The former adaptive
    CPU scheduler remains available explicitly as ``backend="cpu-adaptive"``.
    Exact radial tangencies stabilize the polar chart without treating
    ``n_limb`` as an accuracy order. The production CPU uses one fixed,
    calibrated multipole shortcut gate; it is not a full-solve error
    guarantee. For full one-shot solves, ``estimated_error`` is NaN and
    non-zero ``status`` denotes only a detected structural failure. CPU
    diagnostics are returned when ``return_info=True``; otherwise structurally
    invalid points are mapped to ``NaN``.
    """

    if backend in ("cpu", "cpu-one-shot", "cpu-adaptive"):
        result = _mag_binary_cpu_impl(
            w_points,
            rho,
            s=s,
            q=q,
            u1=u1,
            adaptive=backend == "cpu-adaptive",
        )
        if return_info:
            return result
        return jnp.where(result.status == 0, result.magnification, jnp.nan)

    if backend not in ("accelerator", "gpu"):
        raise ValueError(
            "backend must be 'accelerator', 'gpu', 'cpu', "
            "'cpu-one-shot', or 'cpu-adaptive'"
        )
    if return_info:
        raise ValueError(
            "return_info is available only for backend='cpu', "
            "backend='cpu-one-shot', or backend='cpu-adaptive'"
        )

    return _mag_binary_single_pass_impl(
        w_points,
        rho,
        s=s,
        q=q,
        u1=u1,
        config=config,
    )


def mag_triple(
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
    """Validate the public API before entering the JIT-compiled implementation."""

    return _mag_triple_single_pass_impl(
        w_points,
        rho,
        s=s,
        q=q,
        q3=q3,
        r3=r3,
        psi=psi,
        u1=u1,
        config=config,
    )


mag_triple.__doc__ = _mag_triple_single_pass_impl.__doc__
