"""Single-pass boundary inverse-ray light-curve solvers.

This package contains the current ``mag_binary`` and ``mag_triple`` algorithms:
image-boundary detection followed by one bounded radial integration pass.
Legacy retry and dense-grid implementations live in ``inverse_ray_retry`` and
``inverse_ray_dense`` respectively.

Design highlights
-----------------

- **Hexadecapole-first evaluation**: start from the multipole estimate and
  upgrade only samples that fail the accuracy heuristics.
- **Retry-free public fast path**: each solver sends rejected
  source through one shallow fixed-1 boundary kernel, returning its
  best-effort value unless a structural check fails.
- **Lens-aware triggers**: both solvers use multipole and caustic-proximity
  tests; the binary path adds its planetary-caustic guard.
- **Internal GPU batching**: evaluate inverse-ray calls in fixed-size tiles;
  this scheduler choice is deliberately absent from the numerical API.
- **Limb-darkening aware**: support both uniform and linear limb-darkened
  profiles through the ``u1`` parameter.

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

# Consistent array alias used across modules
Array = jnp.ndarray
_SOURCE_TILE_SIZE = 100
# Match the existing planetary prefilter regime so HMC sees no additional
# parameter-space branch boundary. Both kernels stay in one compiled graph;
# scalar lax.cond executes only the selected source-vmap at runtime.
_PLANETARY_LOCAL_Q_MAX = 1.0e-2


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
) -> tuple[Array, Array]:
    """Return the multipole baseline and its acceptance mask."""

    lens = binary_geometry(s, q)
    w_points_shifted = w_points - lens.shifted

    z, z_mask = _images_point_source(w_points_shifted, nlenses=2, a=lens.a, e1=lens.e1)
    mu_multi, delta_mu_multi = mag_hexadecapole(
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
    test1 = _caustics_proximity_test(
        w_points_shifted,
        z,
        z_mask,
        rho,
        delta_mu_multi,
        nlenses=2,
        s=s,
        q=q,
        a=lens.a,
        e1=lens.e1,
    )
    accepted = lax.cond(
        q < 0.01,
        lambda _: test1 & _planetary_caustic_test(w_points_shifted, rho, s=s, q=q, a=lens.a, e1=lens.e1),
        lambda _: test1,
        operand=None,
    )
    return mu_multi, accepted


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
    """Binary light curve with one fixed, retry-free boundary pass.

    The hexadecapole approximation is evaluated over the complete trajectory.
    Samples rejected by its accuracy tests are compacted once and evaluated by
    one regular shallow boundary kernel. Uniform and linear limb-darkened
    sources use one unsplit radial cell and the fixed EA20 root solver. Uniform
    Both uniform and linear limb-darkened brightness use the embedded G15/K31
    radial rule, keeping their radial root-sampling density consistent.

    No failed source is rerun with another radial depth or backend. In the
    uniform-source graph at ``q < 0.01``, a radially separated and angularly
    narrow planetary branch may carry a local chart centre through the same
    radial scheduler; larger mass ratios use only the global centre-of-mass
    chart. Both paths share one static JAX graph and are selected outside the
    source ``vmap``.
    Root, capacity, topology, and non-finite failures remain ``NaN``.
    A radial embedded-error warning does not discard the one-pass estimate;
    this is deliberately a best-effort fast path rather than a guarantee that
    every value meets ``config.angular_atol``/``config.relative_tolerance``. Use
    ``microjax.inverse_ray_retry.mag_binary_safe`` when that strict acceptance
    contract is required.

    Every point rejected by the multipole gate is evaluated by the boundary
    kernel. GPU tiling is an internal scheduler detail, not a numerical option.
    """

    multipole, accepted = _binary_prefilter(w_points, rho, u1, s, q)

    def make_boundary(use_local_chart):
        if u1 == 0.0:

            def boundary(w):
                result = mag_uniform_boundary(
                    w,
                    rho,
                    margin_r=config.margin_r,
                    Nlimb=config.n_limb,
                    angular_atol=config.angular_atol,
                    relative_tolerance=config.relative_tolerance,
                    parallel_regions=config.parallel_regions,
                    max_radial_subdivisions=1,
                    fixed_radial_order=31,
                    robust_roots=False,
                    certify_topology=False,
                    radial_strategy="fixed",
                    radial_chunk_size=8,
                    return_info=True,
                    _planetary_local_chart=use_local_chart,
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
                    margin_r=config.margin_r,
                    Nlimb=config.n_limb,
                    angular_atol=config.angular_atol,
                    relative_tolerance=config.relative_tolerance,
                    parallel_regions=config.parallel_regions,
                    max_radial_subdivisions=1,
                    robust_roots=False,
                    radial_strategy="fixed",
                    certify_topology=False,
                    radial_chunk_size=8,
                    angular_profile_subdivisions=1,
                    return_info=True,
                    _planetary_local_chart=use_local_chart,
                    s=s,
                    q=q,
                )
                return _validated_magnification(result)

        return boundary

    if w_points.shape[0] == 0:
        return multipole
    full_points, indices, n_active = _select_full_points(w_points, accepted)
    tile_size = min(_SOURCE_TILE_SIZE, w_points.shape[0])

    def solve(boundary):
        full_values = _tiled_vmap_active_scalar(boundary, full_points, n_active, tile_size)
        return _scatter_full_values(multipole, indices, full_values)

    return lax.cond(
        q < _PLANETARY_LOCAL_Q_MAX,
        lambda _: solve(make_boundary(True)),
        lambda _: solve(make_boundary(False)),
        operand=None,
    )


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
    """Triple light curve with one fixed, retry-free boundary pass.

    A triple-lens hexadecapole estimate is evaluated for the complete source
    trajectory. Samples rejected by the generic caustic-proximity tests are
    compacted once and evaluated with exact degree-eight angular boundary
    roots and one unsplit G15/K31 radial pass. Uniform and linear
    limb-darkened sources share the same topology and radial node density.

    Public source coordinates retain the centre of mass of the first two
    lenses. The current triple solver deliberately uses that global polar
    chart throughout; binary-specific planetary re-centring is not applied.
    Structural root, topology, capacity, and non-finite failures return
    ``NaN``. A radial embedded-error warning retains its finite best-effort
    value, matching the retry-free contract of :func:`mag_binary`.
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
                margin_r=config.margin_r,
                angular_atol=config.angular_atol,
                relative_tolerance=config.relative_tolerance,
                parallel_regions=config.parallel_regions,
                max_radial_subdivisions=1,
                radial_strategy="fixed",
                radial_chunk_size=8,
                fixed_radial_order=31,
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
                margin_r=config.margin_r,
                angular_atol=config.angular_atol,
                relative_tolerance=config.relative_tolerance,
                parallel_regions=config.parallel_regions,
                max_radial_subdivisions=1,
                radial_strategy="fixed",
                certify_topology=False,
                radial_chunk_size=8,
                angular_profile_subdivisions=1,
                return_info=True,
            )
            return _validated_magnification(result)

    if w_points.shape[0] == 0:
        return multipole
    full_points, indices, n_active = _select_full_points(w_points, accepted)
    tile_size = min(_SOURCE_TILE_SIZE, w_points.shape[0])
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
) -> Array:
    """Validate the public API before entering the JIT-compiled implementation."""

    return _mag_binary_single_pass_impl(
        w_points,
        rho,
        s=s,
        q=q,
        u1=u1,
        config=config,
    )


mag_binary.__doc__ = _mag_binary_single_pass_impl.__doc__


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
