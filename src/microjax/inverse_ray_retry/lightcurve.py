"""Legacy safety-first inverse-ray light curve with bounded retries.

This module couples the hexadecapole approximation with selective inverse-ray
finite-source integrations and retries failed boundary integrations through a
bounded sequence of local and global kernels. The public entry point is
:func:`mag_binary_safe`.

Design highlights
-----------------

- **Hexadecapole-first evaluation**: start from the multipole estimate and
  upgrade only samples that fail the accuracy heuristics.
- **Explicit safety-first path**: :func:`mag_binary_safe` retains bounded
  local/global retries for difficult topology and root configurations.
- **Hybrid triggers**: combine caustic-proximity and planetary-caustic tests to
  decide when a full inverse-ray solve is required.
- **Chunked batching**: evaluate inverse-ray calls in configurable chunks to
  balance memory usage and accelerator occupancy.
- **Limb-darkening aware**: support both uniform and linear limb-darkened
  profiles through the ``u1`` parameter.

Workflow outline
----------------

1. Build a complex source-plane trajectory ``w_points``.
2. Call :func:`mag_binary_safe` with lens parameters and integration settings.
3. Feed the returned magnifications into downstream likelihoods (see
   :mod:`microjax.likelihood`).

References
----------

- Miyazaki & Kawahara (in prep.) — description of the adaptive microJAX
  solver stack (forthcoming).
"""

__all__ = ["mag_binary_safe"]

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap

from .cond_extended import (
    _caustics_proximity_test,
    _planetary_caustic_test,
)
from .extended_source import (
    mag_limb_dark_boundary,
    mag_uniform_boundary,
    mag_uniform_local_boundary,
)
from microjax.multipole import mag_hexadecapole
from microjax.point_source import _images_point_source

# Consistent array alias used across modules
Array = jnp.ndarray
_BOUNDARY_RETRY_CHUNK_SIZE = 16
_UNIFORM_FIXED_RETRY_CHUNK_SIZE = 8
_LOCAL_BULK_RETRY_CHUNK_SIZE = 16
# Fixed-16, alternate-origin, and final global retries are reached by few lanes.
# Four retains a small GPU batch while bounding final-stage over-compute to
# three lanes. It is a static scheduling choice for the measured A100 path.
_LOCAL_ANCHOR_RETRY_CHUNK_SIZE = 4
_LOCAL_IMAGE_RESCUE_MAX_Q = 1.0e-3
_LOCAL_IMAGE_RESCUE_MAX_RHO = 1.0e-3


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


def _retry_nonfinite_active_scalar(
    values, fallback, data, n_active, chunk_size
):
    """Compact and retry non-finite values in the active static prefix.

    The compaction happens between outer source batches.  This is important on
    accelerators: a source-local ``lax.cond`` nested under ``vmap`` is lowered to
    predicated work and makes every source pay for deep radial refinement.
    """

    capacity = data.shape[0]
    slots = jnp.arange(capacity, dtype=jnp.int32)
    failed = (slots < n_active) & ~jnp.isfinite(values)
    n_failed = jnp.sum(failed, dtype=jnp.int32)

    def run_retry(_):
        sentinel = capacity
        failed_indices = jnp.nonzero(
            failed, size=capacity, fill_value=sentinel
        )[0]
        # Inactive entries still execute in a partially filled SIMD chunk.
        # Fill them with the first actual failure instead of an unrelated
        # trajectory point, which may be a much harder caustic configuration.
        fallback_index = failed_indices[0]
        safe_indices = jnp.where(
            failed_indices < sentinel, failed_indices, fallback_index
        )
        failed_data = data[safe_indices]
        retry_values = _chunked_vmap_active_scalar(
            fallback, failed_data, n_failed, chunk_size
        )
        extended = jnp.concatenate((values, jnp.zeros_like(values[:1])))
        extended = extended.at[failed_indices].set(retry_values)
        return extended[:-1]

    # An explicit outer condition is materially cheaper on GPU than entering
    # a zero-active fixed-shape map for every retry stage.
    return lax.cond(n_failed > 0, run_retry, lambda _: values, operand=None)


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
        "u1",
        "Nlimb",
        "margin_r",
        "MAX_FULL_CALLS",
        "chunk_size",
        "parallel_regions",
        "_local_rescue_policy",
    ),
)
def _mag_binary_boundary_impl(
    w_points: Array,
    rho: float,
    s: float,
    q: float,
    u1: float = 0.0,
    Nlimb: int = 500,
    margin_r: float = 1.0,
    MAX_FULL_CALLS: int | None = None,
    chunk_size: int | None = None,
    angular_atol: float = 1e-5,
    relative_tolerance: float = 1e-4,
    parallel_regions: bool = False,
    _local_rescue_policy: bool | None = None,
) -> Array:
    """Safety-first binary light curve with bounded boundary retries.

    The hexadecapole approximation is evaluated over the complete trajectory.
    Every sample rejected by the multipole accuracy tests is compacted and sent
    to :func:`mag_uniform_boundary` or :func:`mag_limb_dark_boundary`. Uniform
    sources in the validated low-q/small-source domain use one image-local
    topology and a fixed four-way radial pass. Only rejected lanes advance to
    a fixed sixteen-way pass, then an alternate interior-anchor chart
    and finally deep global validation. Outside that domain, uniform sources
    use a fixed-one global bulk pass and compact only failures into a
    fixed-sixteen global retry, matching the regular GPU scheduling used by
    radial profiles. Failures after those retries remain
    ``NaN``; this function never calls the legacy dense polar-grid integrator.
    This is the implementation behind :func:`mag_binary_safe`. Use
    ``microjax.inverse_ray_dense.mag_binary_dense`` explicitly when the dense
    backend is required.

    Parameters
    ----------
    w_points : Array
        One-dimensional complex ``jax.Array`` of source-plane coordinates
        (``x + 1j*y``) sampled along the trajectory. The returned magnification
        array preserves the same ordering.
    rho : float
        Angular source radius in Einstein units.
    s : float
        Binary-lens separation in Einstein units.
    q : float
        Binary mass ratio ``m2 / m1``.
    u1 : float, optional
        Linear limb-darkening coefficient. Use ``0`` for a uniform surface
        brightness.
    Nlimb : int, optional
        Source-limb samples used to discover radial image topology.
    margin_r : float, optional
        Minimum radial support margin in units of ``rho``.
    MAX_FULL_CALLS : int or None, optional
        Optional compute budget for inverse-ray replacements. ``None`` (the
        default) allocates a static selection buffer spanning the trajectory,
        so every point rejected by the multipole tests is refined. A positive
        integer deliberately caps that count; zero disables refinement.
    chunk_size : int or None, optional
        Number of refined points evaluated per :func:`jax.vmap` batch. This
        defaults to one batch spanning the selection buffer. Pass a smaller
        integer to cap accelerator memory.
    angular_atol : float, optional
        Absolute magnification-error target for the boundary angular
        integration.
    relative_tolerance : float, optional
        Relative magnification-error target for the binary boundary backend.
        A direct result is accepted when its estimated error is no larger than
        ``angular_atol + relative_tolerance * abs(magnification)``.
    parallel_regions : bool, optional
        Evaluate all binary image regions simultaneously inside the boundary
        backend. This lowers latency for small outer batches (roughly up to 64
        source points on the benchmark A100), but the sequential default is
        faster and substantially more memory efficient for typical light-curve
        chunks.
    Returns
    -------
    Array
        Real-valued magnification array with the same shape as ``w_points``.

    Notes
    -----
    - ``test`` is ``True`` where the multipole solution is accepted; indices with
      ``False`` are compacted in trajectory order and considered for refinement.
    - Entire inactive full-solve chunks are skipped. At most ``chunk_size - 1``
      inactive buffer entries are evaluated in the partially active final chunk.
    - Local fixed four- and sixteen-way stages use decreasing
      fixed-shape device batches. Their inner radial-cell width is 64 in this A100-tuned
      high-level scheduler; the direct local API keeps its general default.
    - Uniform sources outside the validated local-chart domain use one fixed
      radial subdivision in the bulk pass and sixteen only for compacted
      failures. Direct ``mag_uniform_boundary`` calls retain adaptive radial
      integration and nested topology certification by default.
    - The alternate local-anchor retry is also a compact fixed-shape batch,
      but uses four-lane chunks because only a few lanes normally reach it;
      at most three padded local solves are evaluated in its final chunk.
    - An explicit ``MAX_FULL_CALLS = 0`` produces a purely hexadecapole light
      curve; the default never silently truncates the rejected-point set.
    - A finite-source failure is intentionally visible as ``NaN``. No
      dense-grid resolution arguments are accepted by this boundary API.
    """
    if MAX_FULL_CALLS is not None and MAX_FULL_CALLS < 0:
        raise ValueError("MAX_FULL_CALLS must be non-negative or None.")
    multipole, accepted, lens_params = _binary_prefilter(
        w_points, rho, u1, s, q
    )
    # The validated low-q/small-source local route already closes its saved
    # failures with an eight-way final global check.  The broader global route
    # needs sixteen-way refinement for two saved rho=1e-2 tolerance failures.
    # This static policy keeps the extra graph depth out of the common local
    # executable while retaining the stronger bounded retry elsewhere.
    deep_global_subdivisions = (
        8 if u1 == 0.0 and _local_rescue_policy is True else 16
    )
    # Outside the low-q/small-source local-chart domain, use the same regular
    # fixed scheduler as the validated radial-profile route.  Uniform brightness
    # does not justify paying for adaptive compact/scatter and two topology-area
    # evaluations at every bulk point.  The conservative adaptive/certified
    # kernel remains the final local-rescue fallback and the dynamic-policy
    # path; ordinary global points use fixed-1 then retry only failures at
    # fixed-16. Uniform area has a sharper radial step than the brightness-
    # weighted profile path, so its validated deep stage is one level finer
    # than the profile scheduler's fixed-4 retry.
    fixed_global_uniform = u1 == 0.0 and _local_rescue_policy is False

    if u1 == 0.0:

        def full_boundary(w):
            return mag_uniform_boundary(
                w,
                rho,
                margin_r=margin_r,
                Nlimb=Nlimb,
                angular_atol=angular_atol,
                relative_tolerance=relative_tolerance,
                parallel_regions=parallel_regions,
                max_radial_subdivisions=(
                    1 if fixed_global_uniform else 2
                ),
                robust_roots=not fixed_global_uniform,
                certify_topology=not fixed_global_uniform,
                deep_topology_sampling=False,
                radial_strategy=(
                    "fixed" if fixed_global_uniform else "adaptive"
                ),
                radial_chunk_size=8,
                s=s,
                q=q,
            )

        def refined_boundary(w):
            return mag_uniform_boundary(
                w,
                rho,
                margin_r=margin_r,
                Nlimb=Nlimb,
                angular_atol=angular_atol,
                relative_tolerance=relative_tolerance,
                parallel_regions=parallel_regions,
                max_radial_subdivisions=(
                    16
                    if fixed_global_uniform
                    else deep_global_subdivisions
                ),
                robust_roots=True,
                certify_topology=not fixed_global_uniform,
                radial_strategy=(
                    "fixed" if fixed_global_uniform else "adaptive"
                ),
                radial_chunk_size=8,
                s=s,
                q=q,
            )

        def local_boundary(w):
            return mag_uniform_local_boundary(
                w,
                rho,
                margin_r=margin_r,
                Nlimb=Nlimb,
                angular_atol=angular_atol,
                relative_tolerance=relative_tolerance,
                parallel_regions=parallel_regions,
                max_radial_subdivisions=4,
                refinement_safety_factor=1.0,
                radial_chunk_size=64,
                radial_strategy="fixed",
                certify_topology=False,
                s=s,
                q=q,
            )

        def refined_local_boundary(w):
            return mag_uniform_local_boundary(
                w,
                rho,
                margin_r=margin_r,
                Nlimb=Nlimb,
                angular_atol=angular_atol,
                relative_tolerance=relative_tolerance,
                parallel_regions=parallel_regions,
                max_radial_subdivisions=16,
                refinement_safety_factor=1.0,
                radial_chunk_size=64,
                radial_strategy="fixed",
                certify_topology=False,
                s=s,
                q=q,
            )

        def interior_local_boundary(w):
            return mag_uniform_local_boundary(
                w,
                rho,
                margin_r=margin_r,
                Nlimb=Nlimb,
                angular_atol=angular_atol,
                relative_tolerance=relative_tolerance,
                parallel_regions=parallel_regions,
                max_radial_subdivisions=16,
                refinement_safety_factor=1.0,
                radial_chunk_size=64,
                radial_strategy="fixed",
                certify_topology=False,
                prefer_interior_anchor=True,
                s=s,
                q=q,
            )
    else:

        def full_boundary(w):
            return mag_limb_dark_boundary(
                w,
                rho,
                nlenses=2,
                u1=u1,
                margin_r=margin_r,
                Nlimb=Nlimb,
                angular_atol=angular_atol,
                relative_tolerance=relative_tolerance,
                parallel_regions=parallel_regions,
                max_radial_subdivisions=1,
                robust_roots=False,
                radial_strategy="fixed",
                certify_topology=False,
                radial_chunk_size=8,
                s=s,
                q=q,
            )

        def refined_boundary(w):
            return mag_limb_dark_boundary(
                w,
                rho,
                nlenses=2,
                u1=u1,
                margin_r=margin_r,
                Nlimb=Nlimb,
                angular_atol=angular_atol,
                relative_tolerance=relative_tolerance,
                parallel_regions=parallel_regions,
                max_radial_subdivisions=4,
                robust_roots=True,
                radial_strategy="fixed",
                certify_topology=False,
                radial_chunk_size=8,
                angular_profile_subdivisions=2,
                s=s,
                q=q,
            )

    max_full_calls = (
        w_points.shape[0]
        if MAX_FULL_CALLS is None
        else min(MAX_FULL_CALLS, w_points.shape[0])
    )
    if max_full_calls == 0:
        return multipole
    effective_chunk_size = max_full_calls if chunk_size is None else chunk_size
    full_points, indices, n_active = _select_binary_full_points(
        w_points, accepted, max_full_calls
    )
    full_boundary = jax.checkpoint(
        full_boundary,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=False,
    )
    refined_boundary = jax.checkpoint(
        refined_boundary,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=False,
    )
    if u1 == 0.0:
        local_boundary = jax.checkpoint(
            local_boundary,
            policy=jax.checkpoint_policies.nothing_saveable,
            prevent_cse=False,
        )
        refined_local_boundary = jax.checkpoint(
            refined_local_boundary,
            policy=jax.checkpoint_policies.nothing_saveable,
            prevent_cse=False,
        )
        interior_local_boundary = jax.checkpoint(
            interior_local_boundary,
            policy=jax.checkpoint_policies.nothing_saveable,
            prevent_cse=False,
        )
    primary_boundary = (
        local_boundary
        if u1 == 0.0 and _local_rescue_policy is True
        else full_boundary
    )
    full_values = _chunked_vmap_active_scalar(
        primary_boundary,
        full_points,
        n_active,
        effective_chunk_size,
    )
    if u1 == 0.0:
        # The image-local chart is a targeted small-planet/small-source rescue.
        # Keeping the dispatch as one outer device condition preserves a fixed
        # GPU workload and avoids evaluating a more weakly validated chart
        # decomposition for ordinary binary-lens configurations.
        use_local_rescue = (q <= _LOCAL_IMAGE_RESCUE_MAX_Q) & (
            rho <= _LOCAL_IMAGE_RESCUE_MAX_RHO
        )

        def retry_local(values):
            # For a dynamically selected policy, global bulk was the primary
            # stage. Ordinary scalar q/rho specialise to local fixed-four above
            # and skip this redundant call.
            if _local_rescue_policy is not True:
                values = _retry_nonfinite_active_scalar(
                    values,
                    local_boundary,
                    full_points,
                    n_active,
                    min(_LOCAL_BULK_RETRY_CHUNK_SIZE, effective_chunk_size),
                )
            values = _retry_nonfinite_active_scalar(
                values,
                refined_local_boundary,
                full_points,
                n_active,
                min(_LOCAL_ANCHOR_RETRY_CHUNK_SIZE, effective_chunk_size),
            )
            values = _retry_nonfinite_active_scalar(
                values,
                interior_local_boundary,
                full_points,
                n_active,
                min(
                    _LOCAL_ANCHOR_RETRY_CHUNK_SIZE,
                    effective_chunk_size,
                ),
            )
            return _retry_nonfinite_active_scalar(
                values,
                refined_boundary,
                full_points,
                n_active,
                min(_LOCAL_ANCHOR_RETRY_CHUNK_SIZE, effective_chunk_size),
            )

        def retry_deep_global(values):
            return _retry_nonfinite_active_scalar(
                values,
                refined_boundary,
                full_points,
                n_active,
                min(
                    (
                        _UNIFORM_FIXED_RETRY_CHUNK_SIZE
                        if fixed_global_uniform
                        else _BOUNDARY_RETRY_CHUNK_SIZE
                    ),
                    effective_chunk_size,
                ),
            )

        if _local_rescue_policy is True:
            full_values = retry_local(full_values)
        elif _local_rescue_policy is False:
            full_values = retry_deep_global(full_values)
        else:
            full_values = lax.cond(
                use_local_rescue,
                retry_local,
                retry_deep_global,
                full_values,
            )
    else:
        full_values = _retry_nonfinite_active_scalar(
            full_values,
            refined_boundary,
            full_points,
            n_active,
            min(_BOUNDARY_RETRY_CHUNK_SIZE, effective_chunk_size),
        )
    return _scatter_binary_full_values(
        multipole, accepted, indices, full_values
    )




def mag_binary_safe(
    w_points: Array,
    rho: float,
    *,
    s: float,
    q: float,
    u1: float = 0.0,
    Nlimb: int = 500,
    margin_r: float = 1.0,
    MAX_FULL_CALLS: int | None = None,
    chunk_size: int | None = None,
    angular_atol: float = 1e-5,
    relative_tolerance: float = 1e-4,
    parallel_regions: bool = False,
) -> Array:
    """Validate and dispatch the safety-first binary API."""

    # With ordinary scalar lens parameters the validated rescue domain is
    # known before tracing.  Specialising this one policy bit prevents XLA from
    # embedding both the deep-global and three local kernels in every
    # executable.  When q/rho are tracers (for example inside an outer JIT),
    # retain the dynamic device condition and full generality.
    local_rescue_policy: bool | None
    if u1 != 0.0:
        local_rescue_policy = False
    elif isinstance(q, jax.core.Tracer) or isinstance(rho, jax.core.Tracer):
        local_rescue_policy = None
    else:
        local_rescue_policy = bool(
            q <= _LOCAL_IMAGE_RESCUE_MAX_Q
            and rho <= _LOCAL_IMAGE_RESCUE_MAX_RHO
        )

    return _mag_binary_boundary_impl(
        w_points,
        rho,
        s=s,
        q=q,
        u1=u1,
        Nlimb=Nlimb,
        margin_r=margin_r,
        MAX_FULL_CALLS=MAX_FULL_CALLS,
        chunk_size=chunk_size,
        angular_atol=angular_atol,
        relative_tolerance=relative_tolerance,
        parallel_regions=parallel_regions,
        _local_rescue_policy=local_rescue_policy,
    )


mag_binary_safe.__doc__ = _mag_binary_boundary_impl.__doc__
