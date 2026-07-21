"""Retry-capable boundary finite-source magnification integrators.

The binary-lens backend removes the dense angular axis. It solves the
source-boundary crossings on each image-plane ring, preserves the radial image
topology with fixed-shape buffers, and applies embedded angular/radial
quadrature.  It supports both uniform disks and arbitrary smooth axisymmetric
brightness callables without a ``th_resolution`` parameter. Fixed capacities
and explicit status bits are retained because JAX compilation still requires
static shapes and difficult caustic configurations must remain detectable.

All implementations use lens-centre-of-mass coordinates consistently with the
point-source utilities.
"""

import jax
import jax.numpy as jnp
from microjax.lens_geometry import triple_lens_geometry
from microjax.point_source import _images_point_source
from .merge_area import calc_source_limb
from .limb_darkening import linear_limb_intensity
from .boundary import distance_from_source
from .angular import (
    ANGULAR_CAPACITY,
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
    angular_intervals_binary_roots,
    angular_intervals_triple_roots,
    angular_measure_binary_roots,
)
from .angular_quadrature import integrate_angular_profile
from .levelset import binary_level_set, triple_level_set
from .radial import (
    RADIAL_CAPACITY,
    RADIAL_INTERVAL_CAPACITY,
    RADIAL_LOCAL_RETRY_BREAKPOINT_CAPACITY,
    RADIAL_LOCAL_RETRY_INTERVAL_CAPACITY,
    RADIAL_OK,
    RADIAL_RETRY_BREAKPOINT_CAPACITY,
    RADIAL_RETRY_INTERVAL_CAPACITY,
    RADIAL_TOLERANCE,
    RADIAL_TOPOLOGY,
    build_local_image_charts,
    define_radial_topology,
)
from .radial_quadrature import (
    RadialIntegral,
    RadialIntegrand,
    adaptive_radial_integral,
    fixed_radial_integral,
)
from typing import Callable, NamedTuple, Optional, Union

# Simple alias for readability in type hints
Array = jnp.ndarray
# Eight cells retain A100 throughput while cutting the direct outer-vmap peak
# by about 2.8x versus the historical 16-cell scheduler. Parallel mode uses the
# exact topology capacity below, so it never pads 64 live slots back to 128.
_SEQUENTIAL_RADIAL_CHUNK_SIZE = 8


class BoundaryMagnificationResult(NamedTuple):
    """Magnification and diagnostics from boundary-aware polar ICRS."""

    magnification: Array
    estimated_error: Array
    status: Array


def _integration_dtypes(w_center: complex) -> tuple[jnp.dtype, jnp.dtype]:
    """Return the real and complex dtypes used by boundary integration."""

    real_dtype = jnp.asarray(w_center).real.dtype
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    return real_dtype, complex_dtype



def mag_uniform_boundary(
    w_center: complex,
    rho: float,
    *,
    s: float,
    q: float,
    Nlimb: int = 500,
    margin_r: float = 0.5,
    angular_atol: float = 1e-5,
    relative_tolerance: float = 1e-4,
    parallel_regions: bool = False,
    return_info: bool = False,
    track_limb_roots: bool = True,
    jacobian_radial_margin: bool = True,
    max_radial_subdivisions: int = 8,
    robust_roots: bool = True,
    certify_topology: bool = True,
    deep_topology_sampling: bool = True,
    radial_strategy: str = "adaptive",
    radial_chunk_size: int = _SEQUENTIAL_RADIAL_CHUNK_SIZE,
    fixed_radial_order: int = 31,
) -> Union[Array, BoundaryMagnificationResult]:
    """Uniform-source magnification without a dense angular resolution.

    For each radial node, the binary source-boundary level set is recovered as
    a degree-three Fourier polynomial.  Its degree-six self-inversive boundary
    polynomial supplies all angular crossings directly, after unit-circle and
    residual validation.  Radial topology changes are integrated with an
    endpoint-transformed embedded Gauss rule.  ``angular_atol`` is retained as
    the public absolute magnification-error target for the combined angular and
    radial calculation. ``relative_tolerance`` adds a magnification-scaled
    budget, so acceptance requires ``error <= angular_atol +
    relative_tolerance * abs(magnification)``. Neither angular nor radial grid
    resolution is selected by the caller.

    ``calc_source_limb`` is retained to locate the radial support of all image
    components.  Unlike the legacy dense-grid path, the limb images are not
    clustered into independent radial and angular histograms: their sampled
    radial projections are merged, and every active ring is integrated over
    the complete angular period.  Only binary lenses and uniform brightness
    are supported in this prototype.

    Set ``parallel_regions=True`` to evaluate all fixed-capacity radial cells in
    one inner ``vmap``.  The default evaluates active cells in chunks of 8,
    which uses much less memory when an outer source batch already saturates an
    accelerator.

    ``Nlimb`` remains a topology-tracing parameter rather than a quadrature
    resolution. Adjacent limb roots are matched by default because
    polynomial-root ordering changes otherwise create false radial extrema;
    ``track_limb_roots=False`` is retained for diagnostic comparisons.
    The radial support margin is also scaled by the implicitly differentiated
    image motion by default. This is essential near caustics, where a fixed
    source-plane multiple of ``rho`` is not a safe image-plane margin;
    ``jacobian_radial_margin=False`` is diagnostic only.
    ``max_radial_subdivisions`` bounds the static radial refinement schedule.
    The default 8 is intended for scalar/direct calls.  The light-curve API
    instead selects a bounded fixed-shape schedule: its ordinary global route
    uses an EA20, one-subdivision bulk pass and compacts only failures into an
    EA40, sixteen-subdivision retry.  The separately gated local-chart route
    keeps its own fixed-depth stages and a certified adaptive global fallback.
    ``robust_roots`` selects the fixed 40-step EA schedule and remains the
    direct-call default.  Production calls that request topology certification
    compare two
    genuinely nested source-limb phases: the shallow route uses ``Nlimb`` and
    ``2 * Nlimb - 1``, while the deepest robust route uses ``2 * Nlimb - 1``
    and ``4 * Nlimb - 3``.  ``deep_topology_sampling=False`` retains the EA40
    root schedule but uses the shallower ``Nlimb``/``2 * Nlimb - 1`` phase
    pair; this is useful for a GPU bulk stage that must validate roots without
    paying for the largest topology trace.  ``radial_strategy="fixed"``
    evaluates every active topology interval once at the requested subdivision
    and avoids adaptive compact/scatter.  With ``certify_topology=False`` only
    the denser selected limb phase is integrated.  This single-pass combination
    is intended for a bounded GPU scheduler whose shallow failures are retried
    by a separately validated fixed-deep stage; direct calls retain adaptive
    integration and the nested topology certificate by default.
    ``fixed_radial_order=47`` selects independent G23/G47 rules when the fixed
    strategy uses one subdivision; the default 31 retains G15/K31.
    """

    if radial_strategy not in ("adaptive", "fixed"):
        raise ValueError("radial_strategy must be 'adaptive' or 'fixed'")
    if fixed_radial_order not in (31, 47):
        raise ValueError("fixed_radial_order must be 31 or 47")
    if radial_chunk_size <= 0:
        raise ValueError("radial_chunk_size must be positive")

    a = 0.5 * s
    e1 = q / (1.0 + q)
    lens_params = {"q": q, "s": s, "a": a, "e1": e1}
    shifted = a * (1.0 - q) / (1.0 + q)
    w_center_shifted = w_center - shifted
    if certify_topology:
        nested_limb_count = (
            4 * Nlimb - 3
            if robust_roots and deep_topology_sampling
            else 2 * Nlimb - 1
        )
        nested_image_limb, nested_mask_limb = calc_source_limb(
            w_center, rho, nested_limb_count, nlenses=2, **lens_params
        )
        image_limb = nested_image_limb[:, ::2]
        mask_limb = nested_mask_limb[:, ::2]
    else:
        selected_limb_count = (
            4 * Nlimb - 3
            if robust_roots and deep_topology_sampling
            else 2 * Nlimb - 1
        )
        image_limb, mask_limb = calc_source_limb(
            w_center,
            rho,
            selected_limb_count,
            nlenses=2,
            **lens_params,
        )
    origin_inside = binary_level_set(
        jnp.asarray(0.0 + 0.0j),
        w_center_shifted,
        rho,
        shifted,
        a=a,
        e1=e1,
    ) <= 0.0
    topology = define_radial_topology(
        image_limb,
        mask_limb,
        rho,
        margin_r=margin_r,
        origin_inside=origin_inside,
        track_roots=track_limb_roots,
        binary_margin_parameters=(
            (shifted, a, e1) if jacobian_radial_margin else None
        ),
    )

    real_dtype, complex_dtype = _integration_dtypes(w_center)
    rho_grid = jnp.asarray(rho, dtype=real_dtype)
    shifted_grid = jnp.asarray(shifted, dtype=real_dtype)
    a_grid = jnp.asarray(a, dtype=real_dtype)
    e1_grid = jnp.asarray(e1, dtype=real_dtype)
    w_center_shifted_grid = jnp.asarray(
        w_center_shifted, dtype=complex_dtype
    )
    angular_atol_grid = jnp.asarray(angular_atol, dtype=real_dtype)
    relative_tolerance_grid = jnp.asarray(
        relative_tolerance, dtype=real_dtype
    )
    output_dtype = jnp.asarray(w_center).real.dtype
    normalization = jnp.pi * rho_grid**2
    cell_tolerance = 64.0 * jnp.finfo(real_dtype).eps

    def radial_integrand(r):
        angular = angular_measure_binary_roots(
            r,
            0.0,
            2.0 * jnp.pi,
            w_center_shifted_grid,
            rho_grid,
            shifted_grid,
            cell_tolerance,
            a=a_grid,
            e1=e1_grid,
            robust_roots=robust_roots,
            propagate_coefficient_padding=not robust_roots,
        )
        return RadialIntegrand(
            r * angular.measure,
            jnp.abs(r) * angular.error,
            angular.status,
        )

    # The bulk EA20 pass retains the conservative node-wise coefficient
    # padding.  EA40 propagates measured residuals node-wise and applies the
    # common x64 floor once below.  The 144-point audit exposed one false EA20
    # acceptance when the correlated model was applied before robust root
    # convergence, so that tempting shortcut is deliberately not used.
    refinement_safety_factor = (
        0.01 if robust_roots and radial_strategy == "adaptive" else 1.0
    )
    def integrate_topology(selected_topology):
        quadrature_chunk_size = (
            selected_topology.intervals.shape[0]
            if parallel_regions
            and selected_topology.intervals.shape[0]
            <= RADIAL_INTERVAL_CAPACITY
            else radial_chunk_size
        )
        integration_options = dict(
            relative_tolerance=(
                refinement_safety_factor * relative_tolerance_grid
            ),
            # A topology-capacity bit must not suppress refinement of the
            # bounded comparison value.  Reattach the structural status after
            # quadrature; only the outer nested certificate may consume it.
            initial_status=RADIAL_OK,
            chunk_size=quadrature_chunk_size,
        )
        if radial_strategy == "fixed":
            integrated = fixed_radial_integral(
                radial_integrand,
                selected_topology.intervals,
                selected_topology.n_intervals,
                refinement_safety_factor * angular_atol_grid * normalization,
                subdivisions=max_radial_subdivisions,
                single_cell_order=fixed_radial_order,
                **integration_options,
            )
        else:
            integrated = adaptive_radial_integral(
                radial_integrand,
                selected_topology.intervals,
                selected_topology.n_intervals,
                refinement_safety_factor * angular_atol_grid * normalization,
                max_subdivisions=max_radial_subdivisions,
                **integration_options,
            )
        return integrated._replace(
            status=jnp.bitwise_or(integrated.status, selected_topology.status)
        )

    radial = integrate_topology(topology)
    topology_error = jnp.asarray(0.0, dtype=real_dtype)
    if certify_topology:
        # The deep retry is also the radial-phase certificate.  It uses a
        # nested source-limb sample and a larger, separately compiled static
        # topology buffer.  The small kernel remains useful even after a
        # capacity event: its bounded coarsened value is compared with the
        # complete large-kernel result, but its capacity bit is never silently
        # treated as success on its own.
        retry_breakpoint_capacity = (
            RADIAL_RETRY_BREAKPOINT_CAPACITY
            if robust_roots
            else RADIAL_INTERVAL_CAPACITY
        )
        retry_interval_capacity = (
            RADIAL_RETRY_INTERVAL_CAPACITY
            if robust_roots
            else RADIAL_INTERVAL_CAPACITY
        )
        nested_topology = define_radial_topology(
            nested_image_limb,
            nested_mask_limb,
            rho,
            margin_r=margin_r,
            origin_inside=origin_inside,
            track_roots=track_limb_roots,
            binary_margin_parameters=(
                (shifted, a, e1) if jacobian_radial_margin else None
            ),
            breakpoint_capacity=retry_breakpoint_capacity,
            interval_capacity=retry_interval_capacity,
        )
        nested_radial = integrate_topology(nested_topology)
        coarse_magnification = radial.value / normalization
        magnification = nested_radial.value / normalization
        topology_error = jnp.abs(
            nested_radial.value - radial.value
        ) / normalization
        # A coarse capacity event is consumed only by this explicit nested
        # comparison.  All other coarse structural bits and every nested
        # structural bit remain fatal.
        coarse_status = jnp.bitwise_and(
            radial.status,
            jnp.bitwise_not(
                jnp.int32(RADIAL_CAPACITY | RADIAL_TOLERANCE)
            ),
        )
        nested_status = jnp.bitwise_and(
            nested_radial.status,
            jnp.bitwise_not(jnp.int32(RADIAL_TOLERANCE)),
        )
        status = jnp.bitwise_or(coarse_status, nested_status)
        radial_error = nested_radial.error
        phase_tolerance = angular_atol_grid + (
            relative_tolerance_grid * jnp.abs(magnification)
        )
        status = jnp.bitwise_or(
            status,
            jnp.where(
                jnp.isfinite(coarse_magnification)
                & jnp.isfinite(magnification)
                & jnp.isfinite(topology_error)
                & (topology_error <= phase_tolerance),
                jnp.int32(RADIAL_OK),
                jnp.int32(RADIAL_TOPOLOGY),
            ),
        )
    else:
        magnification = radial.value / normalization
        radial_error = radial.error
        status = jnp.bitwise_and(
            radial.status, jnp.bitwise_not(jnp.int32(RADIAL_TOLERANCE))
        )
    # Fourier coefficient roundoff is correlated across all radial nodes and
    # must not be accumulated once per G15/K31 evaluation. Near a tangency the
    # worst local root sensitivity scales as sqrt(eps), so retain one global
    # x64 roundoff floor while the node-wise propagation carries measured root
    # residuals and embedded radial disagreement. This is an empirical
    # numerical certificate, not a formal interval-arithmetic bound.
    small_source_weight = jnp.minimum(
        1.0,
        jnp.asarray(3.0e-5, dtype=real_dtype)
        / jnp.maximum(rho_grid, jnp.finfo(real_dtype).tiny),
    )
    roundoff_floor = (
        (1024.0 if robust_roots else 0.0)
        * small_source_weight
        * jnp.sqrt(jnp.finfo(real_dtype).eps)
        * jnp.abs(magnification)
    )
    estimated_error = jnp.maximum(
        jnp.maximum(radial_error / normalization, topology_error),
        roundoff_floor,
    )
    # RADIAL_TOLERANCE above belongs to the deliberately tighter EA40 internal
    # refinement guard. Structural failures remain fatal; the public contract
    # is applied exactly once to the final normalized estimate.
    tolerance = angular_atol_grid + relative_tolerance_grid * jnp.abs(
        magnification
    )
    tolerance_failed = (status == RADIAL_OK) & ~(
        jnp.isfinite(magnification)
        & jnp.isfinite(estimated_error)
        & (estimated_error <= tolerance)
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            tolerance_failed,
            jnp.int32(RADIAL_TOLERANCE),
            jnp.int32(RADIAL_OK),
        ),
    )
    result = BoundaryMagnificationResult(
        jnp.asarray(magnification, dtype=output_dtype),
        jnp.asarray(estimated_error, dtype=output_dtype),
        status,
    )
    if return_info:
        return result

    fatal = (
        status
        & (
            ANGULAR_CAPACITY
            | ANGULAR_DEGENERATE
            | ANGULAR_ROOT_FAILURE
            | RADIAL_CAPACITY
            | RADIAL_TOLERANCE
            | RADIAL_TOPOLOGY
        )
    ) != 0
    return jnp.where(fatal, jnp.nan, result.magnification)


def mag_uniform_local_boundary(
    w_center: complex,
    rho: float,
    *,
    s: float,
    q: float,
    Nlimb: int = 500,
    margin_r: float = 1.0,
    angular_atol: float = 1e-5,
    relative_tolerance: float = 1e-4,
    parallel_regions: bool = False,
    return_info: bool = False,
    max_radial_subdivisions: int = 8,
    refinement_safety_factor: float = 0.01,
    prefer_interior_anchor: bool = False,
    radial_chunk_size: int = _SEQUENTIAL_RADIAL_CHUNK_SIZE,
    radial_strategy: str = "adaptive",
    certify_topology: bool = True,
) -> Union[Array, BoundaryMagnificationResult]:
    """Rescue a uniform binary source in disjoint image-local polar charts.

    The five tracked source-limb root slots are enclosed by fixed-capacity
    disks and overlapping disks are merged.  Each remaining component is
    integrated around its own image-plane center, avoiding the tiny angular
    differences created by a distant planetary image in the global COM frame.
    This is still boundary-area inverse ray shooting, not contour integration.
    ``refinement_safety_factor`` is a static, local-only refinement guard.  A
    value below one asks the embedded radial estimator to refine before the
    public tolerance is exhausted; final acceptance is still tested against
    the caller's original absolute and relative tolerance.
    ``prefer_interior_anchor=True`` selects the alternate fixed-shape origin
    for non-annular fold groups with multiple source-centre images.  The
    high-level scheduler invokes it only for lanes rejected by the normal
    mean-origin local kernel. ``radial_chunk_size`` controls the fixed inner
    batch of radial cells. The direct API defaults to eight for portability;
    the high-level A100 scheduler passes 64 after an explicit width sweep.
    ``radial_strategy="fixed"`` evaluates every active radial interval once at
    ``max_radial_subdivisions`` equal subcells instead of recomputing rejected
    intervals through the adaptive 1/2/4/8/16 schedule. ``certify_topology=False``
    skips the duplicate coarse chart-area calculation. The high-level scheduler
    uses this only inside its empirically validated low-q/small-source gate;
    it is not a formal topology-completeness certificate.
    """

    if not 0.0 < refinement_safety_factor <= 1.0:
        raise ValueError("refinement_safety_factor must be in (0, 1]")
    if radial_chunk_size <= 0:
        raise ValueError("radial_chunk_size must be positive")
    if radial_strategy not in ("adaptive", "fixed"):
        raise ValueError("radial_strategy must be 'adaptive' or 'fixed'")

    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    w_center_shifted = w_center - shifted
    lens_params = {"q": q, "s": s, "a": a, "e1": e1}
    nested_image_limb, nested_mask_limb = calc_source_limb(
        w_center, rho, 2 * Nlimb - 1, nlenses=2, **lens_params
    )
    image_limb = nested_image_limb[:, ::2]
    mask_limb = nested_mask_limb[:, ::2]
    interior_images, interior_mask = _images_point_source(
        w_center_shifted,
        nlenses=2,
        a=a,
        e1=e1,
    )
    if certify_topology:
        charts = build_local_image_charts(
            image_limb,
            mask_limb,
            rho,
            margin_r=margin_r,
            shifted=shifted,
            a=a,
            e1=e1,
            interior_images=interior_images + shifted,
            interior_mask=interior_mask,
            prefer_interior_anchor=prefer_interior_anchor,
        )
    nested_charts = build_local_image_charts(
        nested_image_limb,
        nested_mask_limb,
        rho,
        margin_r=margin_r,
        shifted=shifted,
        a=a,
        e1=e1,
        interior_images=interior_images + shifted,
        interior_mask=interior_mask,
        prefer_interior_anchor=prefer_interior_anchor,
    )

    real_dtype, complex_dtype = _integration_dtypes(w_center)
    rho_grid = jnp.asarray(rho, dtype=real_dtype)
    shifted_grid = jnp.asarray(shifted, dtype=real_dtype)
    a_grid = jnp.asarray(a, dtype=real_dtype)
    e1_grid = jnp.asarray(e1, dtype=real_dtype)
    w_center_shifted_grid = jnp.asarray(
        w_center_shifted, dtype=complex_dtype
    )
    normalization = jnp.pi * rho_grid**2
    angular_atol_grid = jnp.asarray(angular_atol, dtype=real_dtype)
    relative_tolerance_grid = jnp.asarray(
        relative_tolerance, dtype=real_dtype
    )
    def integrate_chart_set(
        selected_charts, breakpoint_capacity, interval_capacity
    ):
        chart_count = jnp.maximum(jnp.sum(selected_charts.active), 1)
        chart_absolute_tolerance = (
            angular_atol_grid * normalization / chart_count
        )
        internal_absolute_tolerance = (
            refinement_safety_factor * chart_absolute_tolerance
        )
        internal_relative_tolerance = (
            refinement_safety_factor * relative_tolerance_grid
        )
        slots = jnp.arange(
            selected_charts.centers.shape[0], dtype=jnp.int32
        )
        first_active = jnp.argmax(selected_charts.active)
        chart_indices = jnp.nonzero(
            selected_charts.active,
            size=selected_charts.centers.shape[0],
            fill_value=first_active,
        )[0]
        safe_centers = selected_charts.centers[chart_indices]
        active_slots = slots < jnp.sum(
            selected_charts.active, dtype=jnp.int32
        )

        def integrate_chart(chart_input):
            chart_index, chart_center = chart_input
            branch_mask = selected_charts.mask_limb & (
                selected_charts.branch_labels[:, None] == chart_index
            )
            origin_inside = binary_level_set(
                chart_center,
                w_center_shifted_grid,
                rho_grid,
                shifted_grid,
                a=a_grid,
                e1=e1_grid,
            ) <= 0.0
            topology = define_radial_topology(
                selected_charts.image_limb,
                branch_mask,
                rho_grid,
                margin_r=margin_r,
                origin_inside=origin_inside,
                track_roots=False,
                binary_margin_parameters=(shifted_grid, a_grid, e1_grid),
                radial_origin=chart_center,
                sampled_turning_points=True,
                filter_roundoff_turning_points=True,
                breakpoint_capacity=breakpoint_capacity,
                interval_capacity=interval_capacity,
            )
            def radial_integrand(local_radius):
                angular = angular_measure_binary_roots(
                    local_radius,
                    0.0,
                    2.0 * jnp.pi,
                    w_center_shifted_grid,
                    rho_grid,
                    shifted_grid,
                    64.0 * jnp.finfo(real_dtype).eps,
                    a=a_grid,
                    e1=e1_grid,
                    robust_roots=True,
                    chart_center=chart_center,
                )
                return RadialIntegrand(
                    local_radius * angular.measure,
                    jnp.abs(local_radius) * angular.error,
                    angular.status,
                )

            quadrature_chunk_size = (
                interval_capacity
                if parallel_regions
                and interval_capacity <= RADIAL_INTERVAL_CAPACITY
                else radial_chunk_size
            )
            if radial_strategy == "fixed":
                integrated = fixed_radial_integral(
                    radial_integrand,
                    topology.intervals,
                    topology.n_intervals,
                    internal_absolute_tolerance,
                    relative_tolerance=internal_relative_tolerance,
                    initial_status=RADIAL_OK,
                    chunk_size=quadrature_chunk_size,
                    subdivisions=max_radial_subdivisions,
                )
            else:
                integrated = adaptive_radial_integral(
                    radial_integrand,
                    topology.intervals,
                    topology.n_intervals,
                    internal_absolute_tolerance,
                    relative_tolerance=internal_relative_tolerance,
                    initial_status=RADIAL_OK,
                    chunk_size=quadrature_chunk_size,
                    max_subdivisions=max_radial_subdivisions,
                )
            return integrated._replace(
                status=jnp.bitwise_or(integrated.status, topology.status)
            )

        def evaluate_chart(chart_input):
            chart_index, chart_center, is_active = chart_input

            def inactive(_):
                zero = jnp.asarray(0.0, dtype=real_dtype)
                return RadialIntegral(zero, zero, zero, jnp.int32(0))

            return jax.lax.cond(
                is_active,
                integrate_chart,
                inactive,
                (chart_index, chart_center),
            )

        # At most five chart representatives exist.  A fixed device map keeps
        # the shape static but, unlike a masked vmap, does not deliberately
        # evaluate the first active chart again in every unused slot.
        chart_results = jax.lax.map(
            evaluate_chart, (chart_indices, safe_centers, active_slots)
        )
        value = jnp.sum(chart_results.value)
        error = jnp.sum(chart_results.error)
        # The per-chart tolerance is an internal refinement guard.  Preserve
        # angular and topology failures and apply the public total error budget
        # after all chart contributions have been summed.
        integration_status = jnp.bitwise_or.reduce(
            jnp.where(
                active_slots,
                jnp.bitwise_and(
                    chart_results.status,
                    jnp.bitwise_not(jnp.int32(RADIAL_TOLERANCE)),
                ),
                jnp.int32(0),
            )
        )
        return value, error, integration_status, selected_charts.status

    value, error, integration_status, chart_status = integrate_chart_set(
        nested_charts,
        RADIAL_LOCAL_RETRY_BREAKPOINT_CAPACITY,
        RADIAL_LOCAL_RETRY_INTERVAL_CAPACITY,
    )
    magnification = value / normalization
    if certify_topology:
        coarse_value, _, coarse_integration_status, coarse_chart_status = (
            integrate_chart_set(
                charts, RADIAL_INTERVAL_CAPACITY, RADIAL_INTERVAL_CAPACITY
            )
        )
        topology_error = jnp.abs(value - coarse_value) / normalization
    else:
        topology_error = jnp.asarray(0.0, dtype=real_dtype)
        coarse_integration_status = jnp.int32(RADIAL_OK)
        coarse_chart_status = jnp.int32(RADIAL_OK)
    estimated_error = jnp.maximum(error / normalization, topology_error)
    tolerance = angular_atol_grid + relative_tolerance_grid * jnp.abs(
        magnification
    )
    # Only a coarse topology-buffer event may be consumed by agreement with
    # the clean larger nested kernel.  Chart overlap/emptiness and every nested
    # capacity event remain fatal.
    status = jnp.bitwise_or(
        coarse_chart_status,
        jnp.bitwise_or(
            jnp.bitwise_and(
                coarse_integration_status,
                jnp.bitwise_not(jnp.int32(RADIAL_CAPACITY)),
            ),
            jnp.bitwise_or(chart_status, integration_status),
        ),
    )
    if certify_topology:
        status = jnp.bitwise_or(
            status,
            jnp.where(
                jnp.isfinite(topology_error) & (topology_error <= tolerance),
                jnp.int32(RADIAL_OK),
                jnp.int32(RADIAL_TOPOLOGY),
            ),
        )
    tolerance_failed = (status == RADIAL_OK) & ~(
        jnp.isfinite(magnification)
        & jnp.isfinite(estimated_error)
        & (estimated_error <= tolerance)
    )
    status = jnp.bitwise_or(
        status,
        jnp.where(
            tolerance_failed,
            jnp.int32(RADIAL_TOLERANCE),
            jnp.int32(RADIAL_OK),
        ),
    )
    output_dtype = jnp.asarray(w_center).real.dtype
    result = BoundaryMagnificationResult(
        jnp.asarray(magnification, dtype=output_dtype),
        jnp.asarray(estimated_error, dtype=output_dtype),
        status,
    )
    if return_info:
        return result
    return jnp.where(status == RADIAL_OK, result.magnification, jnp.nan)


def mag_radial_profile_boundary(
    w_center: complex,
    rho: float,
    radial_intensity: Callable[[Array], Array],
    intensity_flux: float,
    *,
    s: float,
    q: float,
    q3: Optional[float] = None,
    r3: Optional[float] = None,
    psi: Optional[float] = None,
    nlenses: int = 2,
    Nlimb: int = 500,
    margin_r: float = 0.5,
    angular_atol: float = 1e-5,
    relative_tolerance: float = 1e-4,
    parallel_regions: bool = False,
    endpoint_value_bound: Optional[float] = None,
    return_info: bool = False,
    track_limb_roots: bool = True,
    jacobian_radial_margin: bool = True,
    max_radial_subdivisions: int = 8,
    robust_roots: bool = True,
    radial_strategy: str = "adaptive",
    certify_topology: bool = True,
    radial_chunk_size: int = _SEQUENTIAL_RADIAL_CHUNK_SIZE,
    angular_profile_subdivisions: int = 1,
) -> Union[Array, BoundaryMagnificationResult]:
    """Axisymmetric finite-source profile without a dense angular resolution.

    The outer source boundary supplies the same topology-preserving radial
    cells as :func:`mag_uniform_boundary`. On each image-plane ring, validated
    boundary roots produce at most four binary or five triple linear inside
    intervals. The
    callable ``radial_intensity(d / rho)`` is integrated directly over those
    intervals with a two-sided sine-squared map and embedded G15/K31 quadrature.
    ``intensity_flux`` must be its positive dimensionless unlensed disk flux,
    ``2*pi*integral_0^1 x*radial_intensity(x) dx``.  This explicit normalization
    keeps the core applicable to linear, quadratic, tabulated, and other smooth
    radial profiles without imposing a particular parameterization.

    For binary lenses, ``relative_tolerance`` applies the same
    ``atol + rtol * abs(magnification)`` acceptance rule as the uniform
    backend. Triple lenses retain their absolute-only radial criterion while
    their radial strategy is evaluated separately.

    ``endpoint_value_bound`` defaults to the absolute intensity at the source
    limb and propagates boundary-root uncertainty into the error estimate.
    Adjacent source-limb roots are matched by default before constructing the
    radial topology; disabling ``track_limb_roots`` is diagnostic only.
    ``jacobian_radial_margin`` applies the same branch-wise image-motion guard
    used by the uniform backend. Binary and triple lenses share this profile
    integration; no brightness-specific stack of uniform disks is introduced.
    Binary direct calls use the robust 40-step fixed root schedule by default;
    ``robust_roots=False`` is intended for the high-throughput first pass of a
    caller that independently retries failed points.
    For binary lenses, ``radial_strategy="fixed"`` evaluates each selected
    topology interval once at a fixed fine subdivision. With
    ``certify_topology=False`` only the denser nested topology is integrated;
    this removes the duplicate area calculation and is intended for a bounded
    fixed-grid scheduler with an external exact-reference validation matrix.
    """

    if radial_strategy not in ("adaptive", "fixed"):
        raise ValueError("radial_strategy must be 'adaptive' or 'fixed'")
    if radial_chunk_size <= 0:
        raise ValueError("radial_chunk_size must be positive")
    if nlenses != 2 and (
        radial_strategy != "adaptive" or not certify_topology
    ):
        raise ValueError(
            "fixed/single-topology radial profile is currently binary-only"
        )

    if nlenses == 2:
        a = 0.5 * s
        e1 = q / (1.0 + q)
        lens_params = {"q": q, "s": s, "a": a, "e1": e1}
        shifted = a * (1.0 - q) / (1.0 + q)
        lens_margin_parameters = None
        binary_margin_parameters = (
            (shifted, a, e1) if jacobian_radial_margin else None
        )
    elif nlenses == 3:
        triple_params = {"q3": q3, "r3": r3, "psi": psi}
        missing = [
            name for name, value in triple_params.items() if value is None
        ]
        if missing:
            raise ValueError(
                "missing triple-lens parameters: " + ", ".join(missing)
            )
        geometry = triple_lens_geometry(s, q, q3, r3, psi)
        a, e1, e2 = geometry.a, geometry.e1, geometry.e2
        shifted = geometry.shifted
        lens_params = {
            "s": s,
            "q": q,
            "q3": q3,
            "r3": r3,
            "psi": psi,
            "a": a,
            "e1": e1,
            "e2": e2,
        }
        binary_margin_parameters = None
        lens_margin_parameters = (
            (
                shifted,
                jnp.asarray([a, -a, geometry.r3_complex]),
                jnp.asarray([e1, e2, geometry.e3]),
            )
            if jacobian_radial_margin
            else None
        )
    else:
        raise NotImplementedError(
            "mag_radial_profile_boundary supports binary and triple lenses"
        )

    w_center_shifted = w_center - shifted
    if nlenses == 2:
        nested_limb_count = (
            4 * Nlimb - 3 if robust_roots else 2 * Nlimb - 1
        )
        nested_image_limb, nested_mask_limb = calc_source_limb(
            w_center, rho, nested_limb_count, nlenses=2, **lens_params
        )
        image_limb = nested_image_limb[:, ::2]
        mask_limb = nested_mask_limb[:, ::2]
    else:
        image_limb, mask_limb = calc_source_limb(
            w_center, rho, Nlimb, nlenses=nlenses, **lens_params
        )
    if nlenses == 2:
        origin_inside = binary_level_set(
            jnp.asarray(0.0 + 0.0j),
            w_center_shifted,
            rho,
            shifted,
            a=a,
            e1=e1,
        ) <= 0.0
    else:
        origin_inside = triple_level_set(
            jnp.asarray(0.0 + 0.0j),
            w_center_shifted,
            rho,
            shifted,
            a=a,
            e1=e1,
            e2=e2,
            r3_complex=geometry.r3_complex,
        ) <= 0.0
    topology = define_radial_topology(
        image_limb,
        mask_limb,
        rho,
        margin_r=margin_r,
        origin_inside=origin_inside,
        track_roots=track_limb_roots,
        binary_margin_parameters=binary_margin_parameters,
        lens_margin_parameters=lens_margin_parameters,
    )

    real_dtype, complex_dtype = _integration_dtypes(w_center)
    rho_grid = jnp.asarray(rho, dtype=real_dtype)
    intensity_flux_grid = jnp.asarray(intensity_flux, dtype=real_dtype)
    valid_flux = jnp.isfinite(intensity_flux_grid) & (intensity_flux_grid > 0.0)
    safe_intensity_flux_grid = jnp.where(
        valid_flux, intensity_flux_grid, jnp.asarray(1.0, dtype=real_dtype)
    )
    shifted_grid = jnp.asarray(
        shifted, dtype=real_dtype if nlenses == 2 else complex_dtype
    )
    a_grid = jnp.asarray(a, dtype=real_dtype)
    e1_grid = jnp.asarray(e1, dtype=real_dtype)
    if nlenses == 3:
        e2_grid = jnp.asarray(e2, dtype=real_dtype)
        r3_grid = jnp.asarray(r3, dtype=real_dtype)
        psi_grid = jnp.asarray(psi, dtype=real_dtype)
        r3_complex_grid = jnp.asarray(
            geometry.r3_complex, dtype=complex_dtype
        )
    w_center_shifted_grid = jnp.asarray(
        w_center_shifted, dtype=complex_dtype
    )
    angular_atol_grid = jnp.asarray(angular_atol, dtype=real_dtype)
    relative_tolerance_grid = jnp.asarray(
        relative_tolerance if nlenses == 2 else 0.0,
        dtype=real_dtype,
    )
    output_dtype = jnp.asarray(w_center).real.dtype
    normalization = rho_grid**2 * safe_intensity_flux_grid
    cell_tolerance = 64.0 * jnp.finfo(real_dtype).eps
    if endpoint_value_bound is None:
        endpoint_intensity = jnp.abs(
            radial_intensity(jnp.asarray(1.0, dtype=real_dtype))
        )
    else:
        endpoint_intensity = jnp.asarray(endpoint_value_bound, dtype=real_dtype)

    def radial_integrand(r):
        if nlenses == 2:
            intervals = angular_intervals_binary_roots(
                r,
                0.0,
                2.0 * jnp.pi,
                w_center_shifted_grid,
                rho_grid,
                shifted_grid,
                cell_tolerance,
                a=a_grid,
                e1=e1_grid,
                robust_roots=robust_roots,
                propagate_coefficient_padding=not robust_roots,
            )
        else:
            intervals = angular_intervals_triple_roots(
                r,
                0.0,
                2.0 * jnp.pi,
                w_center_shifted_grid,
                rho_grid,
                shifted_grid,
                cell_tolerance,
                a=a_grid,
                e1=e1_grid,
                e2=e2_grid,
                r3_complex=r3_complex_grid,
            )

        def brightness(theta):
            if nlenses == 2:
                distance = distance_from_source(
                    r,
                    theta,
                    w_center_shifted_grid,
                    shifted_grid,
                    nlenses=2,
                    a=a_grid,
                    e1=e1_grid,
                )
            else:
                distance = distance_from_source(
                    r,
                    theta,
                    w_center_shifted_grid,
                    shifted_grid,
                    nlenses=3,
                    a=a_grid,
                    e1=e1_grid,
                    e2=e2_grid,
                    r3=r3_grid,
                    psi=psi_grid,
                )
            distance = jnp.nan_to_num(
                distance,
                nan=jnp.inf,
                posinf=jnp.inf,
                neginf=jnp.inf,
            )
            return radial_intensity(distance / rho_grid)

        angular = integrate_angular_profile(
            brightness,
            intervals,
            endpoint_value_bound=endpoint_intensity,
            subdivisions=angular_profile_subdivisions,
        )
        return RadialIntegrand(
            r * angular.value,
            jnp.abs(r) * angular.error,
            angular.status,
        )

    flux_status = jnp.where(
        valid_flux,
        jnp.int32(0),
        jnp.int32(ANGULAR_ROOT_FAILURE),
    )

    def integrate_topology(selected_topology):
        quadrature_chunk_size = (
            selected_topology.intervals.shape[0]
            if parallel_regions
            and selected_topology.intervals.shape[0]
            <= RADIAL_INTERVAL_CAPACITY
            else radial_chunk_size
        )
        if radial_strategy == "fixed":
            integrated = fixed_radial_integral(
                radial_integrand,
                selected_topology.intervals,
                selected_topology.n_intervals,
                angular_atol_grid * normalization,
                relative_tolerance=relative_tolerance_grid,
                initial_status=flux_status,
                chunk_size=quadrature_chunk_size,
                subdivisions=max_radial_subdivisions,
            )
        else:
            integrated = adaptive_radial_integral(
                radial_integrand,
                selected_topology.intervals,
                selected_topology.n_intervals,
                angular_atol_grid * normalization,
                relative_tolerance=relative_tolerance_grid,
                initial_status=flux_status,
                chunk_size=quadrature_chunk_size,
                max_subdivisions=max_radial_subdivisions,
            )
        return integrated._replace(
            status=jnp.bitwise_or(integrated.status, selected_topology.status)
        )

    if nlenses != 2 or certify_topology:
        radial = integrate_topology(topology)
        magnification = radial.value / normalization
        estimated_error = radial.error / normalization
        status = radial.status
    if nlenses == 2:
        retry_breakpoint_capacity = (
            RADIAL_RETRY_BREAKPOINT_CAPACITY
            if robust_roots
            else RADIAL_INTERVAL_CAPACITY
        )
        retry_interval_capacity = (
            RADIAL_RETRY_INTERVAL_CAPACITY
            if robust_roots
            else RADIAL_INTERVAL_CAPACITY
        )
        nested_topology = define_radial_topology(
            nested_image_limb,
            nested_mask_limb,
            rho,
            margin_r=margin_r,
            origin_inside=origin_inside,
            track_roots=track_limb_roots,
            binary_margin_parameters=binary_margin_parameters,
            breakpoint_capacity=retry_breakpoint_capacity,
            interval_capacity=retry_interval_capacity,
        )
        nested_radial = integrate_topology(nested_topology)
        if certify_topology:
            coarse_magnification = magnification
            magnification = nested_radial.value / normalization
            topology_error = jnp.abs(
                nested_radial.value - radial.value
            ) / normalization
            estimated_error = jnp.maximum(
                nested_radial.error / normalization, topology_error
            )
            coarse_status = jnp.bitwise_and(
                radial.status,
                jnp.bitwise_not(
                    jnp.int32(RADIAL_CAPACITY | RADIAL_TOLERANCE)
                ),
            )
            status = jnp.bitwise_or(coarse_status, nested_radial.status)
        else:
            magnification = nested_radial.value / normalization
            topology_error = jnp.asarray(0.0, dtype=real_dtype)
            estimated_error = nested_radial.error / normalization
            status = nested_radial.status
        tolerance = angular_atol_grid + (
            relative_tolerance_grid * jnp.abs(magnification)
        )
        if certify_topology:
            status = jnp.bitwise_or(
                status,
                jnp.where(
                    jnp.isfinite(coarse_magnification)
                    & jnp.isfinite(magnification)
                    & jnp.isfinite(topology_error)
                    & (topology_error <= tolerance),
                    jnp.int32(RADIAL_OK),
                    jnp.int32(RADIAL_TOPOLOGY),
                ),
            )
        tolerance_failed = (status == RADIAL_OK) & ~(
            jnp.isfinite(estimated_error)
            & (estimated_error <= tolerance)
        )
        status = jnp.bitwise_or(
            status,
            jnp.where(
                tolerance_failed,
                jnp.int32(RADIAL_TOLERANCE),
                jnp.int32(RADIAL_OK),
            ),
        )
    result = BoundaryMagnificationResult(
        jnp.asarray(
            jnp.where(valid_flux, magnification, jnp.nan),
            dtype=output_dtype,
        ),
        jnp.asarray(
            jnp.where(valid_flux, estimated_error, jnp.nan),
            dtype=output_dtype,
        ),
        status,
    )
    if return_info:
        return result

    fatal = (
        result.status
        & (
            ANGULAR_CAPACITY
            | ANGULAR_DEGENERATE
            | ANGULAR_ROOT_FAILURE
            | RADIAL_CAPACITY
            | RADIAL_TOLERANCE
            | RADIAL_TOPOLOGY
        )
    ) != 0
    return jnp.where(fatal, jnp.nan, result.magnification)


def mag_limb_dark_boundary(
    w_center: complex,
    rho: float,
    *,
    s: float,
    q: float,
    q3: Optional[float] = None,
    r3: Optional[float] = None,
    psi: Optional[float] = None,
    nlenses: int = 2,
    u1: float = 0.0,
    Nlimb: int = 500,
    margin_r: float = 0.5,
    angular_atol: float = 1e-5,
    relative_tolerance: float = 1e-4,
    parallel_regions: bool = False,
    return_info: bool = False,
    track_limb_roots: bool = True,
    jacobian_radial_margin: bool = True,
    max_radial_subdivisions: int = 8,
    robust_roots: bool = True,
    radial_strategy: str = "adaptive",
    certify_topology: bool = True,
    radial_chunk_size: int = _SEQUENTIAL_RADIAL_CHUNK_SIZE,
    angular_profile_subdivisions: int = 1,
) -> Union[Array, BoundaryMagnificationResult]:
    """Linear limb-darkening through the generic radial-profile backend."""

    u1_array = jnp.asarray(u1)

    def intensity(distance_over_rho):
        return linear_limb_intensity(distance_over_rho, u1=u1_array)

    return mag_radial_profile_boundary(
        w_center,
        rho,
        intensity,
        1.0,
        s=s,
        q=q,
        q3=q3,
        r3=r3,
        psi=psi,
        nlenses=nlenses,
        Nlimb=Nlimb,
        margin_r=margin_r,
        angular_atol=angular_atol,
        relative_tolerance=relative_tolerance,
        parallel_regions=parallel_regions,
        track_limb_roots=track_limb_roots,
        jacobian_radial_margin=jacobian_radial_margin,
        max_radial_subdivisions=max_radial_subdivisions,
        robust_roots=robust_roots,
        radial_strategy=radial_strategy,
        certify_topology=certify_topology,
        radial_chunk_size=radial_chunk_size,
        angular_profile_subdivisions=angular_profile_subdivisions,
        return_info=return_info,
    )
