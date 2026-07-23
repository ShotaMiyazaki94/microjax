"""Uniform binary-source boundary integration."""

from typing import Union

import jax.numpy as jnp

from ..geometry.limb import calc_source_limb
from ..geometry.lens import binary_geometry
from ..geometry.topology import (
    RADIAL_CAPACITY,
    RADIAL_INTERVAL_CAPACITY,
    RADIAL_OK,
    RADIAL_RETRY_BREAKPOINT_CAPACITY,
    RADIAL_RETRY_INTERVAL_CAPACITY,
    RADIAL_TOLERANCE,
    RADIAL_TOPOLOGY,
    define_radial_topology,
)
from ..quadrature.radial import RadialIntegrand, adaptive_radial_integral, fixed_radial_integral
from ..roots.angular import (
    ANGULAR_CAPACITY,
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
    angular_measure_binary_roots,
)
from ..roots.level_set import binary_level_set
from .charts import _planetary_mixed_topology
from .common import (
    Array,
    BoundaryMagnificationResult,
    SEQUENTIAL_RADIAL_CHUNK_SIZE,
    integration_dtypes,
    unwrap_boundary_result,
)


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
    radial_chunk_size: int = SEQUENTIAL_RADIAL_CHUNK_SIZE,
    fixed_radial_order: int = 31,
    _planetary_local_chart: bool = False,
) -> Union[Array, BoundaryMagnificationResult]:
    """Integrate a uniform binary source from exact angular boundary roots.

    Nlimb controls source-limb topology tracing, not quadrature density.
    Radial cells are adaptive by default; radial_strategy="fixed" provides
    the bounded GPU scheduler used by the light-curve API. The optional
    planetary chart is valid only for fixed, uncertified single-pass calls.

    With return_info=True, return magnification, error, and status bits.
    """

    if radial_strategy not in ("adaptive", "fixed"):
        raise ValueError("radial_strategy must be 'adaptive' or 'fixed'")
    if fixed_radial_order not in (31, 47):
        raise ValueError("fixed_radial_order must be 31 or 47")
    if radial_chunk_size <= 0:
        raise ValueError("radial_chunk_size must be positive")
    if _planetary_local_chart and (radial_strategy != "fixed" or certify_topology):
        raise ValueError("the planetary chart requires fixed, uncertified radial integration")

    lens = binary_geometry(s, q)
    a, e1, shifted = lens.a, lens.e1, lens.shifted
    lens_params = {"q": q, "s": s, "a": a, "e1": e1}
    w_center_shifted = w_center - shifted
    if certify_topology:
        nested_limb_count = 4 * Nlimb - 3 if robust_roots and deep_topology_sampling else 2 * Nlimb - 1
        nested_image_limb, nested_mask_limb = calc_source_limb(
            w_center, rho, nested_limb_count, nlenses=2, **lens_params
        )
        image_limb = nested_image_limb[:, ::2]
        mask_limb = nested_mask_limb[:, ::2]
    else:
        selected_limb_count = 4 * Nlimb - 3 if robust_roots and deep_topology_sampling else 2 * Nlimb - 1
        image_limb, mask_limb = calc_source_limb(
            w_center,
            rho,
            selected_limb_count,
            nlenses=2,
            **lens_params,
        )
    origin_inside = (
        binary_level_set(
            jnp.asarray(0.0 + 0.0j),
            w_center_shifted,
            rho,
            shifted,
            a=a,
            e1=e1,
        )
        <= 0.0
    )
    interval_centers = None
    if _planetary_local_chart:
        topology, interval_centers = _planetary_mixed_topology(
            image_limb,
            mask_limb,
            rho,
            margin_r=margin_r,
            lens=lens,
            w_center_shifted=w_center_shifted,
            origin_inside=origin_inside,
            jacobian_radial_margin=jacobian_radial_margin,
        )
    else:
        topology = define_radial_topology(
            image_limb,
            mask_limb,
            rho,
            margin_r=margin_r,
            origin_inside=origin_inside,
            track_roots=track_limb_roots,
            binary_margin_parameters=((shifted, a, e1) if jacobian_radial_margin else None),
        )

    real_dtype, complex_dtype = integration_dtypes(w_center)
    rho_grid = jnp.asarray(rho, dtype=real_dtype)
    shifted_grid = jnp.asarray(shifted, dtype=real_dtype)
    a_grid = jnp.asarray(a, dtype=real_dtype)
    e1_grid = jnp.asarray(e1, dtype=real_dtype)
    w_center_shifted_grid = jnp.asarray(w_center_shifted, dtype=complex_dtype)
    angular_atol_grid = jnp.asarray(angular_atol, dtype=real_dtype)
    relative_tolerance_grid = jnp.asarray(relative_tolerance, dtype=real_dtype)
    output_dtype = jnp.asarray(w_center).real.dtype
    normalization = jnp.pi * rho_grid**2
    cell_tolerance = 64.0 * jnp.finfo(real_dtype).eps

    def radial_integrand(r, chart_center=0.0 + 0.0j):
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
            chart_center=chart_center,
            propagate_coefficient_padding=not robust_roots,
        )
        return RadialIntegrand(
            r * angular.measure,
            jnp.abs(r) * angular.error,
            angular.status,
        )

    # The bulk EA32 pass retains the conservative node-wise coefficient
    # padding.  EA40 propagates measured residuals node-wise and applies the
    # common x64 floor once below.  A 144-point audit exposed one false
    # acceptance when the correlated model was applied before robust root
    # convergence, so that tempting shortcut is deliberately not used.
    refinement_safety_factor = 0.01 if robust_roots and radial_strategy == "adaptive" else 1.0

    def integrate_topology(selected_topology):
        quadrature_chunk_size = (
            selected_topology.intervals.shape[0]
            if parallel_regions and selected_topology.intervals.shape[0] <= RADIAL_INTERVAL_CAPACITY
            else radial_chunk_size
        )
        integration_options = dict(
            relative_tolerance=(refinement_safety_factor * relative_tolerance_grid),
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
                interval_parameters=interval_centers,
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
        return integrated._replace(status=jnp.bitwise_or(integrated.status, selected_topology.status))

    radial = integrate_topology(topology)
    topology_error = jnp.asarray(0.0, dtype=real_dtype)
    if certify_topology:
        # The deep retry is also the radial-phase certificate.  It uses a
        # nested source-limb sample and a larger, separately compiled static
        # topology buffer.  The small kernel remains useful even after a
        # capacity event: its bounded coarsened value is compared with the
        # complete large-kernel result, but its capacity bit is never silently
        # treated as success on its own.
        retry_breakpoint_capacity = RADIAL_RETRY_BREAKPOINT_CAPACITY if robust_roots else RADIAL_INTERVAL_CAPACITY
        retry_interval_capacity = RADIAL_RETRY_INTERVAL_CAPACITY if robust_roots else RADIAL_INTERVAL_CAPACITY
        nested_topology = define_radial_topology(
            nested_image_limb,
            nested_mask_limb,
            rho,
            margin_r=margin_r,
            origin_inside=origin_inside,
            track_roots=track_limb_roots,
            binary_margin_parameters=((shifted, a, e1) if jacobian_radial_margin else None),
            breakpoint_capacity=retry_breakpoint_capacity,
            interval_capacity=retry_interval_capacity,
        )
        nested_radial = integrate_topology(nested_topology)
        coarse_magnification = radial.value / normalization
        magnification = nested_radial.value / normalization
        topology_error = jnp.abs(nested_radial.value - radial.value) / normalization
        # A coarse capacity event is consumed only by this explicit nested
        # comparison.  All other coarse structural bits and every nested
        # structural bit remain fatal.
        coarse_status = jnp.bitwise_and(
            radial.status,
            jnp.bitwise_not(jnp.int32(RADIAL_CAPACITY | RADIAL_TOLERANCE)),
        )
        nested_status = jnp.bitwise_and(
            nested_radial.status,
            jnp.bitwise_not(jnp.int32(RADIAL_TOLERANCE)),
        )
        status = jnp.bitwise_or(coarse_status, nested_status)
        radial_error = nested_radial.error
        phase_tolerance = angular_atol_grid + (relative_tolerance_grid * jnp.abs(magnification))
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
        status = jnp.bitwise_and(radial.status, jnp.bitwise_not(jnp.int32(RADIAL_TOLERANCE)))
    # Fourier coefficient roundoff is correlated across all radial nodes and
    # must not be accumulated once per G15/K31 evaluation. Near a tangency the
    # worst local root sensitivity scales as sqrt(eps), so retain one global
    # x64 roundoff floor while the node-wise propagation carries measured root
    # residuals and embedded radial disagreement. This is an empirical
    # numerical certificate, not a formal interval-arithmetic bound.
    small_source_weight = jnp.minimum(
        1.0,
        jnp.asarray(3.0e-5, dtype=real_dtype) / jnp.maximum(rho_grid, jnp.finfo(real_dtype).tiny),
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
    tolerance = angular_atol_grid + relative_tolerance_grid * jnp.abs(magnification)
    tolerance_failed = (status == RADIAL_OK) & ~(
        jnp.isfinite(magnification) & jnp.isfinite(estimated_error) & (estimated_error <= tolerance)
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
    fatal_statuses = (
        ANGULAR_CAPACITY
        | ANGULAR_DEGENERATE
        | ANGULAR_ROOT_FAILURE
        | RADIAL_CAPACITY
        | RADIAL_TOLERANCE
        | RADIAL_TOPOLOGY
    )
    return unwrap_boundary_result(result, return_info, fatal_statuses)
