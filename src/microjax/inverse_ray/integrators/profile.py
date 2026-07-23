"""Boundary integration for radial and linear limb-darkening profiles."""

from typing import Callable, Optional, Union

import jax.numpy as jnp
from jax import lax

from microjax.lens_geometry import triple_lens_geometry
from ..geometry.limb import calc_source_limb
from ..geometry.lens import binary_geometry
from ..geometry.mapping import distance_from_source
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
from ..quadrature.angular import integrate_angular_profile
from ..quadrature.radial import RadialIntegrand, adaptive_radial_integral, fixed_radial_integral
from ..roots.angular import (
    ANGULAR_CAPACITY,
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
    angular_intervals_binary_roots,
    angular_intervals_triple_roots,
)
from ..roots.level_set import binary_level_set, triple_level_set
from .charts import _owned_angular_intervals, _planetary_mixed_topology, _triple_compact_mixed_topology
from .common import (
    Array,
    BoundaryMagnificationResult,
    SEQUENTIAL_RADIAL_CHUNK_SIZE,
    integration_dtypes,
    unwrap_boundary_result,
)


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
    radial_chunk_size: int = SEQUENTIAL_RADIAL_CHUNK_SIZE,
    angular_profile_subdivisions: int = 1,
    _planetary_local_chart: bool = False,
    _compact_local_chart: bool = False,
) -> Union[Array, BoundaryMagnificationResult]:
    """Integrate an axisymmetric profile over boundary-root intervals.

    radial_intensity(d / rho) is evaluated directly in each image interval;
    intensity_flux is its positive unlensed disk flux. Binary calls support
    adaptive/fixed radial strategies. Optional binary planetary and generic
    triple compact-image charts retain one fixed-shape integration graph.

    With return_info=True, return magnification, error, and status bits.
    """

    if radial_strategy not in ("adaptive", "fixed"):
        raise ValueError("radial_strategy must be 'adaptive' or 'fixed'")
    if radial_chunk_size <= 0:
        raise ValueError("radial_chunk_size must be positive")
    if _planetary_local_chart and (nlenses != 2 or radial_strategy != "fixed" or certify_topology):
        raise ValueError(
            "the single-pass planetary chart requires binary fixed, " "uncertified radial-profile integration"
        )
    if _compact_local_chart and (nlenses != 3 or radial_strategy != "fixed" or certify_topology):
        raise ValueError("the compact-image chart requires triple fixed, uncertified radial-profile integration")

    if nlenses == 2:
        binary_lens = binary_geometry(s, q)
        a, e1, shifted = binary_lens.a, binary_lens.e1, binary_lens.shifted
        lens_params = {"q": q, "s": s, "a": a, "e1": e1}
        lens_margin_parameters = None
        binary_margin_parameters = (shifted, a, e1) if jacobian_radial_margin else None
    elif nlenses == 3:
        triple_params = {"q3": q3, "r3": r3, "psi": psi}
        missing = [name for name, value in triple_params.items() if value is None]
        if missing:
            raise ValueError("missing triple-lens parameters: " + ", ".join(missing))
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
        raise NotImplementedError("mag_radial_profile_boundary supports binary and triple lenses")

    w_center_shifted = w_center - shifted
    if nlenses == 2:
        nested_limb_count = 4 * Nlimb - 3 if robust_roots else 2 * Nlimb - 1
        nested_image_limb, nested_mask_limb = calc_source_limb(
            w_center, rho, nested_limb_count, nlenses=2, **lens_params
        )
        if certify_topology:
            image_limb = nested_image_limb[:, ::2]
            mask_limb = nested_mask_limb[:, ::2]
    else:
        image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, nlenses=nlenses, **lens_params)
    if nlenses == 2:
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
    else:
        origin_inside = (
            triple_level_set(
                jnp.asarray(0.0 + 0.0j),
                w_center_shifted,
                rho,
                shifted,
                a=a,
                e1=e1,
                e2=e2,
                r3_complex=geometry.r3_complex,
            )
            <= 0.0
        )
    interval_parameters = None
    charted = None
    if nlenses != 2 or certify_topology:
        if _compact_local_chart:
            charted = _triple_compact_mixed_topology(
                image_limb,
                mask_limb,
                rho,
                margin_r=margin_r,
                w_center_shifted=w_center_shifted,
                origin_inside=origin_inside,
                shifted=shifted,
                a=a,
                e1=e1,
                e2=e2,
                r3_complex=geometry.r3_complex,
                lens_margin_parameters=lens_margin_parameters,
            )
            topology = charted.topology
            interval_parameters = charted.interval_parameters
        elif _planetary_local_chart:
            topology, interval_parameters = _planetary_mixed_topology(
                image_limb,
                mask_limb,
                rho,
                margin_r=margin_r,
                lens=binary_lens,
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
                binary_margin_parameters=binary_margin_parameters,
                lens_margin_parameters=lens_margin_parameters,
            )

    if nlenses == 3:
        # Triple-lens support bounds are padded into zero-measure regions.
        # Their sampled min/max motion has no exact boundary contribution and
        # only differentiates fixed-quadrature truncation error near caustics.
        topology = topology._replace(intervals=lax.stop_gradient(topology.intervals))

    real_dtype, complex_dtype = integration_dtypes(w_center)
    rho_grid = jnp.asarray(rho, dtype=real_dtype)
    intensity_flux_grid = jnp.asarray(intensity_flux, dtype=real_dtype)
    valid_flux = jnp.isfinite(intensity_flux_grid) & (intensity_flux_grid > 0.0)
    safe_intensity_flux_grid = jnp.where(valid_flux, intensity_flux_grid, jnp.asarray(1.0, dtype=real_dtype))
    shifted_grid = jnp.asarray(shifted, dtype=real_dtype if nlenses == 2 else complex_dtype)
    a_grid = jnp.asarray(a, dtype=real_dtype)
    e1_grid = jnp.asarray(e1, dtype=real_dtype)
    if nlenses == 3:
        e2_grid = jnp.asarray(e2, dtype=real_dtype)
        r3_grid = jnp.asarray(r3, dtype=real_dtype)
        psi_grid = jnp.asarray(psi, dtype=real_dtype)
        r3_complex_grid = jnp.asarray(geometry.r3_complex, dtype=complex_dtype)
    w_center_shifted_grid = jnp.asarray(w_center_shifted, dtype=complex_dtype)
    angular_atol_grid = jnp.asarray(angular_atol, dtype=real_dtype)
    relative_tolerance_grid = jnp.asarray(relative_tolerance, dtype=real_dtype)
    output_dtype = jnp.asarray(w_center).real.dtype
    normalization = rho_grid**2 * safe_intensity_flux_grid
    cell_tolerance = 64.0 * jnp.finfo(real_dtype).eps
    if endpoint_value_bound is None:
        endpoint_intensity = jnp.abs(radial_intensity(jnp.asarray(1.0, dtype=real_dtype)))
    else:
        endpoint_intensity = jnp.asarray(endpoint_value_bound, dtype=real_dtype)

    def radial_integrand(r, interval_parameter=0.0 + 0.0j):
        chart_center = interval_parameter[0] if _compact_local_chart else interval_parameter
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
                chart_center=chart_center,
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
                chart_center=chart_center,
            )
            if _compact_local_chart:
                owned, n_owned = _owned_angular_intervals(
                    intervals.intervals,
                    intervals.n_intervals,
                    r,
                    interval_parameter,
                    charted,
                )
                intervals = intervals._replace(intervals=owned, n_intervals=n_owned)

        def brightness(theta):
            if nlenses == 2:
                distance = distance_from_source(
                    r,
                    theta,
                    w_center_shifted_grid,
                    shifted_grid,
                    nlenses=2,
                    chart_center=chart_center,
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
                    chart_center=chart_center,
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

    def integrate_topology(selected_topology, selected_centers=None):
        quadrature_chunk_size = (
            selected_topology.intervals.shape[0]
            if parallel_regions and selected_topology.intervals.shape[0] <= RADIAL_INTERVAL_CAPACITY
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
                interval_parameters=selected_centers,
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
        return integrated._replace(status=jnp.bitwise_or(integrated.status, selected_topology.status))

    if nlenses != 2 or certify_topology:
        radial = integrate_topology(topology, interval_parameters)
        magnification = radial.value / normalization
        estimated_error = radial.error / normalization
        status = radial.status
    if nlenses == 2:
        retry_breakpoint_capacity = RADIAL_RETRY_BREAKPOINT_CAPACITY if robust_roots else RADIAL_INTERVAL_CAPACITY
        retry_interval_capacity = RADIAL_RETRY_INTERVAL_CAPACITY if robust_roots else RADIAL_INTERVAL_CAPACITY
        nested_centers = None
        if _planetary_local_chart:
            nested_topology, nested_centers = _planetary_mixed_topology(
                nested_image_limb,
                nested_mask_limb,
                rho,
                margin_r=margin_r,
                lens=binary_lens,
                w_center_shifted=w_center_shifted,
                origin_inside=origin_inside,
                jacobian_radial_margin=jacobian_radial_margin,
            )
        else:
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
        nested_radial = integrate_topology(nested_topology, nested_centers)
        if certify_topology:
            coarse_magnification = magnification
            magnification = nested_radial.value / normalization
            topology_error = jnp.abs(nested_radial.value - radial.value) / normalization
            estimated_error = jnp.maximum(nested_radial.error / normalization, topology_error)
            coarse_status = jnp.bitwise_and(
                radial.status,
                jnp.bitwise_not(jnp.int32(RADIAL_CAPACITY | RADIAL_TOLERANCE)),
            )
            status = jnp.bitwise_or(coarse_status, nested_radial.status)
        else:
            magnification = nested_radial.value / normalization
            topology_error = jnp.asarray(0.0, dtype=real_dtype)
            estimated_error = nested_radial.error / normalization
            status = nested_radial.status
        tolerance = angular_atol_grid + (relative_tolerance_grid * jnp.abs(magnification))
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
        tolerance_failed = (status == RADIAL_OK) & ~(jnp.isfinite(estimated_error) & (estimated_error <= tolerance))
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
    fatal_statuses = (
        ANGULAR_CAPACITY
        | ANGULAR_DEGENERATE
        | ANGULAR_ROOT_FAILURE
        | RADIAL_CAPACITY
        | RADIAL_TOLERANCE
        | RADIAL_TOPOLOGY
    )
    return unwrap_boundary_result(result, return_info, fatal_statuses)
