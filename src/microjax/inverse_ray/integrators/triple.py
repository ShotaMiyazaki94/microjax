"""Uniform triple-lens boundary integration."""

from typing import Union

import jax.numpy as jnp

from microjax.lens_geometry import triple_lens_geometry
from ..geometry.limb import calc_source_limb
from ..geometry.topology import (
    RADIAL_CAPACITY,
    RADIAL_INTERVAL_CAPACITY,
    RADIAL_TOLERANCE,
    RADIAL_TOPOLOGY,
    define_radial_topology,
)
from ..quadrature.radial import RadialIntegrand, adaptive_radial_integral, fixed_radial_integral
from ..roots.angular import (
    ANGULAR_CAPACITY,
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
    angular_measure_triple_roots,
)
from ..roots.level_set import triple_level_set
from .common import (
    Array,
    BoundaryMagnificationResult,
    SEQUENTIAL_RADIAL_CHUNK_SIZE,
    integration_dtypes,
    unwrap_boundary_result,
)


def mag_uniform_triple_boundary(
    w_center: complex,
    rho: float,
    *,
    s: float,
    q: float,
    q3: float,
    r3: float,
    psi: float,
    Nlimb: int = 500,
    margin_r: float = 0.5,
    angular_atol: float = 1e-5,
    relative_tolerance: float = 1e-4,
    parallel_regions: bool = False,
    return_info: bool = False,
    track_limb_roots: bool = True,
    jacobian_radial_margin: bool = True,
    max_radial_subdivisions: int = 8,
    radial_strategy: str = "adaptive",
    radial_chunk_size: int = SEQUENTIAL_RADIAL_CHUNK_SIZE,
    fixed_radial_order: int = 31,
) -> Union[Array, BoundaryMagnificationResult]:
    """Integrate a uniform triple-lens source from exact angular roots.

    Nlimb controls topology tracing; angular integration uses exact roots.
    ``radial_strategy="fixed"`` provides the bounded G15/K31 pass used by the
    public triple light-curve API, while ``"adaptive"`` retains the diagnostic
    error-controlled path.
    """

    if radial_strategy not in ("adaptive", "fixed"):
        raise ValueError("radial_strategy must be 'adaptive' or 'fixed'")
    if fixed_radial_order not in (31, 47):
        raise ValueError("fixed_radial_order must be 31 or 47")
    if radial_chunk_size <= 0:
        raise ValueError("radial_chunk_size must be positive")

    geometry = triple_lens_geometry(s, q, q3, r3, psi)
    lens_params = {
        "s": s,
        "q": q,
        "q3": q3,
        "r3": r3,
        "psi": psi,
        "a": geometry.a,
        "e1": geometry.e1,
        "e2": geometry.e2,
    }
    w_center_shifted = w_center - geometry.shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, nlenses=3, **lens_params)
    origin_inside = (
        triple_level_set(
            jnp.asarray(0.0 + 0.0j),
            w_center_shifted,
            rho,
            geometry.shifted,
            a=geometry.a,
            e1=geometry.e1,
            e2=geometry.e2,
            r3_complex=geometry.r3_complex,
        )
        <= 0.0
    )
    lens_positions = jnp.asarray([geometry.a, -geometry.a, geometry.r3_complex])
    lens_masses = jnp.asarray([geometry.e1, geometry.e2, geometry.e3])
    margin_parameters = (geometry.shifted, lens_positions, lens_masses) if jacobian_radial_margin else None
    topology = define_radial_topology(
        image_limb,
        mask_limb,
        rho,
        margin_r=margin_r,
        origin_inside=origin_inside,
        track_roots=track_limb_roots,
        lens_margin_parameters=margin_parameters,
    )

    real_dtype, complex_dtype = integration_dtypes(w_center)
    rho_grid = jnp.asarray(rho, dtype=real_dtype)
    shifted_grid = jnp.asarray(geometry.shifted, dtype=complex_dtype)
    a_grid = jnp.asarray(geometry.a, dtype=real_dtype)
    e1_grid = jnp.asarray(geometry.e1, dtype=real_dtype)
    e2_grid = jnp.asarray(geometry.e2, dtype=real_dtype)
    r3_complex_grid = jnp.asarray(geometry.r3_complex, dtype=complex_dtype)
    w_center_shifted_grid = jnp.asarray(w_center_shifted, dtype=complex_dtype)
    angular_atol_grid = jnp.asarray(angular_atol, dtype=real_dtype)
    relative_tolerance_grid = jnp.asarray(relative_tolerance, dtype=real_dtype)
    output_dtype = jnp.asarray(w_center).real.dtype
    normalization = jnp.pi * rho_grid**2
    cell_tolerance = 64.0 * jnp.finfo(real_dtype).eps

    def radial_integrand(radius):
        angular = angular_measure_triple_roots(
            radius,
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
        return RadialIntegrand(
            radius * angular.measure,
            jnp.abs(radius) * angular.error,
            angular.status,
        )

    chunk_size = RADIAL_INTERVAL_CAPACITY if parallel_regions else radial_chunk_size
    integration_options = dict(
        relative_tolerance=relative_tolerance_grid,
        initial_status=topology.status,
        chunk_size=chunk_size,
    )
    if radial_strategy == "fixed":
        radial = fixed_radial_integral(
            radial_integrand,
            topology.intervals,
            topology.n_intervals,
            angular_atol_grid * normalization,
            subdivisions=max_radial_subdivisions,
            single_cell_order=fixed_radial_order,
            **integration_options,
        )
    else:
        radial = adaptive_radial_integral(
            radial_integrand,
            topology.intervals,
            topology.n_intervals,
            angular_atol_grid * normalization,
            max_subdivisions=max_radial_subdivisions,
            **integration_options,
        )
    result = BoundaryMagnificationResult(
        jnp.asarray(radial.value / normalization, dtype=output_dtype),
        jnp.asarray(radial.error / normalization, dtype=output_dtype),
        radial.status,
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
