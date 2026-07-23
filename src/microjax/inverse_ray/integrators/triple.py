"""Uniform triple-lens boundary integration."""

from typing import Union

import jax
import jax.numpy as jnp
from jax import lax

from microjax.lens_geometry import triple_lens_geometry
from ..geometry.limb import calc_source_limb
from ..geometry.mapping import distance_from_source
from ..geometry.topology import (
    RADIAL_CAPACITY,
    RADIAL_INTERVAL_CAPACITY,
    RADIAL_TOLERANCE,
    RADIAL_TOPOLOGY,
    define_radial_topology,
)
from ..quadrature.radial import RadialIntegrand, adaptive_radial_integral, fixed_radial_integral
from ..quadrature.rules import G15_W_ON_GK31, GK31_X, GL47_W, GL47_X
from ..roots.angular import (
    ANGULAR_CAPACITY,
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
    angular_intervals_triple_roots,
)
from ..roots.level_set import triple_level_set
from .charts import _owned_angular_intervals, _triple_compact_mixed_topology
from .common import (
    Array,
    BoundaryMagnificationResult,
    SEQUENTIAL_RADIAL_CHUNK_SIZE,
    integration_dtypes,
    unwrap_boundary_result,
)

_TRIPLE_TANGENT_EDGE_SHARPNESS = 30.0
_G15_ACTIVE = G15_W_ON_GK31 != 0.0
_G15_X = GK31_X[_G15_ACTIVE]
_G15_W = G15_W_ON_GK31[_G15_ACTIVE]


@jax.custom_jvp
def _hard_value_soft_jvp(hard_value: Array, soft_value: Array) -> Array:
    """Return the exact hard value while taking tangents from a soft profile."""

    return hard_value


@_hard_value_soft_jvp.defjvp
def _hard_value_soft_jvp_rule(primals, tangents):
    hard_value, _ = primals
    _, soft_dot = tangents
    return hard_value, soft_dot


def _compact_edge_intensity(normalized_distance: Array) -> Array:
    """Unit-flux profile that tapers smoothly to zero at the hard limb."""

    sharpness = _TRIPLE_TANGENT_EDGE_SHARPNESS

    def unnormalized(x):
        return jax.nn.sigmoid(sharpness * (1.0 - x)) - 0.5

    nodes = 0.5 * (jnp.asarray(GL47_X, dtype=normalized_distance.dtype) + 1.0)
    weights = 0.5 * jnp.asarray(GL47_W, dtype=normalized_distance.dtype)
    relative_flux = 2.0 * jnp.sum(weights * nodes * unnormalized(nodes))
    bounded_distance = jnp.clip(normalized_distance, 0.0, 1.0)
    return unnormalized(bounded_distance) / relative_flux


def _integrate_compact_edge(brightness, intervals) -> Array:
    """Integrate the JVP-only profile with the embedded rule's 15 Gauss nodes."""

    active = jnp.arange(intervals.intervals.shape[0]) < intervals.n_intervals

    def integrate(bounds):
        safe_bounds = jnp.where(active[:, None], bounds, bounds[0])

        def integrate_interval(pair):
            lower, upper = pair
            dtype = pair.dtype
            angle = 0.25 * jnp.pi * (jnp.asarray(_G15_X, dtype=dtype) + 1.0)
            width = upper - lower
            theta = lower + width * jnp.sin(angle) ** 2
            jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * angle)
            return jnp.sum(jnp.asarray(_G15_W, dtype=dtype) * jacobian * jax.vmap(brightness)(theta))

        values = jax.vmap(integrate_interval)(safe_bounds)
        return jnp.sum(jnp.where(active, values, 0.0))

    return jax.lax.cond(
        intervals.n_intervals > 0,
        integrate,
        lambda _: jnp.asarray(0.0, dtype=intervals.intervals.dtype),
        intervals.intervals,
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
    _compact_local_chart: bool = False,
) -> Union[Array, BoundaryMagnificationResult]:
    """Integrate a uniform triple-lens source from exact angular roots.

    Nlimb controls topology tracing; angular integration uses exact roots.
    ``radial_strategy="fixed"`` provides the bounded G15/K31 pass used by the
    public triple light-curve API, while ``"adaptive"`` retains the diagnostic
    error-controlled path. The primal is the exact hard-edge area. Its custom
    JVP reuses the same intervals and radial nodes with a unit-flux compact
    sigmoid profile, evaluated by G15, so source-limb contacts have a bounded
    tangent without changing the reported magnification.
    """

    if radial_strategy not in ("adaptive", "fixed"):
        raise ValueError("radial_strategy must be 'adaptive' or 'fixed'")
    if fixed_radial_order not in (31, 47):
        raise ValueError("fixed_radial_order must be 31 or 47")
    if radial_chunk_size <= 0:
        raise ValueError("radial_chunk_size must be positive")
    if _compact_local_chart and radial_strategy != "fixed":
        raise ValueError("the compact-image chart requires fixed radial integration")
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
    charted = None
    if _compact_local_chart:
        charted = _triple_compact_mixed_topology(
            image_limb,
            mask_limb,
            rho,
            margin_r=margin_r,
            w_center_shifted=w_center_shifted,
            origin_inside=origin_inside,
            shifted=geometry.shifted,
            a=geometry.a,
            e1=geometry.e1,
            e2=geometry.e2,
            r3_complex=geometry.r3_complex,
            lens_margin_parameters=margin_parameters,
        )
        topology = charted.topology
        interval_parameters = charted.interval_parameters
    else:
        topology = define_radial_topology(
            image_limb,
            mask_limb,
            rho,
            margin_r=margin_r,
            origin_inside=origin_inside,
            track_roots=track_limb_roots,
            lens_margin_parameters=margin_parameters,
        )
        interval_parameters = None

    # These intervals are conservative integration supports padded beyond the
    # physical image boundary, where the angular measure is identically zero.
    # Their motion therefore has no boundary contribution to the exact area.
    # Differentiating the sampled min/max construction would nevertheless move
    # every fixed quadrature node and amplify the (small) radial quadrature
    # error into a noisy Jacobian near caustics.
    topology = topology._replace(intervals=lax.stop_gradient(topology.intervals))

    real_dtype, complex_dtype = integration_dtypes(w_center)
    rho_grid = jnp.asarray(rho, dtype=real_dtype)
    shifted_grid = jnp.asarray(geometry.shifted, dtype=complex_dtype)
    a_grid = jnp.asarray(geometry.a, dtype=real_dtype)
    e1_grid = jnp.asarray(geometry.e1, dtype=real_dtype)
    e2_grid = jnp.asarray(geometry.e2, dtype=real_dtype)
    r3_complex_grid = jnp.asarray(geometry.r3_complex, dtype=complex_dtype)
    r3_grid = jnp.asarray(r3, dtype=real_dtype)
    psi_grid = jnp.asarray(psi, dtype=real_dtype)
    w_center_shifted_grid = jnp.asarray(w_center_shifted, dtype=complex_dtype)
    angular_atol_grid = jnp.asarray(angular_atol, dtype=real_dtype)
    relative_tolerance_grid = jnp.asarray(relative_tolerance, dtype=real_dtype)
    output_dtype = jnp.asarray(w_center).real.dtype
    normalization = jnp.pi * rho_grid**2
    cell_tolerance = 64.0 * jnp.finfo(real_dtype).eps

    def radial_integrand(radius, interval_parameter=None):
        chart_center = interval_parameter[0] if _compact_local_chart else 0.0 + 0.0j
        intervals = angular_intervals_triple_roots(
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
            chart_center=chart_center,
        )
        if _compact_local_chart:
            owned, n_owned = _owned_angular_intervals(
                intervals.intervals, intervals.n_intervals, radius, interval_parameter, charted
            )
            intervals = intervals._replace(intervals=owned, n_intervals=n_owned)

        active = jnp.arange(intervals.intervals.shape[0]) < intervals.n_intervals
        widths = intervals.intervals[:, 1] - intervals.intervals[:, 0]
        hard_measure = jnp.sum(jnp.where(active, widths, 0.0))
        soft_intervals = intervals._replace(intervals=lax.stop_gradient(intervals.intervals))

        def soft_brightness(theta):
            distance = distance_from_source(
                radius,
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
            distance = jnp.nan_to_num(distance, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)
            return _compact_edge_intensity(distance / rho_grid)

        soft_measure = _integrate_compact_edge(soft_brightness, soft_intervals)
        measure = _hard_value_soft_jvp(hard_measure, soft_measure)
        return RadialIntegrand(
            radius * measure,
            jnp.abs(radius) * intervals.error,
            intervals.status,
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
            interval_parameters=interval_parameters,
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
