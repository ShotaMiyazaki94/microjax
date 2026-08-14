"""Linear limb-darkening adapter for the radial-profile integrator."""

from typing import Optional, Union

import jax.numpy as jnp

from ..profiles import linear_limb_intensity
from .common import Array, BoundaryMagnificationResult, SEQUENTIAL_RADIAL_CHUNK_SIZE
from .profile import mag_radial_profile_boundary


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
    deep_topology_sampling: bool = True,
    radial_strategy: str = "adaptive",
    certify_topology: bool = True,
    radial_chunk_size: int = SEQUENTIAL_RADIAL_CHUNK_SIZE,
    fixed_radial_order: int = 31,
    angular_profile_subdivisions: int = 1,
    _planetary_local_chart: bool = False,
    _planetary_cartesian_chart: bool = False,
    _compact_local_chart: bool = False,
    _radial_interval_capacity: int = 64,
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
        deep_topology_sampling=deep_topology_sampling,
        radial_strategy=radial_strategy,
        certify_topology=certify_topology,
        radial_chunk_size=radial_chunk_size,
        fixed_radial_order=fixed_radial_order,
        angular_profile_subdivisions=angular_profile_subdivisions,
        _planetary_local_chart=_planetary_local_chart,
        _planetary_cartesian_chart=_planetary_cartesian_chart,
        _linear_limb_u1=u1_array,
        _compact_local_chart=_compact_local_chart,
        _radial_interval_capacity=_radial_interval_capacity,
        return_info=return_info,
    )
