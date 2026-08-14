"""Angle-first radial-profile moments for linear limb darkening."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from microjax.point_source import lens_eq

from ..geometry.lens import binary_geometry
from .angular_moment import (
    ANGULAR_MOMENT_EXHAUSTED,
    ANGULAR_MOMENT_SUPPORT,
    ANGULAR_MOMENT_TOPOLOGY,
    AngularSupport,
    AngularMomentResult,
    _angular_support_cells,
    _ray_intervals_complex,
    binary_radial_level_set_coefficients,
)
from .quadrature import G7_W_ON_GK15, GK15_W, GK15_X

Array = jnp.ndarray


def _profile_ray_moments(
    theta: Array,
    coefficients: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    n_radial: int,
    estimate_error: bool = True,
) -> tuple[Array, Array, Array]:
    lower, upper, inside, invalid = _ray_intervals_complex(coefficients)
    lens = binary_geometry(s, q)

    def integrate(nodes, weights):
        nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
        weights = jnp.asarray(weights, dtype=w_center.real.dtype)
        transform = 0.25 * jnp.pi * (nodes + 1.0)

        def one_angle(angle, lo, hi, active):
            width = hi - lo
            radius = lo[:, None] + width[:, None] * jnp.sin(transform) ** 2
            jacobian = (
                0.25 * jnp.pi * width[:, None] * jnp.sin(2.0 * transform)[None, :]
            )
            images = radius * jnp.exp(1.0j * angle)
            mapped = (
                lens_eq(
                    images - lens.shifted,
                    nlenses=2,
                    a=lens.a,
                    e1=lens.e1,
                )
                + lens.shifted
            )
            normalized = jnp.abs(mapped - w_center) / rho
            # The fixed interval capacity evaluates inactive lanes too.
            # Clipping their negative radicand to zero is value-safe but its
            # square-root JVP is ``0 * inf``.  Evaluate a benign radicand on
            # inactive/outside lanes and mask the resulting profile instead.
            strictly_inside = active[:, None] & (normalized < 1.0)
            safe_radicand = jnp.where(
                strictly_inside,
                (1.0 - normalized) * (1.0 + normalized),
                1.0,
            )
            mu = jnp.where(strictly_inside, jnp.sqrt(safe_radicand), 0.0)
            values = jnp.sum(
                weights[None, :] * radius * mu * jacobian,
                axis=-1,
            )
            return jnp.sum(jnp.where(active, values, 0.0))

        return jax.vmap(one_angle)(theta, lower, upper, inside)

    fine_nodes, fine_weights = np.polynomial.legendre.leggauss(n_radial)
    residual_fine = integrate(fine_nodes, fine_weights)
    if estimate_error:
        coarse_nodes, coarse_weights = np.polynomial.legendre.leggauss(
            max(4, n_radial // 2)
        )
        residual_coarse = integrate(coarse_nodes, coarse_weights)
    else:
        residual_coarse = residual_fine
    uniform = 0.5 * jnp.sum(
        jnp.where(inside, (upper - lower) * (upper + lower), 0.0), axis=-1
    )
    return uniform, residual_fine, jnp.abs(residual_fine - residual_coarse) + invalid


def _profile_ray_moments_gk15(
    theta: Array,
    coefficients: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate uniform and nested GK15/G7 LD ray moments once."""

    lower, upper, inside, invalid = _ray_intervals_complex(coefficients)
    lens = binary_geometry(s, q)
    nodes = jnp.asarray(GK15_X, dtype=w_center.real.dtype)
    fine_weights = jnp.asarray(GK15_W, dtype=w_center.real.dtype)
    coarse_weights = jnp.asarray(G7_W_ON_GK15, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)

    def one_angle(angle, lo, hi, active):
        width = hi - lo
        radius = lo[:, None] + width[:, None] * jnp.sin(transform) ** 2
        jacobian = (
            0.25 * jnp.pi * width[:, None] * jnp.sin(2.0 * transform)[None, :]
        )
        images = radius * jnp.exp(1.0j * angle)
        mapped = (
            lens_eq(
                images - lens.shifted,
                nlenses=2,
                a=lens.a,
                e1=lens.e1,
            )
            + lens.shifted
        )
        normalized = jnp.abs(mapped - w_center) / rho
        strictly_inside = active[:, None] & (normalized < 1.0)
        safe_radicand = jnp.where(
            strictly_inside,
            (1.0 - normalized) * (1.0 + normalized),
            1.0,
        )
        mu = jnp.where(strictly_inside, jnp.sqrt(safe_radicand), 0.0)
        integrand = radius * mu * jacobian
        fine = jnp.sum(fine_weights[None, :] * integrand, axis=-1)
        coarse = jnp.sum(coarse_weights[None, :] * integrand, axis=-1)
        return (
            jnp.sum(jnp.where(active, fine, 0.0)),
            jnp.sum(jnp.where(active, coarse, 0.0)),
        )

    residual_fine, residual_coarse = jax.vmap(one_angle)(
        theta, lower, upper, inside
    )
    uniform = 0.5 * jnp.sum(
        jnp.where(inside, (upper - lower) * (upper + lower), 0.0), axis=-1
    )
    return uniform, residual_fine, residual_coarse, invalid


def _mag_limb_dark_angular_moment_gk15_pair_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    u1: Array,
    support: AngularSupport,
) -> tuple[AngularMomentResult, AngularMomentResult]:
    """Evaluate an embedded GK15/G7 angular and radial LD pair once."""

    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    nodes = jnp.asarray(GK15_X, dtype=w_center.real.dtype)
    fine_weights = jnp.asarray(GK15_W, dtype=w_center.real.dtype)
    coarse_weights = jnp.asarray(G7_W_ON_GK15, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)

    def integrate_cell(inputs):
        lower, width, is_active = inputs

        def evaluate(_):
            theta = lower + width * jnp.sin(transform) ** 2
            jacobian = 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
            coefficients = jax.vmap(
                lambda angle: binary_radial_level_set_coefficients(
                    angle, w_center, rho, s=s, q=q
                )
            )(theta)
            uniform, residual_fine, residual_coarse, invalid = (
                _profile_ray_moments_gk15(
                    theta,
                    coefficients,
                    w_center,
                    rho,
                    s=s,
                    q=q,
                )
            )
            fine_profile = (1.0 - u1) * uniform + u1 * residual_fine
            coarse_profile = (1.0 - u1) * uniform + u1 * residual_coarse
            fine = jnp.sum(fine_weights * jacobian * fine_profile)
            coarse = jnp.sum(coarse_weights * jacobian * coarse_profile)
            radial_error = jnp.sum(
                jnp.abs(fine_weights * jacobian * u1)
                * jnp.abs(residual_fine - residual_coarse)
            )
            return fine, coarse, radial_error, jnp.sum(invalid, dtype=jnp.int32)

        return jax.lax.cond(
            is_active,
            evaluate,
            lambda _: (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.int32(0),
            ),
            operand=None,
        )

    widths = cells[:, 1] - cells[:, 0]
    fine_values, coarse_values, radial_errors, invalid_counts = jax.lax.map(
        integrate_cell, (cells[:, 0], widths, active)
    )
    normalization = 3.0 / (jnp.pi * rho**2 * (3.0 - u1))
    finite = (
        jnp.all(jnp.isfinite(fine_values))
        & jnp.all(jnp.isfinite(coarse_values))
        & jnp.all(jnp.isfinite(radial_errors))
    )
    status = jnp.bitwise_or(
        jnp.bitwise_or(
            jnp.where(finite, jnp.int32(0), jnp.int32(1)),
            jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0)),
        ),
        jnp.where(
            jnp.all(tangencies_valid),
            jnp.int32(0),
            jnp.int32(ANGULAR_MOMENT_SUPPORT),
        ),
    )
    common = dict(
        invalid_root_count=jnp.sum(invalid_counts, dtype=jnp.int32),
        ghost_residual_ratio=ghost / jnp.maximum(
            rho, jnp.finfo(rho.dtype).tiny
        ),
        limb_topology=limb_topology,
        status=status,
    )
    fine = AngularMomentResult(
        normalization * jnp.sum(fine_values),
        normalization * jnp.sum(radial_errors),
        jnp.int32(GK15_X.size) * jnp.sum(active, dtype=jnp.int32),
        **common,
    )
    coarse = AngularMomentResult(
        normalization * jnp.sum(coarse_values),
        jnp.asarray(0.0, dtype=w_center.real.dtype),
        jnp.int32(np.count_nonzero(G7_W_ON_GK15))
        * jnp.sum(active, dtype=jnp.int32),
        **common,
    )
    return coarse, fine


def _mag_limb_dark_angular_moment_from_support(
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    u1: Array,
    n_theta: int,
    n_radial: int,
    support: AngularSupport,
    estimate_radial_error: bool = True,
) -> AngularMomentResult:
    """Evaluate one LD polar tier on precomputed angular support."""

    cells, active, topology, ghost, limb_topology, tangencies_valid = support
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    nodes = jnp.asarray(nodes, dtype=w_center.real.dtype)
    weights = jnp.asarray(weights, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (nodes + 1.0)

    def integrate_cell(inputs):
        lower, width, is_active = inputs

        def evaluate(_):
            theta = lower + width * jnp.sin(transform) ** 2
            angular_weights = (
                weights * 0.25 * jnp.pi * width * jnp.sin(2.0 * transform)
            )
            coefficients = jax.vmap(
                lambda angle: binary_radial_level_set_coefficients(
                    angle, w_center, rho, s=s, q=q
                )
            )(theta)
            uniform, residual, radial_error = _profile_ray_moments(
                theta,
                coefficients,
                w_center,
                rho,
                s=s,
                q=q,
                n_radial=n_radial,
                estimate_error=estimate_radial_error,
            )
            profile = (1.0 - u1) * uniform + u1 * residual
            return (
                jnp.sum(angular_weights * profile),
                jnp.sum(jnp.abs(angular_weights * u1) * radial_error),
            )

        return jax.lax.cond(
            is_active,
            evaluate,
            lambda _: (
                jnp.asarray(0.0, dtype=w_center.real.dtype),
                jnp.asarray(0.0, dtype=w_center.real.dtype),
            ),
            operand=None,
        )

    widths = cells[:, 1] - cells[:, 0]
    values, radial_errors = jax.lax.map(
        integrate_cell, (cells[:, 0], widths, active)
    )
    normalization = 3.0 / (jnp.pi * rho**2 * (3.0 - u1))
    finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(radial_errors))
    return AngularMomentResult(
        normalization * jnp.sum(values),
        normalization * jnp.sum(radial_errors),
        jnp.int32(n_theta) * jnp.sum(active, dtype=jnp.int32),
        jnp.int32(0),
        ghost / jnp.maximum(rho, jnp.finfo(rho.dtype).tiny),
        limb_topology,
        jnp.bitwise_or(
            jnp.bitwise_or(
                jnp.where(finite, jnp.int32(0), jnp.int32(1)),
                jnp.where(
                    topology,
                    jnp.int32(ANGULAR_MOMENT_TOPOLOGY),
                    jnp.int32(0),
                ),
            ),
            jnp.where(
                jnp.all(tangencies_valid),
                jnp.int32(0),
                jnp.int32(ANGULAR_MOMENT_SUPPORT),
            ),
        ),
    )


def mag_limb_dark_angular_moment_fixed(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    n_theta: int = 16,
    n_radial: int = 12,
    n_limb: int = 128,
    return_info: bool = False,
) -> Array | AngularMomentResult:
    """Evaluate one angle-first linear-LD profile-moment tier."""

    if n_theta <= 0 or n_radial <= 0 or n_limb <= 0:
        raise ValueError("quadrature and support sizes must be positive")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    u1 = jnp.asarray(u1, dtype=w_center.real.dtype)
    support = _angular_support_cells(
        w_center, rho, s=s, q=q, n_limb=n_limb
    )
    result = _mag_limb_dark_angular_moment_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_theta=n_theta,
        n_radial=n_radial,
        support=support,
    )
    return result if return_info else result.magnification


def mag_limb_dark_angular_moment_compact(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    rtol: float | Array = 1.0e-3,
    _support: AngularSupport | None = None,
    return_info: bool = False,
) -> Array | AngularMomentResult:
    """Cross-certify compact 8x6 and 12x8 LD polar moments."""

    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    u1 = jnp.asarray(u1, dtype=w_center.real.dtype)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    support = (
        _angular_support_cells(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=128,
        )
        if _support is None
        else _support
    )
    coarse = _mag_limb_dark_angular_moment_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_theta=8,
        n_radial=6,
        support=support,
    )
    fine = _mag_limb_dark_angular_moment_from_support(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_theta=12,
        n_radial=8,
        support=support,
    )
    scale = jnp.maximum(jnp.abs(fine.magnification), 1.0)
    difference = jnp.abs(fine.magnification - coarse.magnification)
    structural_status = jnp.bitwise_and(
        fine.status,
        jnp.bitwise_not(jnp.int32(ANGULAR_MOMENT_TOPOLOGY)),
    )
    estimated_error = jnp.maximum(
        2.0 * difference + fine.estimated_error,
        0.75 * rtol * scale,
    )
    certified = (
        (structural_status == 0)
        & jnp.isfinite(fine.magnification)
        & jnp.isfinite(estimated_error)
        & (estimated_error <= rtol * scale)
    )
    result = fine._replace(
        estimated_error=estimated_error,
        status=jnp.where(
            certified,
            jnp.int32(0),
            jnp.bitwise_or(fine.status, jnp.int32(ANGULAR_MOMENT_EXHAUSTED)),
        ),
    )
    return result if return_info else result.magnification


def mag_limb_dark_angular_moment_refined(
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
    u1: float | Array,
    rtol: float | Array = 1.0e-3,
    return_info: bool = False,
) -> Array | AngularMomentResult:
    """Refine the linear-LD profile in the polar coordinate chart."""

    w_center = jnp.asarray(w_center)
    rtol = jnp.asarray(rtol, dtype=w_center.real.dtype)
    medium = mag_limb_dark_angular_moment_fixed(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_theta=16,
        n_radial=8,
        return_info=True,
    )
    high = mag_limb_dark_angular_moment_fixed(
        w_center,
        rho,
        s=s,
        q=q,
        u1=u1,
        n_theta=24,
        n_radial=12,
        return_info=True,
    )
    scale = jnp.maximum(jnp.abs(high.magnification), 1.0)
    difference = jnp.abs(high.magnification - medium.magnification)
    structural_status = jnp.bitwise_and(
        high.status, jnp.bitwise_not(jnp.int32(ANGULAR_MOMENT_TOPOLOGY))
    )
    estimated_error = jnp.maximum(
        2.0 * difference + high.estimated_error,
        0.75 * rtol * scale,
    )
    certified = (
        (structural_status == 0)
        & jnp.isfinite(high.magnification)
        & jnp.isfinite(estimated_error)
        & (estimated_error <= rtol * scale)
    )
    result = high._replace(
        estimated_error=estimated_error,
        status=jnp.where(
            certified,
            jnp.int32(0),
            jnp.bitwise_or(high.status, jnp.int32(ANGULAR_MOMENT_EXHAUSTED)),
        ),
    )
    return result if return_info else result.magnification


__all__ = [
    "mag_limb_dark_angular_moment_compact",
    "mag_limb_dark_angular_moment_fixed",
    "mag_limb_dark_angular_moment_refined",
]
