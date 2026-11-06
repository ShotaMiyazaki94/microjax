"""Polar inverse-ray moments with twelve inner Jacobi nodes."""

from __future__ import annotations
import jax
import jax.numpy as jnp
from microjax.point_source import lens_eq
from ..geometry.lens import binary_geometry
from .angular_moment import _stable_radial_level_value, ANGULAR_MOMENT_SUPPORT, ANGULAR_MOMENT_EXHAUSTED
from .line_roots import binary_line_level_set_roots, physical_binary_line_level_set_roots
from .projection_moments import PROFILE_NODES, PROFILE_WEIGHTS, integrate_adaptive_moments
from .result import CpuMagnificationResult, INVALID_ROOTS, NONFINITE

Array = jnp.ndarray


def _physical_ray_intervals(
    theta: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    refine_roots: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Classify radial intervals; refine endpoints only for LD integration.

    Support occupancy checks retain the cheaper polynomial-only solve.
    ``refine_roots`` is a static choice at each caller, not a geometry route.
    """

    solve = physical_binary_line_level_set_roots if refine_roots else binary_line_level_set_roots
    roots = solve(0.0j, jnp.exp(1.0j * theta), w_center, rho, s=s, q=q)
    valid = (roots.imag == 0) & jnp.isfinite(roots.real) & (roots.real >= 0)
    radii = jnp.sort(jnp.where(valid, roots.real, jnp.inf), axis=-1)
    outer = jnp.maximum(jnp.max(jnp.where(valid, roots.real, 0), axis=-1) + 1, 1)
    radii = jnp.where(jnp.isfinite(radii), radii, outer[:, None])
    boundaries = jnp.concatenate([jnp.zeros_like(outer[:, None]), radii, outer[:, None]], axis=-1)
    lower, upper = boundaries[:, :-1], boundaries[:, 1:]
    values = _stable_radial_level_value(theta, 0.5 * (lower + upper), w_center, rho, s=s, q=q)
    inside = (values <= 0) & (upper > lower) & jnp.any(valid, axis=-1)[:, None]
    invalid = jnp.sum(~jnp.isfinite(roots), axis=-1, dtype=jnp.int32)
    return lower, upper, inside, invalid


def _profile_ray_moments(theta, w_center, rho, *, s, q):
    lower, upper, active, invalid = _physical_ray_intervals(theta, w_center, rho, s=s, q=q, refine_roots=True)
    lens = binary_geometry(s, q)
    nodes = jnp.asarray(PROFILE_NODES, dtype=w_center.real.dtype)
    weights = jnp.asarray(PROFILE_WEIGHTS, dtype=w_center.real.dtype)
    jacobi_weight = jnp.sqrt((1.0 - nodes) * (1.0 + nodes))

    def one_angle(angle, lo, hi, inside):
        midpoint = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo)
        radius = midpoint[:, None] + half_width[:, None] * nodes[None, :]
        images = radius * jnp.exp(1j * angle)
        mapped = lens_eq(images - lens.shifted, nlenses=2, a=lens.a, e1=lens.e1) + lens.shifted
        normalized = jnp.abs(mapped - w_center) / rho
        strictly_inside = inside[:, None] & (normalized < 1.0)
        radicand = jnp.where(strictly_inside, (1.0 - normalized) * (1.0 + normalized), 1.0)
        mu = jnp.where(strictly_inside, jnp.sqrt(radicand), 0.0)
        smooth = radius * mu / jacobi_weight[None, :]
        values = half_width * jnp.sum(weights[None, :] * smooth, axis=-1)
        return jnp.sum(jnp.where(inside, values, 0.0))

    profile = jax.vmap(one_angle)(theta, lower, upper, active)
    uniform = 0.5 * jnp.sum(jnp.where(active, (upper - lower) * (upper + lower), 0.0), axis=-1)
    return jnp.stack((uniform, profile), axis=-1), invalid


def _mag_limb_dark_angular_moment_from_support(w_center, rho, *, s, q, u1, support):
    """Integrate one traced polar support with the shared bounded allocator."""
    integral = integrate_adaptive_moments(
        lambda theta: _profile_ray_moments(theta, w_center, rho, s=s, q=q),
        support.cells,
        support.active,
    )
    flux = (1.0 - u1) * integral.value[0] + u1 * integral.value[1]
    magnification = 3.0 / (jnp.pi * rho**2 * (3.0 - u1)) * flux
    # Trace topology is a routing warning here; physical event support is the
    # structural acceptance signal, matching the shared scheduler's contract.
    status = (
        jnp.where(jnp.all(support.tangencies_valid), jnp.int32(0), jnp.int32(ANGULAR_MOMENT_SUPPORT))
        | jnp.where(integral.status == 0, jnp.int32(0), jnp.int32(INVALID_ROOTS | ANGULAR_MOMENT_EXHAUSTED))
        | jnp.where(jnp.isfinite(magnification), jnp.int32(0), jnp.int32(NONFINITE))
    )
    return CpuMagnificationResult(magnification, status)
