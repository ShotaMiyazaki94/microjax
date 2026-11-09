"""Cartesian inverse-ray moments with twelve inner Jacobi nodes."""

from __future__ import annotations
import jax
import jax.numpy as jnp
from .lens import lens_map
from .line_roots import line_level_set_coefficients, physical_line_level_set_roots
from .projection_moments import PROFILE_NODES, PROFILE_WEIGHTS, integrate_adaptive_moments
from .result import CpuMagnificationResult, INVALID_ROOTS, NONFINITE
from .angular_moment import ANGULAR_MOMENT_TOPOLOGY


def _strip_profile_moments(abscissa, w_center, rho, *, lens, axis):
    coefficients = jax.vmap(lambda x: line_level_set_coefficients(x * axis, 1j * axis, w_center, rho, lens=lens))(
        abscissa
    )
    roots = physical_line_level_set_roots(abscissa * axis, 1j * axis, w_center, rho, lens=lens)
    real = roots.imag == 0
    residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    valid = real & (residual <= 2.0e-7)
    ordered = jnp.sort(jnp.where(valid, roots.real, jnp.inf), axis=-1)
    root_count = jnp.sum(valid, axis=-1, dtype=jnp.int32)
    pair = jnp.arange(0, roots.shape[-1], 2)
    active = pair[None, :] + 1 < root_count[:, None]
    lower = jnp.where(active, ordered[:, pair], 0.0)
    upper = jnp.where(active, ordered[:, pair + 1], 0.0)
    invalid = jnp.sum((~valid) & real, axis=-1, dtype=jnp.int32) + jnp.mod(root_count, 2)
    nodes = jnp.asarray(PROFILE_NODES, dtype=w_center.real.dtype)
    weights = jnp.asarray(PROFILE_WEIGHTS, dtype=w_center.real.dtype)
    jacobi_weight = jnp.sqrt((1.0 - nodes) * (1.0 + nodes))

    def one_strip(x, lo, hi, interval_active):
        midpoint = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo)
        ordinate = midpoint[:, None] + half_width[:, None] * nodes[None, :]
        images = x * axis + ordinate * (1j * axis)
        mapped = lens_map(images, lens)
        normalized = jnp.abs(mapped - w_center) / rho
        # Benign inactive lanes prevent sqrt's derivative from forming 0*inf.
        inside = interval_active[:, None] & (normalized < 1.0)
        radicand = jnp.where(inside, (1.0 - normalized) * (1.0 + normalized), 1.0)
        mu = jnp.where(inside, jnp.sqrt(radicand), 0.0)
        profile = half_width * jnp.sum(weights[None, :] * (mu / jacobi_weight[None, :]), axis=-1)
        return (jnp.sum(jnp.where(interval_active, hi - lo, 0.0)), jnp.sum(jnp.where(interval_active, profile, 0.0)))

    uniform, profile = jax.vmap(one_strip)(abscissa, lower, upper, active)
    return jnp.stack((uniform, profile), axis=-1), invalid


def _mag_limb_dark_cartesian_impl(w_center, rho, *, lens, u1, axis, support):
    """Integrate one traced Cartesian support with the shared bounded allocator."""
    cells, active, topology, _, _ = support
    integral = integrate_adaptive_moments(
        lambda x: _strip_profile_moments(x, w_center, rho, lens=lens, axis=axis), cells, active
    )
    uniform, profile = integral.value
    normalization = 3.0 / (jnp.pi * rho**2 * (3.0 - u1))
    magnification = normalization * ((1.0 - u1) * uniform + u1 * profile)
    status = (
        jnp.where(topology, jnp.int32(ANGULAR_MOMENT_TOPOLOGY), jnp.int32(0))
        | jnp.where(integral.status == 0, jnp.int32(0), jnp.int32(INVALID_ROOTS))
        | jnp.where(jnp.isfinite(magnification), jnp.int32(0), jnp.int32(NONFINITE))
    )
    return CpuMagnificationResult(magnification, status)
