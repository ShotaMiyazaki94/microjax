"""Physical lens-boundary line roots in a locally centered coordinate."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry import binary_geometry
from .roots import batched_polished_real_companion_roots
from .lens import binary_lens, lens_map, lens_shear
from .lens import _polynomial_product_ascending as _fixed_polynomial_product_ascending

Array = jnp.ndarray


def binary_line_level_set_coefficients(
    offset: complex | Array,
    direction: complex | Array,
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
) -> Array:
    """Return coefficients of ``H(offset + t * direction)`` for real ``t``."""

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    offset = jnp.asarray(offset, dtype=w_center.dtype)
    direction = jnp.asarray(direction, dtype=w_center.dtype)
    rho = jnp.asarray(rho, dtype=real_dtype)
    lens = binary_geometry(s, q)

    base_bar = jnp.conjugate(offset) - lens.shifted
    direction_bar = jnp.conjugate(direction)
    denominator = jnp.asarray(
        [
            base_bar**2 - lens.a**2,
            2.0 * base_bar * direction_bar,
            direction_bar**2,
        ],
        dtype=w_center.dtype,
    )
    deflection_constant = -lens.shifted + lens.a * (2.0 * lens.e1 - 1.0)
    numerator = _fixed_polynomial_product_ascending(
        jnp.asarray([offset - w_center, direction], dtype=w_center.dtype),
        denominator,
    ) - jnp.asarray(
        [
            jnp.conjugate(offset) + deflection_constant,
            direction_bar,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        dtype=w_center.dtype,
    )
    numerator_square = _fixed_polynomial_product_ascending(numerator, jnp.conjugate(numerator))
    denominator_square = _fixed_polynomial_product_ascending(denominator, jnp.conjugate(denominator))
    ascending = jnp.real(numerator_square - rho**2 * jnp.pad(denominator_square, (0, 2)))
    descending = ascending[::-1]
    scale = jnp.max(jnp.abs(descending))
    return descending / jnp.maximum(scale, jnp.finfo(real_dtype).tiny)


def line_level_set_coefficients(offset, direction, w_center, rho, *, lens):
    """Return the real degree ``2*N+2`` source-disc polynomial on one line."""
    if lens.positions.shape[0] == 2:
        return binary_line_level_set_coefficients(offset, direction, w_center, rho, s=lens.s, q=lens.q)

    w_center = jnp.asarray(w_center)
    offset = jnp.asarray(offset, dtype=w_center.dtype)
    direction = jnp.asarray(direction, dtype=w_center.dtype)
    count = lens.positions.shape[0]
    factors = [jnp.asarray([jnp.conj(offset - position), jnp.conj(direction)]) for position in lens.positions]
    denominator = jnp.ones((1,), dtype=w_center.dtype)
    for factor in factors:
        denominator = _fixed_polynomial_product_ascending(denominator, factor)
    deflection = jnp.zeros((count,), dtype=w_center.dtype)
    for index in range(count):
        term = jnp.ones((1,), dtype=w_center.dtype)
        for other in range(count):
            if other != index:
                term = _fixed_polynomial_product_ascending(term, factors[other])
        deflection = deflection + lens.masses[index] * term
    numerator = _fixed_polynomial_product_ascending(jnp.asarray([offset - w_center, direction]), denominator) - jnp.pad(
        deflection, (0, 2)
    )
    square = _fixed_polynomial_product_ascending(numerator, jnp.conj(numerator))
    denominator_square = _fixed_polynomial_product_ascending(denominator, jnp.conj(denominator))
    coefficients = jnp.real(square - rho**2 * jnp.pad(denominator_square, (0, 2)))[::-1]
    return coefficients / jnp.maximum(jnp.max(jnp.abs(coefficients)), jnp.finfo(w_center.real.dtype).tiny)


def _local_line_polynomial(offset, direction, w_center, rho, lens):
    """Form coefficients locally instead of translating rounded monomials."""
    offset, direction = jnp.broadcast_arrays(offset, direction)
    direction_norm = jnp.real(direction * jnp.conj(direction))
    if lens.positions.shape[0] == 2:
        # Retain the validated binary origin and its operation ordering.
        origin = lens.s * (1 - lens.q) / (1 + lens.q)
    else:
        # Include each lens's Einstein-radius scale: a near-origin primary
        # otherwise wins every polar ray even when a tiny distant image
        # determines the polynomial's conditioning.
        relative = (lens.positions - offset[..., None]) * jnp.conj(direction[..., None])
        scale = relative.imag**2 / direction_norm[..., None] + lens.masses
        origin = lens.positions[jnp.argmin(scale, axis=-1)]
    center = jnp.real((origin - offset) * jnp.conj(direction)) / direction_norm
    local_offset = offset + center * direction
    coefficients = jax.vmap(
        lambda location, vector: line_level_set_coefficients(location, vector, w_center, rho, lens=lens)
    )(local_offset.reshape(-1), direction.reshape(-1))
    return coefficients, center


def line_level_set_roots(offset, direction, w_center, rho, *, lens):
    """Return all real/complex intersections in the original line coordinate."""
    coefficients, center = _local_line_polynomial(offset, direction, w_center, rho, lens)
    roots = batched_polished_real_companion_roots(coefficients)
    return roots.reshape(center.shape + (coefficients.shape[-1] - 1,)) + center[..., None]


def binary_line_level_set_roots(offset, direction, w_center, rho, *, s, q):
    return line_level_set_roots(offset, direction, w_center, rho, lens=binary_lens(s, q))


def _safe_physical_line_radii(radius, real, offset, direction, w_center, rho, lens):
    """Keep unused complex-root lanes away from real-axis lens poles."""
    offset, direction = jnp.broadcast_arrays(offset, direction)
    far = (
        4 + jnp.abs(offset) + jnp.abs(lens.s) + jnp.max(jnp.abs(lens.positions)) + jnp.abs(w_center) + jnp.abs(rho)
    ) / jnp.maximum(jnp.abs(direction), jnp.finfo(radius.dtype).tiny)
    return jnp.where(real, radius, far[..., None])


def _physical_line_level_and_slope(radius, offset, direction, w_center, rho, lens):
    """Evaluate the unexpanded lens constraint and its line derivative."""
    offset, direction = jnp.broadcast_arrays(offset, direction)
    image = offset[..., None] + radius * direction[..., None]
    delta = lens_map(image, lens) - w_center
    level = jnp.real(delta * jnp.conj(delta)) - rho**2
    motion = direction[..., None] + lens_shear(image, lens) * jnp.conj(direction[..., None])
    slope = 2 * jnp.real(jnp.conj(delta) * motion)
    return level, slope


@jax.custom_jvp
def _physical_line_roots(offset, direction, w_center, rho, lens):
    roots = line_level_set_roots(offset, direction, w_center, rho, lens=lens)
    real = roots.imag == 0
    pairs = real[..., :, None] & real[..., None, :] & ~jnp.eye(roots.shape[-1], dtype=bool)

    def correct(_, radius):
        safe = _safe_physical_line_radii(radius, real, offset, direction, w_center, rho, lens)
        level, slope = _physical_line_level_and_slope(safe, offset, direction, w_center, rho, lens)
        step = level / slope
        proposed = radius - step
        safe_proposed = _safe_physical_line_radii(proposed, real, offset, direction, w_center, rho, lens)
        proposed_level, _ = _physical_line_level_and_slope(safe_proposed, offset, direction, w_center, rho, lens)
        separation = jnp.min(jnp.where(pairs, jnp.abs(radius[..., :, None] - radius[..., None, :]), jnp.inf), axis=-1)
        accept = (
            real
            & jnp.isfinite(proposed)
            & jnp.isfinite(proposed_level)
            & jnp.isfinite(separation)
            & (jnp.abs(step) <= 0.25 * separation)
            & (jnp.abs(proposed_level) <= jnp.abs(level))
        )
        return jnp.where(accept, proposed, radius)

    corrected = jax.lax.fori_loop(0, 1, correct, roots.real)
    return jnp.where(real, corrected.astype(roots.dtype), roots)


@_physical_line_roots.defjvp
def _physical_line_roots_jvp(primals, tangents):
    # Re-enter the complete solve so higher forward derivatives include root
    # motion. Complex roots retain their polynomial derivative and identity.
    roots = _physical_line_roots(*primals)
    offset, direction, w_center, rho, lens = primals
    real = roots.imag == 0
    radius = _safe_physical_line_radii(roots.real, real, offset, direction, w_center, rho, lens)
    _, partial_level = jax.jvp(
        lambda *parameters: _physical_line_level_and_slope(radius, *parameters)[0], primals, tangents
    )
    _, slope = _physical_line_level_and_slope(radius, *primals)
    physical_tangent = -partial_level / jnp.where(real, slope, 1)
    # Reuse the solved complex lanes for their polynomial implicit derivative.
    (coefficients, center), (dcoefficients, dcenter) = jax.jvp(_local_line_polynomial, primals, tangents)
    degree = roots.shape[-1]
    local_roots = (roots - center[..., None]).reshape((-1, degree))
    derivative = coefficients[:, :-1] * jnp.arange(degree, 0, -1, dtype=coefficients.dtype)
    numerator = jax.vmap(jnp.polyval)(dcoefficients.astype(roots.dtype), local_roots)
    denominator = jax.vmap(jnp.polyval)(derivative.astype(roots.dtype), local_roots)
    polynomial_tangent = (-numerator / denominator).reshape(roots.shape) + dcenter[..., None]
    return roots, jnp.where(real, physical_tangent.astype(roots.dtype), polynomial_tangent)


def physical_line_level_set_roots(offset, direction, w_center, rho, *, lens):
    """Refine real roots physically, preserving complex lanes and implicit AD."""
    return _physical_line_roots(offset, direction, w_center, rho, lens)


def physical_binary_line_level_set_roots(offset, direction, w_center, rho, *, s, q):
    return physical_line_level_set_roots(offset, direction, w_center, rho, lens=binary_lens(s, q))
