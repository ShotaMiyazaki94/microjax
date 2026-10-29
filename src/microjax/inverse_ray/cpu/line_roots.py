"""Physical binary-lens line roots in a locally centered coordinate."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry.lens import binary_geometry
from .roots import batched_polished_real_companion_roots

Array = jnp.ndarray


def _fixed_polynomial_product_ascending(left: Array, right: Array) -> Array:
    """Multiply two tiny ascending polynomials with statically unrolled sums."""

    terms = []
    for output_index in range(left.shape[0] + right.shape[0] - 1):
        products = [
            left[left_index] * right[output_index - left_index]
            for left_index in range(left.shape[0])
            if 0 <= output_index - left_index < right.shape[0]
        ]
        value = products[0]
        for product in products[1:]:
            value = value + product
        terms.append(value)
    return jnp.stack(terms)


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


def binary_line_level_set_roots(
    offset: complex | Array,
    direction: complex | Array,
    w_center: complex | Array,
    rho: float | Array,
    *,
    s: float | Array,
    q: float | Array,
) -> Array:
    """Solve image-boundary intersections, returning original line coordinates.

    Rebuild the sextic about the line projection of ``s * (1-q) / (1+q)``.
    This origin changes sign under lens exchange and approaches the lower-mass
    lens at extreme mass ratios on either side of unity.  It separates close
    root clusters from large global-coordinate monomials that can change their
    real/complex classification in a companion solve.  Coefficients are formed
    directly from the lens geometry in that coordinate, rather than translating
    already rounded global coefficients.  The same single real companion solve
    and four polynomial Newton steps are used throughout.

    ``offset`` and nonzero ``direction`` broadcast over leading dimensions.
    """

    offset, direction = jnp.broadcast_arrays(offset, direction)
    symmetric_origin = s * (1 - q) / (1 + q)
    center = jnp.real((symmetric_origin - offset) * jnp.conj(direction)) / jnp.real(
        direction * jnp.conj(direction)
    )
    local_offset = offset + center * direction
    coefficients = jax.vmap(
        lambda origin, vector: binary_line_level_set_coefficients(
            origin, vector, w_center, rho, s=s, q=q
        )
    )(local_offset.reshape(-1), direction.reshape(-1))
    roots = batched_polished_real_companion_roots(coefficients)
    return roots.reshape(offset.shape + (6,)) + center[..., None]


def _safe_physical_line_radii(radius, real, offset, direction, w_center, rho, s):
    """Keep unused complex-root lanes away from real-axis lens poles."""
    offset, direction = jnp.broadcast_arrays(offset, direction)
    far = (
        (4 + jnp.abs(offset) + jnp.abs(s) + jnp.abs(w_center) + jnp.abs(rho))
        / jnp.maximum(jnp.abs(direction), jnp.finfo(radius.dtype).tiny)
    )
    return jnp.where(real, radius, far[..., None])


def _physical_line_level_and_slope(radius, offset, direction, w_center, rho, s, q):
    """Evaluate the unexpanded lens constraint and its line derivative."""
    offset, direction = jnp.broadcast_arrays(offset, direction)
    image = offset[..., None] + radius * direction[..., None]
    lens = binary_geometry(s, q)
    conjugate = jnp.conj(image) - lens.shifted
    plus, minus = conjugate - lens.a, conjugate + lens.a
    mapped = image - lens.e1 / plus - (1 - lens.e1) / minus
    shear = lens.e1 / plus**2 + (1 - lens.e1) / minus**2
    delta = mapped - w_center
    level = jnp.real(delta * jnp.conj(delta)) - rho**2
    motion = direction[..., None] + shear * jnp.conj(direction[..., None])
    slope = 2 * jnp.real(jnp.conj(delta) * motion)
    return level, slope


def _polynomial_line_roots(offset, direction, w_center, rho, s, q):
    return binary_line_level_set_roots(offset, direction, w_center, rho, s=s, q=q)


@jax.custom_jvp
def _physical_line_roots(offset, direction, w_center, rho, s, q):
    roots = _polynomial_line_roots(offset, direction, w_center, rho, s, q)
    real = roots.imag == 0
    pairs = real[..., :, None] & real[..., None, :] & ~jnp.eye(roots.shape[-1], dtype=bool)

    def correct(_, radius):
        safe = _safe_physical_line_radii(radius, real, offset, direction, w_center, rho, s)
        level, slope = _physical_line_level_and_slope(safe, offset, direction, w_center, rho, s, q)
        step = level / slope
        proposed = radius - step
        safe_proposed = _safe_physical_line_radii(proposed, real, offset, direction, w_center, rho, s)
        proposed_level, _ = _physical_line_level_and_slope(
            safe_proposed, offset, direction, w_center, rho, s, q
        )
        separation = jnp.min(
            jnp.where(pairs, jnp.abs(radius[..., :, None] - radius[..., None, :]), jnp.inf), axis=-1
        )
        accept = (
            real & jnp.isfinite(proposed) & jnp.isfinite(proposed_level)
            & jnp.isfinite(separation) & (jnp.abs(step) <= .25 * separation)
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
    offset, direction, w_center, rho, s, q = primals
    real = roots.imag == 0
    radius = _safe_physical_line_radii(roots.real, real, offset, direction, w_center, rho, s)
    _, partial_level = jax.jvp(
        lambda *parameters: _physical_line_level_and_slope(radius, *parameters)[0], primals, tangents
    )
    _, slope = _physical_line_level_and_slope(radius, *primals)
    physical_tangent = -partial_level / jnp.where(real, slope, 1)
    _, polynomial_tangent = jax.jvp(_polynomial_line_roots, primals, tangents)
    return roots, jnp.where(real, physical_tangent.astype(roots.dtype), polynomial_tangent)


def physical_binary_line_level_set_roots(offset, direction, w_center, rho, *, s, q):
    """Refine accepted real line roots once in the physical lens equation.

    The correction must reduce the physical residual and stay within a
    quarter of the nearest real-root separation. Complex pairs are preserved;
    this does not recover missing roots or certify root completeness. Real
    roots carry the implicit derivative of the physical equation. The cheaper
    polynomial-only solve remains available for support occupancy checks.
    """
    return _physical_line_roots(offset, direction, w_center, rho, s, q)
