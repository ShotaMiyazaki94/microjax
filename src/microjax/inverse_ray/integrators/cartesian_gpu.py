"""Single-pass Cartesian strip integration for accelerator workloads.

This module deliberately does not import :mod:`microjax.inverse_ray.cpu`.
It reuses the source-limb roots already computed by the boundary solver, then
solves all independent line sextics as one regular accelerator batch.  The
chart is useful when a low-q planetary image is too anisotropic for the
angle-first polar level set.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from microjax.point_source import lens_eq
from microjax.poly_solver import poly_roots
from ..geometry.lens import binary_geometry
from ..geometry.topology import RADIAL_OK, RADIAL_TOPOLOGY, track_limb_images
from .common import Array


_MAX_EXTREMA = 20
_LINE_DEGREE = 6


class CartesianGPUResult(NamedTuple):
    """Magnification and structural diagnostics for one fixed strip pass."""

    magnification: Array
    invalid_root_count: Array
    n_active_cells: Array
    status: Array


def _polynomial_product_ascending(left: Array, right: Array) -> Array:
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


def binary_line_level_set_coefficients_gpu(
    offset: Array,
    direction: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
) -> Array:
    """Return the real line sextic using factored low-q lens offsets."""

    w_center = jnp.asarray(w_center)
    real_dtype = w_center.real.dtype
    complex_dtype = w_center.dtype
    offset = jnp.asarray(offset, dtype=complex_dtype)
    direction = jnp.asarray(direction, dtype=complex_dtype)
    rho = jnp.asarray(rho, dtype=real_dtype)
    lens = binary_geometry(s, q)

    base = jnp.conjugate(offset) - lens.shifted
    slope = jnp.conjugate(direction)
    plus = base - lens.a
    minus = base + lens.a
    denominator = jnp.asarray(
        [plus * minus, slope * (plus + minus), slope**2],
        dtype=complex_dtype,
    )
    deflection_constant = -lens.shifted + lens.a * (2.0 * lens.e1 - 1.0)
    numerator = _polynomial_product_ascending(
        jnp.asarray([offset - w_center, direction], dtype=complex_dtype),
        denominator,
    ) - jnp.asarray(
        [
            jnp.conjugate(offset) + deflection_constant,
            slope,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ],
        dtype=complex_dtype,
    )
    numerator_square = _polynomial_product_ascending(
        numerator, jnp.conjugate(numerator)
    )
    denominator_square = _polynomial_product_ascending(
        denominator, jnp.conjugate(denominator)
    )
    ascending = jnp.real(
        numerator_square - rho**2 * jnp.pad(denominator_square, (0, 2))
    )
    descending = ascending[::-1]
    scale = jnp.max(jnp.abs(descending))
    return descending / jnp.maximum(scale, jnp.finfo(real_dtype).tiny)


def _source_axis(w_center: Array, rho: Array) -> Array:
    magnitude = jnp.abs(w_center)
    radial = jnp.where(
        magnitude > 0.0,
        w_center / jnp.maximum(magnitude, jnp.finfo(w_center.real.dtype).tiny),
        jnp.asarray(1.0 + 0.0j, dtype=w_center.dtype),
    )
    # Small sources need the radial projection to separate the thin planetary
    # pair.  The normal projection is better conditioned for broader sources.
    return jnp.where(rho <= 2.0e-4, radial, 1.0j * radial)


def _support_cells(image_limb: Array, mask_limb: Array, axis: Array):
    image_limb, mask_limb = track_limb_images(image_limb, mask_limb)
    coordinate = jnp.real(image_limb * jnp.conjugate(axis))
    previous = jnp.roll(coordinate, 1, axis=1)
    following = jnp.roll(coordinate, -1, axis=1)
    previous_mask = jnp.roll(mask_limb, 1, axis=1)
    following_mask = jnp.roll(mask_limb, -1, axis=1)
    incoming = coordinate - previous
    outgoing = following - coordinate
    candidates = mask_limb & (
        (incoming * outgoing <= 0.0) | ~previous_mask | ~following_mask
    )

    finite = mask_limb & jnp.isfinite(coordinate)
    branch_active = jnp.any(finite, axis=1)
    branch_has_candidate = jnp.any(candidates, axis=1)
    sample = jnp.arange(coordinate.shape[1], dtype=jnp.int32)
    minimum = jnp.argmin(jnp.where(finite, coordinate, jnp.inf), axis=1)
    maximum = jnp.argmax(jnp.where(finite, coordinate, -jnp.inf), axis=1)
    fallback = (branch_active & ~branch_has_candidate)[:, None] & (
        (sample[None, :] == minimum[:, None]) | (sample[None, :] == maximum[:, None])
    )
    candidates = candidates | fallback

    curvature = previous - 2.0 * coordinate + following
    safe_curvature = jnp.where(
        jnp.abs(curvature) > 64.0 * jnp.finfo(coordinate.dtype).eps,
        curvature,
        1.0,
    )
    offset = jnp.clip(0.5 * (previous - following) / safe_curvature, -1.0, 1.0)
    fitted = (
        coordinate + 0.5 * (following - previous) * offset + 0.5 * curvature * offset**2
    )
    fitted = jnp.where(previous_mask & following_mask, fitted, coordinate)

    flat = candidates.reshape(-1)
    indices = jnp.nonzero(flat, size=_MAX_EXTREMA, fill_value=-1)[0]
    n_extrema = jnp.sum(flat, dtype=jnp.int32)
    selected = fitted.reshape(-1)[jnp.maximum(indices, 0)]
    endpoints = jnp.sort(jnp.where(indices >= 0, selected, jnp.inf))
    cells = jnp.stack((endpoints[:-1], endpoints[1:]), axis=-1)
    active = (
        (jnp.arange(_MAX_EXTREMA - 1, dtype=jnp.int32) < n_extrema - 1)
        & jnp.isfinite(cells[:, 1])
        & (cells[:, 1] > cells[:, 0])
    )
    invalid = (
        (n_extrema < 2)
        | (n_extrema > _MAX_EXTREMA)
        | ~jnp.all(jnp.isfinite(image_limb))
    )
    return cells, active, invalid


def _negative_intervals(coefficients: Array):
    roots = poly_roots(coefficients)
    real = roots.real
    root_scale = jnp.maximum(1.0, jnp.abs(real))
    residual = jnp.abs(jax.vmap(jnp.polyval)(coefficients, roots))
    residual_scale = jax.vmap(jnp.polyval)(jnp.abs(coefficients), jnp.abs(roots))
    valid = (
        jnp.isfinite(roots)
        & (jnp.abs(roots.imag) <= 2.0e-7 * root_scale)
        & (
            residual
            <= 2.0e-7 * jnp.maximum(residual_scale, jnp.finfo(coefficients.dtype).tiny)
        )
    )
    ordered = jnp.sort(jnp.where(valid, real, jnp.inf), axis=-1)
    root_count = jnp.sum(valid, axis=-1, dtype=jnp.int32)
    pair = jnp.arange(0, _LINE_DEGREE, 2)
    active = pair[None, :] + 1 < root_count[:, None]
    lower = jnp.where(active, ordered[:, pair], 0.0)
    upper = jnp.where(active, ordered[:, pair + 1], 0.0)
    invalid = jnp.mod(root_count, 2) + jnp.sum(
        (~valid) & (jnp.abs(roots.imag) <= 2.0e-7 * root_scale),
        axis=-1,
        dtype=jnp.int32,
    )
    return lower, upper, active, invalid


def cartesian_limb_dark_from_trace_gpu(
    image_limb: Array,
    mask_limb: Array,
    w_center: Array,
    rho: Array,
    *,
    s: Array,
    q: Array,
    u1: Array,
    n_slice: int = 31,
    n_profile: int = 8,
) -> CartesianGPUResult:
    """Evaluate one fixed, fully batched Cartesian strip chart."""

    if n_slice < 8:
        raise ValueError("n_slice must be at least 8")
    if n_profile < 4:
        raise ValueError("n_profile must be at least 4")
    w_center = jnp.asarray(w_center)
    rho = jnp.asarray(rho, dtype=w_center.real.dtype)
    u1 = jnp.asarray(u1, dtype=w_center.real.dtype)
    axis = _source_axis(w_center, rho)
    cells, active_cells, support_invalid = _support_cells(image_limb, mask_limb, axis)

    slice_nodes_np, slice_weights_np = np.polynomial.legendre.leggauss(n_slice)
    slice_nodes = jnp.asarray(slice_nodes_np, dtype=w_center.real.dtype)
    slice_weights = jnp.asarray(slice_weights_np, dtype=w_center.real.dtype)
    transform = 0.25 * jnp.pi * (slice_nodes + 1.0)
    safe_cells = jnp.where(active_cells[:, None], cells, 0.0)
    widths = safe_cells[:, 1] - safe_cells[:, 0]
    abscissa = safe_cells[:, :1] + widths[:, None] * jnp.sin(transform)[None, :] ** 2
    jacobian = (
        slice_weights[None, :]
        * 0.25
        * jnp.pi
        * widths[:, None]
        * jnp.sin(2.0 * transform)[None, :]
    )
    flat_x = abscissa.reshape(-1)
    coefficients = jax.vmap(
        lambda x: binary_line_level_set_coefficients_gpu(
            x * axis,
            1.0j * axis,
            w_center,
            rho,
            s=s,
            q=q,
        )
    )(flat_x)
    lower, upper, interval_active, root_invalid = _negative_intervals(coefficients)

    profile_index = jnp.arange(n_profile, 0, -1, dtype=w_center.real.dtype)
    profile_angle = jnp.pi * profile_index / (n_profile + 1)
    profile_nodes = jnp.cos(profile_angle)
    profile_weights = jnp.pi * jnp.sin(profile_angle) ** 2 / (n_profile + 1)
    jacobi_weight = jnp.sqrt((1.0 - profile_nodes) * (1.0 + profile_nodes))
    lens = binary_geometry(s, q)

    def strip_moments(x, lo, hi, intervals_active):
        midpoint = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo)
        ordinate = midpoint[:, None] + half_width[:, None] * profile_nodes[None, :]
        images = x * axis + ordinate * (1.0j * axis)
        centered = images - lens.shifted
        mapped = (
            lens_eq(
                centered,
                nlenses=2,
                a=lens.a,
                e1=lens.e1,
            )
            + lens.shifted
        )
        normalized = jnp.abs(mapped - w_center) / rho
        inside = intervals_active[:, None] & (normalized < 1.0)
        radicand = jnp.where(
            inside,
            (1.0 - normalized) * (1.0 + normalized),
            1.0,
        )
        mu = jnp.where(inside, jnp.sqrt(radicand), 0.0)
        profile = half_width * jnp.sum(
            profile_weights[None, :] * mu / jacobi_weight[None, :], axis=-1
        )
        uniform = hi - lo
        return (
            jnp.sum(jnp.where(intervals_active, uniform, 0.0)),
            jnp.sum(jnp.where(intervals_active, profile, 0.0)),
        )

    uniform, profile = jax.vmap(strip_moments)(flat_x, lower, upper, interval_active)
    node_active = jnp.repeat(active_cells[:, None], n_slice, axis=1).reshape(-1)
    flat_weight = jacobian.reshape(-1)
    uniform_area = jnp.sum(jnp.where(node_active, flat_weight * uniform, 0.0))
    profile_area = jnp.sum(jnp.where(node_active, flat_weight * profile, 0.0))
    normalization = 3.0 / (jnp.pi * rho**2 * (3.0 - u1))
    magnification = normalization * ((1.0 - u1) * uniform_area + u1 * profile_area)
    invalid_root_count = jnp.sum(
        jnp.where(node_active, root_invalid, 0), dtype=jnp.int32
    )
    valid = ~support_invalid & (invalid_root_count == 0) & jnp.isfinite(magnification)
    return CartesianGPUResult(
        magnification,
        invalid_root_count,
        jnp.sum(active_cells, dtype=jnp.int32),
        jnp.where(valid, jnp.int32(RADIAL_OK), jnp.int32(RADIAL_TOPOLOGY)),
    )
