"""Small companion-matrix root solver for CPU angular ICRS."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _complex_dtype(values: Array):
    return jnp.result_type(values, jnp.complex64)


@jax.custom_jvp
def companion_roots(coefficients: Array) -> Array:
    """Return all roots of one fixed-degree polynomial via a companion matrix."""

    coefficients = jnp.asarray(coefficients)
    degree = coefficients.shape[0] - 1
    complex_coefficients = coefficients.astype(_complex_dtype(coefficients))
    monic = complex_coefficients / complex_coefficients[0]
    matrix = jnp.zeros((degree, degree), dtype=complex_coefficients.dtype)
    matrix = matrix.at[0, :].set(-monic[1:])
    matrix = matrix.at[1:, :-1].set(
        jnp.eye(degree - 1, dtype=complex_coefficients.dtype)
    )
    roots = jnp.linalg.eigvals(matrix)
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.real.dtype)
    derivative_coefficients = complex_coefficients[:-1] * powers
    for _ in range(4):
        residual = jnp.polyval(complex_coefficients, roots)
        derivative = jnp.polyval(derivative_coefficients, roots)
        roots = roots - residual / derivative
    return roots


@companion_roots.defjvp
def _companion_roots_jvp(primals, tangents):
    (coefficients,) = primals
    (coefficient_tangent,) = tangents
    roots = companion_roots(coefficients)
    complex_coefficients = coefficients.astype(_complex_dtype(coefficients))
    complex_tangent = coefficient_tangent.astype(_complex_dtype(coefficients))
    degree = coefficients.shape[0] - 1
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.real.dtype)
    derivative_coefficients = complex_coefficients[:-1] * powers
    derivative_at_roots = jnp.polyval(derivative_coefficients, roots)
    exponents = jnp.arange(degree, -1, -1)
    vandermonde = roots[:, None] ** exponents[None, :]
    root_tangent = -(vandermonde @ complex_tangent) / derivative_at_roots
    return roots, root_tangent


def batched_companion_roots(coefficients: Array) -> Array:
    """Vectorize :func:`companion_roots` over all leading dimensions."""

    coefficients = jnp.asarray(coefficients)
    coefficient_count = coefficients.shape[-1]
    flat = coefficients.reshape((-1, coefficient_count))
    roots = jax.vmap(companion_roots)(flat)
    return roots.reshape(coefficients.shape[:-1] + (coefficient_count - 1,))


@jax.custom_jvp
def real_companion_roots(coefficients: Array) -> Array:
    """Return roots of one real polynomial through a real companion matrix.

    LAPACK's real nonsymmetric eigensolver is substantially cheaper on CPU
    than the complex eigensolver used for a generic polynomial.  The returned
    roots are still complex so non-real conjugate pairs remain observable to
    the caller.  The angular kernel subsequently polishes the original Fourier
    equation, so no redundant polynomial-space Newton pass is needed here.
    """

    coefficients = jnp.asarray(coefficients)
    degree = coefficients.shape[0] - 1
    monic = coefficients / coefficients[0]
    matrix = jnp.zeros((degree, degree), dtype=coefficients.dtype)
    matrix = matrix.at[0, :].set(-monic[1:])
    matrix = matrix.at[1:, :-1].set(
        jnp.eye(degree - 1, dtype=coefficients.dtype)
    )
    roots = jnp.linalg.eigvals(matrix)
    return roots


@real_companion_roots.defjvp
def _real_companion_roots_jvp(primals, tangents):
    (coefficients,) = primals
    (coefficient_tangent,) = tangents
    roots = real_companion_roots(coefficients)
    complex_coefficients = coefficients.astype(_complex_dtype(coefficients))
    complex_tangent = coefficient_tangent.astype(_complex_dtype(coefficients))
    degree = coefficients.shape[0] - 1
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.dtype)
    derivative_coefficients = complex_coefficients[:-1] * powers
    derivative_at_roots = jnp.polyval(derivative_coefficients, roots)
    exponents = jnp.arange(degree, -1, -1)
    vandermonde = roots[:, None] ** exponents[None, :]
    root_tangent = -(vandermonde @ complex_tangent) / derivative_at_roots
    return roots, root_tangent


@jax.custom_jvp
def polished_real_companion_roots(coefficients: Array) -> Array:
    """Return polished roots of one real polynomial using the real CPU solver.

    Angle-first radial moments produce real sextics.  Solving their real
    companion matrices avoids the substantially more expensive complex
    eigensolver, while four Newton steps in the original polynomial retain the
    residual quality of :func:`companion_roots`.
    """

    coefficients = jnp.asarray(coefficients)
    roots = real_companion_roots(coefficients)
    degree = coefficients.shape[0] - 1
    complex_coefficients = coefficients.astype(_complex_dtype(coefficients))
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.dtype)
    derivative_coefficients = complex_coefficients[:-1] * powers
    for _ in range(4):
        residual = jnp.polyval(complex_coefficients, roots)
        derivative = jnp.polyval(derivative_coefficients, roots)
        roots = roots - residual / derivative
    return roots


@polished_real_companion_roots.defjvp
def _polished_real_companion_roots_jvp(primals, tangents):
    (coefficients,) = primals
    (coefficient_tangent,) = tangents
    roots = polished_real_companion_roots(coefficients)
    complex_coefficients = coefficients.astype(_complex_dtype(coefficients))
    complex_tangent = coefficient_tangent.astype(_complex_dtype(coefficients))
    degree = coefficients.shape[0] - 1
    powers = jnp.arange(degree, 0, -1, dtype=coefficients.dtype)
    derivative_coefficients = complex_coefficients[:-1] * powers
    derivative_at_roots = jnp.polyval(derivative_coefficients, roots)
    exponents = jnp.arange(degree, -1, -1)
    vandermonde = roots[:, None] ** exponents[None, :]
    root_tangent = -(vandermonde @ complex_tangent) / derivative_at_roots
    return roots, root_tangent


def batched_polished_real_companion_roots(coefficients: Array) -> Array:
    """Vectorize :func:`polished_real_companion_roots` over leading axes."""

    coefficients = jnp.asarray(coefficients)
    coefficient_count = coefficients.shape[-1]
    flat = coefficients.reshape((-1, coefficient_count))
    roots = jax.vmap(polished_real_companion_roots)(flat)
    return roots.reshape(coefficients.shape[:-1] + (coefficient_count - 1,))


def _fixed_ea28_impl(coefficients: Array, ordinate_bound: Array) -> Array:
    """Implementation of the fixed 28-step EA solve on a scaled interval."""

    coefficients = jnp.asarray(coefficients)
    real_dtype = coefficients.real.dtype
    complex_dtype = _complex_dtype(coefficients)
    degree = coefficients.shape[0] - 1
    bound = jnp.maximum(
        jnp.asarray(ordinate_bound, dtype=real_dtype),
        jnp.finfo(real_dtype).tiny,
    )
    powers = jnp.arange(degree, -1, -1, dtype=real_dtype)
    scaled_real = coefficients * bound**powers
    scale = jnp.max(jnp.abs(scaled_real))
    scaled = (scaled_real / jnp.maximum(scale, jnp.finfo(real_dtype).tiny)).astype(
        complex_dtype
    )
    derivative = scaled[:-1] * jnp.arange(degree, 0, -1, dtype=real_dtype)
    leading = jnp.maximum(jnp.abs(scaled[0]), jnp.finfo(real_dtype).eps)
    radius = 1.0 + jnp.max(jnp.abs(scaled[1:]) / leading)
    phase = 2.0j * jnp.pi * (
        jnp.arange(degree, dtype=real_dtype) + 0.25
    ) / degree
    roots = radius * jnp.exp(phase)
    epsilon = 10.0 * jnp.finfo(real_dtype).eps

    def step(_, current):
        values = jnp.polyval(scaled, current)
        slopes = jnp.polyval(derivative, current)
        differences = current[:, None] - current[None, :]
        off_diagonal = ~jnp.eye(degree, dtype=bool)
        guarded = jnp.where(off_diagonal, differences, 1.0 + 0.0j)
        repulsion = jnp.sum(jnp.where(off_diagonal, 1.0 / guarded, 0.0j), axis=-1)
        denominator = slopes - values * repulsion
        denominator = jnp.where(
            jnp.abs(denominator) > epsilon,
            denominator,
            slopes,
        )
        return current - values / denominator

    return jax.lax.fori_loop(0, 28, step, roots) * bound


@jax.custom_jvp
def fixed_ea28_roots(coefficients: Array, ordinate_bound: Array) -> Array:
    """Solve one polynomial with a fixed 28-step Ehrlich--Aberth schedule.

    ``ordinate_bound`` is a numerical scaling bound only.  The returned roots
    solve the original polynomial, and the custom JVP differentiates that
    equation directly so the bound does not enter the derivative.
    """

    return _fixed_ea28_impl(coefficients, ordinate_bound)


@fixed_ea28_roots.defjvp
def _fixed_ea28_roots_jvp(primals, tangents):
    coefficients, ordinate_bound = primals
    coefficient_tangent, _ = tangents
    roots = fixed_ea28_roots(coefficients, ordinate_bound)
    complex_coefficients = coefficients.astype(_complex_dtype(coefficients))
    complex_tangent = coefficient_tangent.astype(_complex_dtype(coefficients))
    degree = coefficients.shape[0] - 1
    derivative = complex_coefficients[:-1] * jnp.arange(
        degree, 0, -1, dtype=coefficients.real.dtype
    )
    numerator = jnp.polyval(complex_tangent, roots)
    denominator = jnp.polyval(derivative, roots)
    return roots, -numerator / denominator


def _fixed_ea_batch_impl(
    coefficients: Array,
    ordinate_bound: Array,
    iterations: int = 28,
) -> Array:
    """Vectorized implementation used by the hot polar radial path."""

    coefficients = jnp.asarray(coefficients)
    coefficient_count = coefficients.shape[-1]
    flat = coefficients.reshape((-1, coefficient_count))
    real_dtype = coefficients.real.dtype
    complex_dtype = _complex_dtype(coefficients)
    bounds = jnp.broadcast_to(
        jnp.asarray(ordinate_bound, dtype=real_dtype),
        coefficients.shape[:-1],
    ).reshape(-1)
    bounds = jnp.maximum(bounds, jnp.finfo(real_dtype).tiny)
    degree = coefficient_count - 1
    powers = jnp.arange(degree, -1, -1, dtype=real_dtype)
    scaled_real = flat * bounds[:, None] ** powers[None, :]
    scale = jnp.max(jnp.abs(scaled_real), axis=-1, keepdims=True)
    scaled = (
        scaled_real / jnp.maximum(scale, jnp.finfo(real_dtype).tiny)
    ).astype(complex_dtype)
    derivative = scaled[:, :-1] * jnp.arange(
        degree, 0, -1, dtype=real_dtype
    )
    leading = jnp.maximum(
        jnp.abs(scaled[:, :1]),
        jnp.finfo(real_dtype).eps,
    )
    radius = 1.0 + jnp.max(
        jnp.abs(scaled[:, 1:]) / leading,
        axis=-1,
    )
    phase = 2.0j * jnp.pi * (
        jnp.arange(degree, dtype=real_dtype) + 0.25
    ) / degree
    roots = radius[:, None] * jnp.exp(phase)[None, :]
    epsilon = 10.0 * jnp.finfo(real_dtype).eps

    def step(_, current):
        values = jax.vmap(jnp.polyval)(scaled, current)
        slopes = jax.vmap(jnp.polyval)(derivative, current)
        differences = current[:, :, None] - current[:, None, :]
        off_diagonal = ~jnp.eye(degree, dtype=bool)
        reciprocal = jnp.where(
            off_diagonal[None, :, :],
            1.0 / jnp.where(off_diagonal[None, :, :], differences, 1.0),
            0.0,
        )
        denominator = slopes - values * jnp.sum(reciprocal, axis=-1)
        denominator = jnp.where(
            jnp.abs(denominator) > epsilon,
            denominator,
            slopes,
        )
        return current - values / denominator

    roots = jax.lax.fori_loop(0, iterations, step, roots)
    return (roots * bounds[:, None]).reshape(
        coefficients.shape[:-1] + (coefficient_count - 1,)
    )


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def batched_fixed_ea_roots(
    coefficients: Array,
    ordinate_bound: Array,
    iterations: int = 28,
) -> Array:
    """Vectorize a fixed EA schedule over coefficient batches."""

    return _fixed_ea_batch_impl(coefficients, ordinate_bound, iterations)


@batched_fixed_ea_roots.defjvp
def _batched_fixed_ea_roots_jvp(iterations, primals, tangents):
    coefficients, ordinate_bound = primals
    coefficient_tangent, _ = tangents
    roots = batched_fixed_ea_roots(coefficients, ordinate_bound, iterations)
    complex_coefficients = coefficients.astype(_complex_dtype(coefficients))
    complex_tangent = coefficient_tangent.astype(_complex_dtype(coefficients))
    degree = coefficients.shape[-1] - 1
    derivative = complex_coefficients[..., :-1] * jnp.arange(
        degree,
        0,
        -1,
        dtype=coefficients.real.dtype,
    )
    numerator = jax.vmap(jnp.polyval)(
        complex_tangent.reshape((-1, degree + 1)),
        roots.reshape((-1, degree)),
    ).reshape(roots.shape)
    denominator = jax.vmap(jnp.polyval)(
        derivative.reshape((-1, degree)),
        roots.reshape((-1, degree)),
    ).reshape(roots.shape)
    return roots, -numerator / denominator


def batched_fixed_ea28_roots(
    coefficients: Array,
    ordinate_bound: Array,
) -> Array:
    """Compatibility wrapper for the original fixed 28-step schedule."""

    return batched_fixed_ea_roots(coefficients, ordinate_bound, 28)


__all__ = [
    "batched_companion_roots",
    "batched_fixed_ea_roots",
    "batched_fixed_ea28_roots",
    "batched_polished_real_companion_roots",
    "companion_roots",
    "fixed_ea28_roots",
    "polished_real_companion_roots",
    "real_companion_roots",
]
