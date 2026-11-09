"""Real companion roots with polynomial polishing and implicit derivatives."""

from __future__ import annotations
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def _complex_dtype(values: Array):
    return jnp.result_type(values, jnp.complex64)


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
    matrix = matrix.at[1:, :-1].set(jnp.eye(degree - 1, dtype=coefficients.dtype))
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
