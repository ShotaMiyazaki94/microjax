"""Root finder utilities based on the Ehrlich–Aberth method.

This module implements a JAX-friendly version of the Ehrlich–Aberth (EA)
iteration to find all complex roots of a polynomial given in `polyval` order.

Design highlights
- Iteration uses `lax.while_loop` for early stopping by tolerance.
- Gradients are provided via implicit differentiation using `custom_jvp`,
  making reverse/forward AD robust regardless of the number of iterations.
- The derivative polynomial is precomputed from coefficients to avoid
  constructing it at every step; no reliance on `jnp.polyder`.

Notes on numerical behavior
- For nearly real coefficients, tiny imaginary parts in the output roots are
  zeroed (threshold ~ 10*tol) purely for presentation; the solver remains fully
  complex and differentiable.
- For ill-conditioned cases (e.g., near-multiple roots), the method can slow
  down; the implementation falls back to a Newton step when the EA denominator
  gets too small to improve stability.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial
from jax import custom_jvp

# Default iteration cap for EA method; used if not specified.
_DEFAULT_MAX_ITER = 100

__all__ = [
    "poly_roots",
    "poly_roots_EA",
    "poly_roots_EA_multi",
]


def _as_complex(x: jax.Array) -> jax.Array:
    """Ensure complex dtype so real inputs can hold complex roots.

    Parameters
    - x: array-like; any real/complex dtype.

    Returns
    - Complex-typed array with a precision matching the input.
    """
    if jnp.iscomplexobj(x):
        return x
    # Promote to complex matching float precision
    dtype = jnp.result_type(x, jnp.complex64)
    return x.astype(dtype)


def _polyder_coeffs(coeffs: jax.Array) -> jax.Array:
    """Return derivative coefficients in `polyval` convention.

    If `coeffs = [c0, c1, ..., cn]` represents `c0*x^n + ... + cn`, then
    `d/dx coeffs = [c0*n, c1*(n-1), ..., c_{n-1}*1]`.

    For constant polynomials (degree 0), returns a zero array broadcast to the
    input shape.
    """
    n = coeffs.shape[-1] - 1
    if n <= 0:
        return jnp.zeros_like(coeffs)
    powers = jnp.arange(n, 0, -1, dtype=coeffs.real.dtype)
    return coeffs[..., :-1] * powers


def _cauchy_radius(coeffs: jax.Array) -> jax.Array:
    """Cauchy root bound radius using the leading coefficient.

    Parameters
    - coeffs: 1D array `[c0, c1, ..., cn]` in `polyval` order.

    Returns
    - A non-negative scalar array giving the radius R s.t. all roots lie in
      `|z| <= R` (simple bound `1 + max |c_i/c0|`).
    """
    c0 = coeffs[..., 0]
    eps = jnp.finfo(coeffs.real.dtype).eps
    denom = jnp.maximum(jnp.abs(c0), eps)
    # 1 + max |c_i / c0|
    r = 1.0 + jnp.max(jnp.abs(coeffs[..., 1:]) / denom[..., None], axis=-1)
    return r


def _ea_step(
    roots: jax.Array,
    coeffs: jax.Array,
    dcoeffs: jax.Array,
    eps: float,
) -> Tuple[jax.Array, jax.Array]:
    """Perform one Ehrlich–Aberth iteration step.

    Parameters
    - roots: shape `(n,)`, current root estimates.
    - coeffs: shape `(n+1,)`, polynomial coefficients (polyval order).
    - dcoeffs: shape `(n,)`, derivative coefficients (polyval order).
    - eps: small positive threshold to guard tiny denominators.

    Returns
    - new_roots: shape `(n,)`, updated root estimates.
    - update: shape `(n,)`, the applied update (for convergence checks).
    """
    # Evaluate polynomial and derivative at current roots
    f = jnp.polyval(coeffs, roots)
    fp = jnp.polyval(dcoeffs, roots)
    # Pairwise differences and reciprocal sum
    diffs = roots[:, None] - roots[None, :]
    inv_diffs = jnp.where(diffs == 0, 0.0 + 0.0j, 1.0 / diffs)
    s = jnp.sum(inv_diffs, axis=1)
    # Denominator stabilization: if nearly zero, fall back to Newton step
    den = fp - f * s
    den = jnp.where(jnp.abs(den) < eps, fp, den)
    update = f / den
    new_roots = roots - update
    return new_roots, update


@custom_jvp
def poly_roots_EA(
    coeffs: jax.Array,
    initial_roots: Optional[jax.Array] = None,
    tol: float = 1e-12,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> jax.Array:
    """Compute all roots via the Ehrlich–Aberth method with early stopping.

    Parameters
    - coeffs: shape `(n+1,)`, polynomial coefficients `[c0, ..., cn]` with
      `polyval` order (i.e., `c0*x^n + ... + cn`). Real or complex.
    - initial_roots: optional, shape `(n,)` complex initial guesses. If not
      provided, points are placed uniformly on the circle given by a Cauchy
      bound.
    - tol: absolute update tolerance per root. A root is considered converged
      when `|Δ| <= tol`.
    - max_iter: maximum number of iterations.

    Returns
    - roots: shape `(n,)`, complex roots in unspecified order.

    AD behavior
    - Reverse/forward AD is supported via implicit differentiation (`custom_jvp`).
      The gradients do not depend on the number of EA iterations.
    """
    coeffs = _as_complex(coeffs)
    n = coeffs.shape[0] - 1
    assert n >= 1, "Polynomial degree must be >= 1"

    # Initial roots by Cauchy bound on a circle
    if initial_roots is None:
        r = _cauchy_radius(coeffs)
        k = jnp.arange(n)
        initial_roots = r * jnp.exp(2j * jnp.pi * k / n)
    else:
        initial_roots = _as_complex(initial_roots)

    dcoeffs = _polyder_coeffs(coeffs)
    eps = jnp.finfo(initial_roots.real.dtype).eps * 10

    def cond_fun(carry):
        it, roots, done = carry
        return jnp.logical_and(it < max_iter, jnp.any(~done))

    def body_fun(carry):
        it, roots, done = carry
        new_roots, update = _ea_step(roots, coeffs, dcoeffs, eps)
        err = jnp.abs(update)
        done_now = err <= tol
        return (it + 1, jnp.where(done[:, None], roots[:, None], new_roots[:, None]).squeeze(-1), done | done_now)

    init_done = jnp.zeros((n,), dtype=bool)
    init_state = (jnp.array(0, dtype=jnp.int32), initial_roots, init_done)
    _, roots, _ = lax.while_loop(cond_fun, body_fun, init_state)

    # If coefficients are (numerically) real, drop tiny imaginary parts in roots
    def _clean_real_roots(roots: jax.Array, coeffs: jax.Array) -> jax.Array:
        """Zero-out tiny imaginary parts for nearly real-coefficient polynomials."""
        imag = jnp.imag(coeffs)
        all_real = jnp.all(jnp.abs(imag) < 10 * tol)
        def clean(r):
            im = jnp.imag(r)
            re = jnp.real(r)
            im_clean = jnp.where(jnp.abs(im) < 10 * tol, 0.0, im)
            return re + 1j * im_clean
        return lax.cond(all_real, clean, lambda r: r, roots)

    roots = _clean_real_roots(roots, coeffs)
    return roots


@poly_roots_EA.defjvp
def _poly_roots_EA_jvp(primals, tangents):
    coeffs, initial_roots, tol, max_iter = primals
    dcoeffs, _, _, _ = tangents

    roots = poly_roots_EA(coeffs, initial_roots, tol, max_iter)
    # Implicit differentiation: f(roots; a) = 0
    dcoeffs = _as_complex(dcoeffs)
    n = coeffs.shape[0] - 1
    # derivative wrt z
    dcoeffs_poly = _polyder_coeffs(_as_complex(coeffs))
    df_dz = jnp.polyval(dcoeffs_poly, roots)
    # derivative wrt coefficients (Vandermonde order matches coeff order)
    exps = jnp.arange(n, -1, -1)  # integer exponents
    V = roots[:, None] ** exps[None, :]
    dz = - (V @ dcoeffs) / df_dz
    return roots, dz


def poly_roots_EA_multi(
    coeffs_matrix: jax.Array,
    custom_init: bool = False,
    initial_roots_matrix: Optional[jax.Array] = None,
    tol: float = 1e-12,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> jax.Array:
    """Vectorized EA solver for batches of polynomials.

    Parameters
    - coeffs_matrix: array with shape `[..., n+1]`.
    - custom_init: whether to use `initial_roots_matrix` as initial guesses.
    - initial_roots_matrix: array with shape `[..., n]` if `custom_init=True`.
    - tol, max_iter: forwarded to `poly_roots_EA`.

    Returns
    - roots: array with shape `[..., n]`.
    """
    if custom_init:
        return jax.vmap(lambda c, r: poly_roots_EA(c, r, tol, max_iter))(coeffs_matrix, initial_roots_matrix)
    else:
        return jax.vmap(lambda c: poly_roots_EA(c, None, tol, max_iter))(coeffs_matrix)


@partial(jit, static_argnames=("custom_init",))
def poly_roots(
    coeffs: jax.Array,
    custom_init: bool = False,
    roots_init: Optional[jax.Array] = None,
) -> jax.Array:
    """Find all roots for a batch of polynomials using EA with early stop.

    Parameters
    - coeffs: array with shape `[..., n+1]` (polyval order). Real/complex.
    - custom_init: whether to use `roots_init` as initial guesses.
    - roots_init: array with shape `[..., n]` if `custom_init=True`.

    Returns
    - roots: array with shape `[..., n]` of complex roots in unspecified order.
    """
    ncoeffs = coeffs.shape[-1]
    output_shape = coeffs.shape[:-1] + (ncoeffs - 1,)
    coeffs_flat = coeffs.reshape((-1, ncoeffs))
    if custom_init:
        roots_init = roots_init.reshape((coeffs_flat.shape[0], ncoeffs - 1))
        roots = poly_roots_EA_multi(coeffs_flat, custom_init=True, initial_roots_matrix=roots_init)
    else:
        roots = poly_roots_EA_multi(coeffs_flat)
    return roots.reshape(output_shape)
