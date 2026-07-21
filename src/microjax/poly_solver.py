"""Root finder utilities based on the Ehrlich–Aberth method.

This module implements a JAX-friendly version of the Ehrlich–Aberth (EA)
iteration to find all complex roots of a polynomial given in ``polyval`` order.

Design highlights
-----------------

- Iteration uses :func:`jax.lax.while_loop` for early stopping by tolerance.
- Gradients are provided via implicit differentiation using ``custom_jvp``,
  making reverse/forward AD robust regardless of the number of iterations.
- The derivative polynomial is precomputed from coefficients to avoid
  constructing it at every step; no reliance on ``jnp.polyder``.

Notes on numerical behavior
---------------------------

- For nearly real coefficients, tiny imaginary parts in the output roots are
  zeroed (threshold ~ ``10 * tol``) purely for presentation; the solver remains
  fully complex and differentiable.
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
    "poly_roots_self_inversive_robust_fixed",
    "poly_roots_self_inversive_fixed",
    "poly_roots_EA",
    "poly_roots_EA_self_inversive_robust_fixed",
    "poly_roots_EA_self_inversive_fixed",
    "poly_roots_EA_multi",
]

_FIXED_EA_ITERATIONS = 20
_ROBUST_EA_ITERATIONS = 40


def _as_complex(x: jax.Array) -> jax.Array:
    """Promote real inputs to a complex dtype compatible with root finding.

    Parameters
    ----------
    x : jax.Array
        Input array with real or complex dtype.

    Returns
    -------
    jax.Array
        Array with complex dtype matching the precision of ``x``.
    """
    if jnp.iscomplexobj(x):
        return x
    # Promote to complex matching float precision
    dtype = jnp.result_type(x, jnp.complex64)
    return x.astype(dtype)


def _polyder_coeffs(coeffs: jax.Array) -> jax.Array:
    """Return derivative coefficients in ``jnp.polyval`` convention.

    Parameters
    ----------
    coeffs : jax.Array
        Coefficient array ``[c0, c1, ..., cn]`` representing
        ``c0 * x**n + ... + cn``.

    Returns
    -------
    jax.Array
        Derivative coefficients ``[c0 * n, c1 * (n-1), ..., c_{n-1}]`` with the
        same broadcasted shape as ``coeffs``.

    Notes
    -----
    Constant polynomials (degree 0) yield a zero array broadcast to the input
    shape.
    """
    n = coeffs.shape[-1] - 1
    if n <= 0:
        return jnp.zeros_like(coeffs)
    powers = jnp.arange(n, 0, -1, dtype=coeffs.real.dtype)
    return coeffs[..., :-1] * powers


def _cauchy_radius(coeffs: jax.Array) -> jax.Array:
    """Compute a Cauchy bound for the polynomial roots.

    Parameters
    ----------
    coeffs : jax.Array
        Coefficient array ``[c0, c1, ..., cn]`` in ``polyval`` order.

    Returns
    -------
    jax.Array
        Non-negative scalar giving the radius ``R`` such that all roots lie
        within ``|z| <= R`` using the bound ``1 + max |c_i / c0|``.
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
    ----------
    roots : jax.Array
        Current root estimates with shape ``(n,)``.
    coeffs : jax.Array
        Polynomial coefficients in ``polyval`` order with shape ``(n+1,)``.
    dcoeffs : jax.Array
        Derivative coefficients in ``polyval`` order with shape ``(n,)``.
    eps : float
        Small positive threshold used to guard nearly singular denominators.

    Returns
    -------
    new_roots : jax.Array
        Updated root estimates with shape ``(n,)``.
    update : jax.Array
        The applied update (``roots - new_roots``) used for convergence tests.
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


def _poly_roots_EA_self_inversive_impl(
    coeffs: jax.Array, iterations: int
) -> jax.Array:
    """Solve one self-inversive polynomial with a static EA iteration count."""

    coeffs = _as_complex(coeffs)
    n = coeffs.shape[0] - 1
    assert n >= 1, "Polynomial degree must be >= 1"
    radius = jnp.asarray(1.1, dtype=coeffs.real.dtype)
    initial_roots = radius * jnp.exp(
        2j * jnp.pi * (jnp.arange(n) + 0.25) / n
    )
    derivative_coeffs = _polyder_coeffs(coeffs)
    eps = jnp.finfo(initial_roots.real.dtype).eps * 10.0

    def step(_, roots):
        updated, _ = _ea_step(roots, coeffs, derivative_coeffs, eps)
        return updated

    return lax.fori_loop(0, iterations, step, initial_roots)


def _self_inversive_root_jvp(solver, primals, tangents):
    """Implicit polynomial-root derivative shared by fixed EA schedules."""

    (coeffs,) = primals
    (coeffs_tangent,) = tangents
    roots = solver(coeffs)
    coeffs_complex = _as_complex(coeffs)
    tangent_complex = _as_complex(coeffs_tangent)
    derivative_coeffs = _polyder_coeffs(coeffs_complex)
    derivative_at_roots = jnp.polyval(derivative_coeffs, roots)
    degree = coeffs.shape[0] - 1
    exponents = jnp.arange(degree, -1, -1)
    vandermonde = roots[:, None] ** exponents[None, :]
    roots_tangent = -(
        vandermonde @ tangent_complex
    ) / derivative_at_roots
    return roots, roots_tangent


@custom_jvp
def poly_roots_EA_self_inversive_fixed(coeffs: jax.Array) -> jax.Array:
    """Solve one polynomial with 20 fixed accelerator-friendly EA steps.

    This schedule is intended for large homogeneous first-pass batches. Rare
    failures remain explicit in downstream residual checks and can be compacted
    into the robust 40-step schedule instead of making every GPU lane pay its
    cost.
    """

    return _poly_roots_EA_self_inversive_impl(
        coeffs, _FIXED_EA_ITERATIONS
    )


@poly_roots_EA_self_inversive_fixed.defjvp
def _poly_roots_EA_self_inversive_fixed_jvp(primals, tangents):
    return _self_inversive_root_jvp(
        poly_roots_EA_self_inversive_fixed, primals, tangents
    )


@custom_jvp
def poly_roots_EA_self_inversive_robust_fixed(
    coeffs: jax.Array,
) -> jax.Array:
    """Solve one difficult polynomial with 40 fixed EA steps."""

    return _poly_roots_EA_self_inversive_impl(
        coeffs, _ROBUST_EA_ITERATIONS
    )


@poly_roots_EA_self_inversive_robust_fixed.defjvp
def _poly_roots_EA_self_inversive_robust_fixed_jvp(primals, tangents):
    return _self_inversive_root_jvp(
        poly_roots_EA_self_inversive_robust_fixed, primals, tangents
    )


@jit
def poly_roots_self_inversive_fixed(coeffs: jax.Array) -> jax.Array:
    """Vectorize the fixed self-inversive solver over coefficient batches."""

    ncoeffs = coeffs.shape[-1]
    output_shape = coeffs.shape[:-1] + (ncoeffs - 1,)
    coeffs_flat = coeffs.reshape((-1, ncoeffs))
    roots = jax.vmap(poly_roots_EA_self_inversive_fixed)(coeffs_flat)
    return roots.reshape(output_shape)


@jit
def poly_roots_self_inversive_robust_fixed(
    coeffs: jax.Array,
) -> jax.Array:
    """Vectorize the robust fixed solver over coefficient batches."""

    ncoeffs = coeffs.shape[-1]
    output_shape = coeffs.shape[:-1] + (ncoeffs - 1,)
    coeffs_flat = coeffs.reshape((-1, ncoeffs))
    roots = jax.vmap(poly_roots_EA_self_inversive_robust_fixed)(coeffs_flat)
    return roots.reshape(output_shape)


@custom_jvp
def poly_roots_EA(
    coeffs: jax.Array,
    initial_roots: Optional[jax.Array] = None,
    tol: float = 1e-12,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> jax.Array:
    """Compute all roots via the Ehrlich–Aberth method with early stopping.

    Parameters
    ----------
    coeffs : jax.Array
        Polynomial coefficients with shape ``(n+1,)`` in ``polyval`` order
        (``c0 * x**n + ... + cn``). Real or complex values are supported.
    initial_roots : Optional[jax.Array], optional
        Complex initial guesses with shape ``(n,)``. When ``None`` (default),
        the solver places points uniformly on the circle defined by a Cauchy
        bound.
    tol : float, optional
        Absolute update tolerance per root. A root is considered converged when
        ``abs(delta) <= tol``. Defaults to ``1e-12``.
    max_iter : int, optional
        Maximum number of iterations. Defaults to ``_DEFAULT_MAX_ITER``.

    Returns
    -------
    jax.Array
        Complex roots with shape ``(n,)`` in unspecified order.

    Notes
    -----
    Reverse/forward automatic differentiation is supported via implicit
    differentiation (``custom_jvp``); gradients do not depend on the number of
    Ehrlich–Aberth iterations executed.
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
    ----------
    coeffs_matrix : jax.Array
        Coefficient array with shape ``[..., n+1]``.
    custom_init : bool, optional
        Use ``initial_roots_matrix`` as initial guesses when ``True``.
    initial_roots_matrix : Optional[jax.Array], optional
        Initial guesses with shape ``[..., n]`` if ``custom_init`` is ``True``.
    tol : float, optional
        Absolute update tolerance per root; forwarded to
        :func:`poly_roots_EA`. Defaults to ``1e-12``.
    max_iter : int, optional
        Maximum number of iterations per root; forwarded to
        :func:`poly_roots_EA`. Defaults to ``_DEFAULT_MAX_ITER``.

    Returns
    -------
    jax.Array
        Roots with shape ``[..., n]``.
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
    ----------
    coeffs : jax.Array
        Coefficient array with shape ``[..., n+1]`` in ``polyval`` order. Real
        or complex values are supported.
    custom_init : bool, optional
        Pass ``True`` to feed ``roots_init`` as custom initial guesses.
    roots_init : Optional[jax.Array], optional
        Initial guesses with shape ``[..., n]`` used when ``custom_init`` is
        ``True``.

    Returns
    -------
    jax.Array
        Complex roots with shape ``[..., n]`` in unspecified order.
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
