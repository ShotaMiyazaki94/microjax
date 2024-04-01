import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial

@partial(jit, static_argnums=(1,))
def poly_roots_EA(coeffs, max_iter=25, init_roots=None):
    """
    Ehrlich-Aberth method using JAX for finding all roots of a complex polynomial.

    :param coeffs: Coefficients of the polynomial, highest order first.
    :param max_iter:    Maximum number of iterations. 
                        This should be conservative to be 20 for n_coeffs=6 and 30 for n_coeffs=11
    :return: Approximation of all roots of the polynomial.
    """

    n = len(coeffs) - 1
    max_coeffs = jnp.max(jnp.abs(coeffs))
    
    if init_roots is not None and len(init_roots)==n:
        initial_roots = init_roots
    else:
        initial_roots = (max_coeffs+1.)*jnp.exp(2j * jnp.pi * jnp.arange(n) / n)    
    
    roots, _ = lax.scan(lambda roots, _: EA_step(roots, coeffs), initial_roots, jnp.arange(max_iter))
    return roots

def EA_step(roots, coeffs):
    """
    Perform one step of the Ehrlich-Aberth iteration.

    :param roots: Current roots estimates.
    :param coeffs: Coefficients of the polynomial.
    :return: Updated roots.
    """
    poly_vals = jnp.polyval(coeffs, roots)
    dpoly_vals = jnp.polyval(jnp.polyder(coeffs), roots)

    # Calculate pairwise differences and their inverses
    root_diffs = roots[:, None] - roots
    inv_root_diffs = jnp.where(root_diffs == 0, 0, 1 / root_diffs)  # Avoid division by zero

    # Sum of inverse differences for each root
    sum_inv_diffs = jnp.sum(inv_root_diffs, axis=1)

    # Calculate updates for each root
    updates = poly_vals / (dpoly_vals - poly_vals * sum_inv_diffs)
    return roots - updates, None