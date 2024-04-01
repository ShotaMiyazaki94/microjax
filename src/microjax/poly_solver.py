import jax
import jax.numpy as jnp
from jax import lax, jit
from functools import partial

@partial(jit, static_argnums=(1,))
def poly_roots_EA_multi(coeffs_matrix, max_iter=25):
    """
    Ehrlich-Aberth method using JAX for finding all roots of multiple complex polynomials.

    :param coeffs_matrix: Matrix of coefficients of the polynomials, highest order first.
                          Each row represents a different polynomial.
    :param max_iter: Maximum number of iterations.
    :return: Approximation of all roots of the polynomials.
    """
    ncoeffs = coeffs_matrix.shape[-1]
    output_shape = coeffs_matrix.shape[:-1] + (ncoeffs - 1,)

    def single_poly_roots(coeffs, max_iter):
        n = len(coeffs) - 1
        max_coeffs = jnp.max(jnp.abs(coeffs))
        initial_roots = (max_coeffs + 1.) * jnp.exp(2j * jnp.pi * jnp.arange(n) / n)

        roots, _ = lax.scan(lambda roots, _: EA_step(roots, coeffs), initial_roots, jnp.arange(max_iter))
        return roots

    # Apply the single polynomial root finding function to each row of the input matrix
    roots_matrix = jax.vmap(single_poly_roots, in_axes=(0, None,))(coeffs_matrix, max_iter)
    
    return roots_matrix
    #return roots_matrix.reshape(output_shape)

@partial(jit, static_argnums=(1,))
def poly_roots_EA(coeffs, max_iter=25, init_roots=None):
    """
    Ehrlich-Aberth method using JAX for finding all roots of a complex polynomial.

    :param coeffs: Coefficients of the polynomial, highest order first. coeffs should be -10<Re(z)<10, -10<Im(z)<10,  
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
    inv_root_diffs = jnp.where(root_diffs == 0, 0, 1.0 / root_diffs)  # Avoid division by zero

    # Sum of inverse differences for each root
    sum_inv_diffs = jnp.sum(inv_root_diffs, axis=1)

    # Calculate updates for each root
    updates = poly_vals / (dpoly_vals - poly_vals * sum_inv_diffs)
    return roots - updates, None