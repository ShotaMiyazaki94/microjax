import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
from jax import custom_jvp

# This structure should be modified...
max_iter=100

@partial(jit, static_argnames=("custom_init"))
def poly_roots(coeffs, custom_init=False, roots_init=None):
    ncoeffs = coeffs.shape[-1]
    output_shape = coeffs.shape[:-1] + (ncoeffs - 1,)
    coeffs_flat = coeffs.reshape((-1, ncoeffs))
    
    if custom_init:
        roots_init = roots_init.reshape((coeffs_flat.shape[0], ncoeffs - 1))
        roots = poly_roots_EA_multi(coeffs_flat, custom_init=True, initial_roots_matrix=roots_init)
    else:
        roots = poly_roots_EA_multi(coeffs_flat)
    
    return roots.reshape(output_shape) 

def EA_step(roots, coeffs):
    """
    Perform one step of the Ehrlich-Aberth iteration using JAX optimizations.
    """
    poly_vals = jnp.polyval(coeffs, roots)
    dpoly_vals = jnp.polyval(jnp.polyder(coeffs), roots)
    root_diffs = roots[:, None] - roots
    inv_root_diffs = jnp.where(root_diffs == 0, 0, 1.0 / root_diffs)
    sum_inv_diffs = jnp.sum(inv_root_diffs, axis=1)
    updates = poly_vals / (dpoly_vals - poly_vals * sum_inv_diffs)
    return roots - updates, None

@custom_jvp
def poly_roots_EA(coeffs, initial_roots=None):
    """
    Enhanced Ehrlich-Aberth method using JAX for finding all roots of a complex polynomial.
    Allows specifying initial roots.
    """
    n = len(coeffs) - 1
    if initial_roots is None:
        radius = jnp.sqrt(jnp.sum(jnp.abs(coeffs[:-1]) / jnp.abs(coeffs[-1])))  # Cauchy's bound
        initial_roots = radius * jnp.exp(2j * jnp.pi * jnp.arange(n) / n)
    roots, _ = lax.scan(lambda roots, _: EA_step(roots, coeffs), initial_roots, jnp.arange(max_iter))
    return roots

@poly_roots_EA.defjvp
def poly_roots_EA_jvp(primals, tangents):
    coeffs, initial_roots = primals
    dcoeffs, _ = tangents
    roots = poly_roots_EA(coeffs, initial_roots)
    df_dz = jnp.polyval(jnp.polyder(coeffs), roots)
    df_da = jnp.vstack([roots**i for i in range(coeffs.size-1, -1, -1)]).T 
    dz = - jnp.dot(df_da, dcoeffs) / df_dz
    return roots, dz

def poly_roots_EA_multi(coeffs_matrix, custom_init=False, initial_roots_matrix=None):
    """
    Process multiple sets of coefficients for root finding with optional initial roots.
    """
    ncoeffs = coeffs_matrix.shape[-1]
    output_shape = coeffs_matrix.shape[:-1] + (ncoeffs - 1,) 
    
    if custom_init:
        roots_matrix = jax.vmap(poly_roots_EA, in_axes=(0, 0))(coeffs_matrix, initial_roots_matrix)
    else:
        roots_matrix = jax.vmap(poly_roots_EA, in_axes=(0,))(coeffs_matrix)
    return roots_matrix.reshape(output_shape)
