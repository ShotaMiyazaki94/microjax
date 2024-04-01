import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
import jax.random as jrandom
from microjax.poly_solver import poly_roots_EA_multi

def test_poly_roots_EA(coeffs_matrix, tol=1e-10):
    """
    Test the poly_roots_EA function against jnp.roots.

    :param coeffs_matrix: Matrix of coefficients of the polynomials, highest order first.
                          Each row represents a different polynomial.
    :param max_iter: Maximum number of iterations for the Ehrlich-Aberth method.
    :param tol: Tolerance for comparing roots.
    :return: Boolean indicating whether the test passed or not.
    """
    # Compute roots using the Ehrlich-Aberth method
    ea_roots = poly_roots_EA_multi(coeffs_matrix)
    ea_roots = jnp.array(ea_roots, dtype=complex)

    # Compute roots using jnp.roots for each polynomial separately
    numpy_roots = [np.roots(coeffs) for coeffs in coeffs_matrix]
    np_roots = jnp.array(numpy_roots, dtype=complex)

    # Sort roots for comparison
    ea_roots_sorted  = jnp.array([jnp.sort_complex(roots) for roots in ea_roots])
    np_roots_sorted  = jnp.array([jnp.sort_complex(roots) for roots in np_roots])

    # Check if the roots are close within the tolerance
    return ea_roots_sorted, np_roots_sorted, jnp.allclose(ea_roots_sorted, np_roots_sorted, atol=tol)


# Test with random coefficients
key = jrandom.PRNGKey(0)
test_deg = 10
length = int(10000)
coeffs_matrix = jnp.array([np.random.uniform(-10, 10, test_deg) + 1j * np.random.uniform(-10, 10, test_deg) for i in range(length)])

test_passed = test_poly_roots_EA(coeffs_matrix)
print("EA:", test_passed[0].ravel())
print("NP:", test_passed[1].ravel())
print("DF:", jnp.sum(test_passed[0].ravel() - test_passed[1].ravel()))
print("Test passed:", test_passed[2])