import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, jit, vmap
from functools import partial
jax.config.update("jax_enable_x64", True)
import jax.random as jrandom
from microjax.poly_solver import poly_roots_EA_multi

def test_poly_roots_EA(coeffs_matrix, tol=1e-10):
    '''
    Test the poly_roots_EA function against jnp.roots.

    :param coeffs_matrix: Matrix of coefficients of the polynomials, highest order first.
                          Each row represents a different polynomial.
    :param max_iter: Maximum number of iterations for the Ehrlich-Aberth method.
    :param tol: Tolerance for comparing roots.
    :return: Boolean indicating whether the test passed or not.
    '''
    # Compute roots using the Ehrlich-Aberth method
    ea_roots = poly_roots_EA_multi(coeffs_matrix)
    ea_roots = jnp.array(ea_roots, dtype=complex)

    # Compute roots using jnp.roots for each polynomial separately
    numpy_roots = [np.roots(np.array(coeffs)) for coeffs in coeffs_matrix]
    np_roots = jnp.array(numpy_roots, dtype=complex)

    # Sort roots for comparison
    ea_roots_sorted  = jnp.array([jnp.sort_complex(roots) for roots in ea_roots])
    np_roots_sorted  = jnp.array([jnp.sort_complex(roots) for roots in np_roots])

    # Check if the roots are close within the tolerance
    return ea_roots_sorted, np_roots_sorted, jnp.allclose(ea_roots_sorted, np_roots_sorted, atol=tol)


# Test with random coefficients
#test_deg = 11
#length = int(1e+3)
#coeffs_matrix = jnp.array([np.random.uniform(-1.0, 1.0, test_deg) + 1j * np.random.uniform(-1.0, 1.0, test_deg) for i in range(length)])
#coeffs_matrix = jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) + 1.0j * jax.random.uniform(key, (length, test_deg), minval=-1.0, maxval=1.0) 

length = int(1e+3)
diff_max = []
diff_mean = []
test_degs = np.arange(5,42,2, dtype=int)
print(test_degs)
for test_deg in test_degs:
    key = jrandom.PRNGKey(0)
    coeffs_matrix = jnp.array([np.random.uniform(-1.0, 1.0, test_deg) + 1j * np.random.uniform(-1.0, 1.0, test_deg) for i in range(length)])
    test_passed = test_poly_roots_EA(coeffs_matrix)
    diffs = test_passed[0].ravel() - test_passed[1].ravel() 
    diff_max.append(np.max(jnp.abs(diffs)))
    diff_mean.append(np.mean(jnp.abs(diffs)))
    jax.device_get(test_passed)

diff_max  = np.array(diff_max)
diff_mean = np.array(diff_mean)
print("max :",diff_max)
print("mean:",diff_mean)

import matplotlib.pyplot as plt
plt.figure()
plt.title("test_multi_accuracy.py")
plt.plot(test_degs, diff_max,label="max")
plt.plot(test_degs, diff_mean,label="mean")
plt.yscale("log")
plt.grid(ls=":")
plt.legend()
plt.xlabel("number of polynomial")
plt.ylabel("difference in 1000 equations")
plt.savefig("tests/integrate/poly_solve/test_multi_accuracy.eps", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

#test_passed = test_poly_roots_EA(coeffs_matrix)
#print("DF:", jnp.sum(test_passed[0].ravel() - test_passed[1].ravel()))
#print("Test passed:", test_passed[2])