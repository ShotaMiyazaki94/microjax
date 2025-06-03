import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

params_init = jnp.array([6.90022772e+03 - 6900.0, jnp.log10(2.32698878e+01), -1.34886439e-01,
                         -3.34134233e+00, -1.25602528e-01, -2.20540555e-01,
                         -3.21033816e+00, 4.71741660e-02, -2.46430115e+00, 
                         4.23139645e-01, 5.50070356e-02, jnp.log10(0.001)], dtype=jnp.float64)


fisher_matrix = np.load("example/OGLE-2014-BLG-1722/fisher_matrix.npy")
damping = 1e-3  # Damping factor for numerical stability
fisher_matrix_pd = fisher_matrix + damping * np.eye(fisher_matrix.shape[0])
fisher_cov = np.linalg.inv(fisher_matrix_pd)
print("Fisher Matrix:")
print(fisher_matrix_pd)
print("Fisher Covariance Matrix:")
print(fisher_cov)
param_stddev = np.sqrt(np.diag(fisher_cov))
keys = ["t0_diff", "log_tE", "u0", "log_q", "log_s", "alpha", "log_q3", "log_s2", "psi", "piEN", "piEE", "log_rho"]
for key, param, sigma in zip(keys, params_init, param_stddev):
    print(f"{key}: {param:.6f} ± {sigma:.6f}")
L = jnp.linalg.cholesky(fisher_cov)
print("Cholesky Decomposition of Fisher Covariance Matrix:")
print(L)

exit(1)


eigvals, eigvecs = np.linalg.eigh(fisher_matrix)
print("Eigenvalues of Fisher Matrix:")
print(eigvals)
eigvals_clipped = np.clip(eigvals, 1e-6, None)
fisher_matrix_pd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
fisher_cov_pd = np.linalg.inv(fisher_matrix_pd)
L = jnp.linalg.cholesky(fisher_cov_pd)
print("Fisher Matrix (Positive Definite):")
print(fisher_matrix_pd)
print("Fisher Covariance Matrix (Positive Definite):")
print(fisher_cov_pd)
print("Cholesky Decomposition of Fisher Covariance Matrix:")
print(L)

#print("Fisher Matrix:")
#print(fisher)
#fisher_cov = np.linalg.inv(fisher + 1e-12 * np.eye(fisher.shape[0]))  # Adding a small value for numerical stability
#print("Fisher Covariance Matrix:")
#print(fisher_cov)
#fisher_cov = 0.5 * (fisher_cov + fisher_cov.T)
#L = jnp.linalg.cholesky(fisher_cov)  # Adding a small value for numerical stability
#print("Cholesky Decomposition of Fisher Covariance Matrix:")
#print(L)