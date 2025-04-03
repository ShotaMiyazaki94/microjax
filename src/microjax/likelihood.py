import jax.numpy as jnp
from jax.scipy.linalg import solve

def nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb):
    """
    Calculate the simplified negative log-likelihood (NLL) for a microlensing model
    under a Gaussian linear fs and fb with diagonal priors.
    Parameters:
        flux       : (n,) array, observed fluxes
        M          : (n, 2) array, design matrix with columns [fs_model, 1] (fs: source flux, fb: blend flux)
        sigma2_obs : (n,) array, variance of observational errors
        sigma2_fs  : float, prior variance for source flux fs
        sigma2_fb  : float, prior variance for blend flux fb
    Returns:
        scalar : negative log-likelihood
    """
    # Inverse prior covariances (regularization terms)
    lambda_fs = 1.0 / sigma2_fs
    lambda_fb = 1.0 / sigma2_fb
    Lambda_inv = jnp.diag(jnp.array([lambda_fs, lambda_fb]))  # shape (2, 2)
    
    # Inverse observational covariance applied to design matrix
    Cinv_M = M / sigma2_obs[:, None]            # shape (n, 2)
    Mt_Cinv_M = Cinv_M.T @ M                    # shape (2, 2)

    # Posterior precision matrix
    A = Lambda_inv + Mt_Cinv_M                  # shape (2, 2)

    # Data term: M.T @ C^-1 @ flux
    Cinv_flux = flux / sigma2_obs               # shape (n,)
    Mt_Cinv_flux = M.T @ Cinv_flux              # shape (2,)

    # Mahalanobis term (residual quadratic form)
    posterior_mean = jnp.linalg.solve(A, Mt_Cinv_flux)  # shape (2,)
    mahal_term = jnp.dot(flux, Cinv_flux) - Mt_Cinv_flux @ posterior_mean 

    # Log-determinant terms
    logdet_C = jnp.sum(jnp.log(sigma2_obs))                     # log|C|, scalar
    logdet_Lambda = jnp.log(sigma2_fs) + jnp.log(sigma2_fb)     # log|Λ|, scalar
    _, logdet_A = jnp.linalg.slogdet(A)                         # log|A|, scalar

    # Final negative log-likelihood
    nll = 0.5 * (mahal_term + logdet_C + logdet_Lambda + logdet_A)
    return nll

def nll_ulens_general(flux, M, C, mu, Lambda):
    """
    Calculate the negative log-likelihood (NLL) for a microlensing model
    with general Gaussian priors and data covariance.
    Parameters:
        flux   : (n,) array, observed fluxes
        M      : (n, 2) array, design matrix (e.g., columns for fs and fb)
        C      : (n, n) array, data covariance matrix
        mu     : (2,) array, prior mean for parameters [fs, fb]
        Lambda : (2, 2) array, prior covariance matrix for [fs, fb]
    Returns:
        scalar : negative log-likelihood
    """
    # Inverse of prior covariance
    Lambda_inv = jnp.linalg.inv(Lambda)            # shape (2, 2)

    # Solve C⁻¹ @ M efficiently
    Cinv_M = solve(C, M)                            # shape (n, 2)
    Mt_Cinv_M = M.T @ Cinv_M                        # shape (2, 2)

    # Posterior precision matrix
    A = Lambda_inv + Mt_Cinv_M                      # shape (2, 2)

    # Residual vector r = flux - M @ mu
    r = flux - M @ mu                               # shape (n,)

    # Solve C⁻¹ @ r efficiently
    Cinv_r = solve(C, r)                            # shape (n,)
    Mt_Cinv_r = M.T @ Cinv_r                        # shape (2,)

    # Posterior mean (optional, not returned but used for Mahalanobis term)
    posterior_mean = jnp.linalg.solve(A, Mt_Cinv_r) # shape (2,)

    # Mahalanobis term
    mahal_term = jnp.dot(r, Cinv_r) - Mt_Cinv_r @ posterior_mean

    # Log-determinant terms
    _, logdet_C = jnp.linalg.slogdet(C)
    _, logdet_Lambda = jnp.linalg.slogdet(Lambda)
    _, logdet_A = jnp.linalg.slogdet(A)

    # Final negative log-likelihood
    nll = 0.5 * (mahal_term + logdet_C + logdet_Lambda + logdet_A)
    return nll