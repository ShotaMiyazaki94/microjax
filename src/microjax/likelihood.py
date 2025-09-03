"""Likelihood and simple linear fits for microlensing photometry.

This module provides small, JAX-friendly utilities to compute weighted linear
least-squares fits and negative log-likelihoods (NLL) under Gaussian models.
Two NLL variants are implemented:

- ``nll_ulens``: fast path with diagonal observational noise and diagonal
  Gaussian priors on source and blend fluxes.
- ``nll_ulens_general``: full-rank observational covariance and arbitrary
  Gaussian priors (mean and covariance).

All routines operate on ``jnp.ndarray`` inputs and are differentiable, making
them suitable for gradient-based optimization or inference.
"""

from typing import Tuple, Union

import jax.numpy as jnp
from jax.scipy.linalg import solve

# Consistent array alias used across modules
Array = jnp.ndarray

def linear_chi2(x: Array, y: Array, err: Union[float, Array] = 0.0) -> Tuple[float, float, float, float, float]:
    """Weighted linear fit ``y ≈ a + b x`` and chi-squared.

    Performs a closed-form, weighted least-squares fit with independent
    Gaussian errors. If any entry of ``err`` is 0 (default), that point is
    treated as unweighted with weight 1.0.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Predictor values.
    y : jnp.ndarray, shape (n,)
        Observed responses.
    err : float or jnp.ndarray, shape (n,), optional
        Per-point standard deviation. A scalar broadcasts to all points. Zeros
        are replaced by 1.0 in the weights.

    Returns
    -------
    b : float
        Best-fit slope.
    be : float
        Standard error on the slope.
    a : float
        Best-fit intercept.
    ae : float
        Standard error on the intercept.
    chi2 : float
        Chi-squared of the fit evaluated at the best-fit parameters.

    Notes
    -----
    The solution minimizes ``sum_i w_i (y_i - a - b x_i)^2`` with
    ``w_i = 1/err_i^2`` when ``err_i > 0`` else ``w_i = 1``.
    """
    wt = jnp.where(err > 0, 1.0 / (err ** 2), 1.0)
    sumw = jnp.sum(wt)
    sumx = jnp.sum(wt * x)
    sumy = jnp.sum(wt * y)
    sumxx = jnp.sum(wt * x * x)
    sumxy = jnp.sum(wt * x * y)
    det = sumw * sumxx - sumx * sumx

    a = (sumxx * sumy - sumx * sumxy) / det
    b = (sumw * sumxy - sumx * sumy) / det

    residual = y - a - b * x
    chi2 = jnp.sum(wt * residual ** 2)
    ae = jnp.sqrt(sumxx / det)
    be = jnp.sqrt(sumw / det)
    return b, be, a, ae, chi2


def nll_ulens(
    flux: Array,
    M: Array,
    sigma2_obs: Array,
    sigma2_fs: float,
    sigma2_fb: float,
) -> float:
    """Negative log-likelihood with diagonal noise and diagonal priors.

    Assumes a linear flux model ``flux ≈ M @ [fs, fb]`` with independent
    observational errors ``C = diag(sigma2_obs)`` and independent Gaussian
    priors on ``fs`` and ``fb`` with variances ``sigma2_fs`` and ``sigma2_fb``.
    All integrals over ``fs, fb`` are done analytically, yielding the standard
    marginalized Gaussian NLL.

    Parameters
    ----------
    flux : jnp.ndarray, shape (n,)
        Observed fluxes.
    M : jnp.ndarray, shape (n, 2)
        Design matrix with columns ``[m_fs, 1]`` where ``m_fs`` is the source
        model term and the constant column captures blend flux.
    sigma2_obs : jnp.ndarray, shape (n,)
        Observational variances (diagonal of the noise covariance).
    sigma2_fs : float
        Prior variance for source flux ``fs``.
    sigma2_fb : float
        Prior variance for blend flux ``fb``.

    Returns
    -------
    nll : float
        Negative log-likelihood value.

    Notes
    -----
    The expression equals 0.5 * (mahalanobis + log|C| + log|Lambda| + log|A|)
    where ``A = M^T C^{-1} M + Lambda^{-1}`` and ``Lambda`` is diagonal with
    entries ``[sigma2_fs, sigma2_fb]``.
    """
    # Prior precision matrix Lambda^{-1} = diag(1/sigma2_fs, 1/sigma2_fb)
    lambda_fs = 1.0 / sigma2_fs
    lambda_fb = 1.0 / sigma2_fb

    # Compute C^{-1} @ M, using element-wise division since C is diagonal
    Cinv_M = M / sigma2_obs[:, None]  # shape (n, 2)

    # Compute matrix A = M^T @ C^{-1} @ M + Lambda^{-1}
    # A is 2x2 symmetric matrix
    a11 = lambda_fs + jnp.dot(M[:, 0], Cinv_M[:, 0])
    a22 = lambda_fb + jnp.dot(M[:, 1], Cinv_M[:, 1])
    a12 = jnp.dot(M[:, 0], Cinv_M[:, 1])  # same as a21

    # Compute vector b = M^T @ C^{-1} @ flux
    Cinv_flux = flux / sigma2_obs
    b1 = jnp.dot(M[:, 0], Cinv_flux)
    b2 = jnp.dot(M[:, 1], Cinv_flux)

    # Solve A @ mu = b for posterior mean mu = (mu_fs, mu_fb)
    detA = a11 * a22 - a12 * a12
    invA00 =  a22 / detA
    invA11 =  a11 / detA
    invA01 = -a12 / detA
    mu_fs = invA00 * b1 + invA01 * b2
    mu_fb = invA01 * b1 + invA11 * b2

    # Compute Mahalanobis part: flux^T @ C^{-1} @ flux - b^T @ A^{-1} @ b
    # Compute log-determinant terms
    mahalanobis = jnp.dot(flux, Cinv_flux) - (mu_fs * b1 + mu_fb * b2)
    logdet_C = jnp.sum(jnp.log(sigma2_obs))                     # log|C|
    logdet_Lambda = jnp.log(sigma2_fs) + jnp.log(sigma2_fb)     # log|Lambda|
    logdet_A = jnp.log(detA)                                    # log|A|

    # Final NLL value
    nll = 0.5 * (mahalanobis + logdet_C + logdet_Lambda + logdet_A)
    return nll
    
def nll_ulens_general(
    flux: Array,
    M: Array,
    C: Array,
    mu: Array,
    Lambda: Array,
) -> float:
    """Negative log-likelihood with full-rank noise and Gaussian priors.

    Generalizes :func:`nll_ulens` to non-diagonal observational covariance
    ``C`` and an arbitrary Gaussian prior ``N(mu, Lambda)`` over ``[fs, fb]``.
    The result is the marginalized NLL after integrating out the linear
    parameters analytically.

    Parameters
    ----------
    flux : jnp.ndarray, shape (n,)
        Observed fluxes.
    M : jnp.ndarray, shape (n, 2)
        Design matrix mapping ``[fs, fb]`` to the model flux.
    C : jnp.ndarray, shape (n, n)
        Observational covariance matrix (symmetric positive definite).
    mu : jnp.ndarray, shape (2,)
        Prior mean for ``[fs, fb]``.
    Lambda : jnp.ndarray, shape (2, 2)
        Prior covariance for ``[fs, fb]`` (symmetric positive definite).

    Returns
    -------
    nll : float
        Negative log-likelihood value.

    Notes
    -----
    Defines ``A = Lambda^{-1} + M^T C^{-1} M`` and posterior mean
    ``m = A^{-1} M^T C^{-1} (flux - M mu)``. The NLL equals
    ``0.5 * ( (flux - M mu)^T C^{-1} (flux - M mu) - m^T A m + log|C| + log|Lambda| + log|A| )``.
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
