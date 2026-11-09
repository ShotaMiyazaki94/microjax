"""Check the HMC coordinate change without evaluating a lens or running NUTS."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist
from numpyro.distributions.transforms import LowerCholeskyAffine
from numpyro.infer.util import potential_energy
from scipy.integrate import quad
from scipy.special import ndtr

spec = spec_from_file_location(
    "hmc_example", Path(__file__).resolve().parents[2] / "example" / "hmc" / "injection_recovery.py"
)
example = module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (3.0, 0.7)])
def test_whitening_preserves_original_density_and_gradient(scale, offset):
    n = len(example.NAMES)
    times = jnp.array([-0.8, 0.2, 1.1])
    flux = jnp.array([1.2, 1.7, 1.4])
    error = jnp.array([0.2, 0.3, 0.4])

    def forward_model(parameters, times):
        return 1.5 + 0.1 * times + 0.02 * parameters[0] * times**2 + 0.001 * parameters.sum()

    def original_model():
        unit = numpyro.sample("unit", dist.Uniform(0.0, 1.0).expand((n,)).to_event(1))
        magnification = forward_model(example.physical_parameters(unit), times)
        residual, correction, _, _ = example.flux_terms(magnification, flux, error)
        numpyro.factor("flux", -0.5 * jnp.sum(residual**2) - correction)

    # Unequal scales and correlations, with two unrelated choices of fit geometry.
    cholesky = scale * jnp.diag(jnp.geomspace(0.003, 0.5, n)) @ (jnp.eye(n) + 0.2 * jnp.tril(jnp.ones((n, n)), -1))
    to_logit = LowerCholeskyAffine(jnp.linspace(-0.8, 0.4, n) + offset, cholesky)
    kwargs = {"forward_model": forward_model, "to_logit": to_logit}

    def original_energy(z):
        return potential_energy(original_model, (), {}, {"unit": z})

    def whitened_energy(w):
        return potential_energy(example.model, (times, flux, error), kwargs, {"whitened": w})

    z = jnp.stack([jnp.linspace(-2.0, 1.0, n), jnp.linspace(0.5, -0.5, n)])
    w = to_logit.inv(z)
    np.testing.assert_allclose(to_logit(w), z, atol=1e-13)
    logdet = jnp.log(jnp.diag(cholesky)).sum()
    np.testing.assert_allclose(
        jax.vmap(whitened_energy)(w), jax.vmap(original_energy)(z) - logdet, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jacfwd(whitened_energy)(w[0]), cholesky.T @ jax.jacfwd(original_energy)(z[0]), rtol=1e-11, atol=1e-10
    )


@pytest.mark.parametrize("fs, fb, amplitude", [(1.0, 0.2, 1.0), (0.015, 4.98, 1.0), (1.0, 0.2, 0.001)])
def test_uniform_flux_marginal_matches_independent_integral(fs, fb, amplitude):
    a = 1 + amplitude * np.array([0.0, 0.3, 1.0, 3.0, 5.0, 1.5])
    error = np.array([0.1, 0.12, 0.18, 0.3, 0.4, 0.2])
    flux = fs * a + fb

    def reference(scale):
        design = np.column_stack((scale * a, np.ones(a.size)))
        covariance = np.linalg.inv(design.T @ (design / error[:, None] ** 2))
        mean = covariance @ design.T @ (flux / error**2)
        sigma_s = np.sqrt(covariance[0, 0])
        sigma_b_given_s = np.sqrt(covariance[1, 1] - covariance[0, 1] ** 2 / covariance[0, 0])
        bounds = example.FLUX_BOUNDS

        def integrand(x):
            mu_b = mean[1] + covariance[0, 1] / sigma_s * x
            lo, hi = (bounds[1] - mu_b) / sigma_b_given_s
            return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi) * (ndtr(hi) - ndtr(lo))

        left, right = np.clip((bounds[0] - mean[0]) / sigma_s, -38, 38)
        edges = (bounds[1] - mean[1]) * sigma_s / covariance[0, 1]
        points = np.unique(np.clip(np.r_[-12, -5, -2, 0, 2, 5, 12, edges], left, right))
        probability = quad(integrand, left, right, points=points, epsabs=1e-11, epsrel=1e-11)[0]
        residual = (design @ mean - flux) / error
        return (
            0.5 * residual @ residual
            - 0.5 * np.linalg.slogdet(covariance)[1]
            - np.log(probability)
            + np.log(np.diff(bounds, axis=1)).sum()
            + np.log(error).sum()
            + 0.5 * (a.size - 2) * np.log(2 * np.pi)
        )

    def actual(scale):
        residual, correction, _, _ = example.flux_terms(scale * jnp.asarray(a), jnp.asarray(flux), jnp.asarray(error))
        return 0.5 * jnp.sum(residual**2) + correction

    np.testing.assert_allclose(actual(1.0), reference(1.0), atol=2e-8)
    step = 1e-4
    finite_difference = (reference(1.0 + step) - reference(1.0 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jacfwd(actual)(1.0), finite_difference, rtol=2e-5, atol=2e-5)
