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
        _, nll, _, _ = example.flux_terms(magnification, flux, error)
        numpyro.factor("flux", -nll)

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


@pytest.mark.parametrize("amplitude", [1.0, 0.001, 0.0])
def test_gaussian_flux_marginal_matches_full_covariance(amplitude):
    # Include constant magnification: the proper Gaussian prior keeps it finite.
    a = 1 + amplitude * jnp.array([0.0, 0.3, 1.0, 3.0, 5.0, 1.5])
    error = jnp.array([0.01, 0.012, 0.018, 0.03, 0.04, 0.02])
    flux = 1.1 * a + 0.3 + error * jnp.array([0.4, -0.8, 0.3, 0.1, -0.5, 0.6])
    variance = example.FLUX_PRIOR_SCALE**2

    def reference(scale):
        design = jnp.column_stack((scale * a, jnp.ones_like(a)))
        covariance = jnp.diag(error**2) + variance * design @ design.T
        return -dist.MultivariateNormal(jnp.zeros_like(flux), covariance_matrix=covariance).log_prob(flux)

    def actual(scale):
        return example.flux_terms(scale * a, flux, error)[1]

    np.testing.assert_allclose(actual(1.0), reference(1.0), rtol=0, atol=2e-7)
    np.testing.assert_allclose(jax.jacfwd(actual)(1.0), jax.jacfwd(reference)(1.0), rtol=2e-6, atol=2e-7)

    # The curve draws must use the same prior-conditioned mean and covariance.
    design = jnp.column_stack((a, jnp.ones_like(a)))
    marginal_covariance = jnp.diag(error**2) + variance * design @ design.T
    expected_mean = variance * design.T @ jnp.linalg.solve(marginal_covariance, flux)
    expected_covariance = variance * jnp.eye(2) - variance**2 * design.T @ jnp.linalg.solve(marginal_covariance, design)
    _, _, mean, covariance = example.flux_terms(a, flux, error)
    np.testing.assert_allclose(mean, expected_mean, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(covariance, expected_covariance, rtol=2e-6, atol=2e-7)
