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
    flux = jnp.array([0.1, 0.3, -0.2])
    error = jnp.array([0.2, 0.3, 0.4])

    def forward_model(parameters, times):
        return jnp.sin(times[:, None] * jnp.arange(1, n + 1)) @ parameters

    def original_model():
        unit = numpyro.sample("unit", dist.Uniform(0.0, 1.0).expand((n,)).to_event(1))
        mean = forward_model(example.physical_parameters(unit), times)
        numpyro.sample("flux", dist.Normal(mean, error).to_event(1), obs=flux)

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
