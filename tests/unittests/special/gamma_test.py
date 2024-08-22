from microjax.fastlens.special import gamma
from microjax.fastlens.special import digamma_
from scipy.special import gamma as gamma_s
from scipy.special import digamma as digamma_s
import numpy as np
import jax.numpy as jnp
from jax import jit

from jax import config

config.update("jax_enable_x64", True)


def test_gamma():
    """
    This function tests the gamma function by comparing the results of the gamma function from microjax and scipy.
    The function generates random complex numbers and compares the results of the gamma function from microjax and scipy.
    The function passes if the results are close within a tolerance of 1e-14.
    """
    random_complex = np.random.uniform(
        low=-100, high=100, size=1000
    ) + 1.0j * np.random.uniform(low=-100, high=100, size=1000)
    random_complex_jax = jnp.array(random_complex)
    gamma_jax = gamma(random_complex_jax)
    gamma_scipy = gamma_s(random_complex)
    atol = 1e-14

    assert jnp.allclose(gamma_jax, gamma_scipy, atol=atol)


def test_digamma():
    """
    This function tests the digamma function by comparing the results of the digamma function from microjax and scipy.
    The function generates random complex numbers and compares the results of the digamma function from microjax and scipy.
    The function passes if the results are close within a tolerance of 1e-14.
    """
    random_complex = np.random.uniform(
        low=-100, high=100, size=1000
    ) + 1.0j * np.random.uniform(low=-100, high=100, size=1000)
    random_complex_jax = jnp.array(random_complex)
    digamma = jit(digamma_)
    digamma_jax = digamma(random_complex_jax)
    digamma_scipy = digamma_s(random_complex)
    atol = 1e-14

    assert jnp.allclose(digamma_jax, digamma_scipy, atol=atol)


if __name__ == "__main__":
    test_gamma()
    test_digamma()
    print("All tests passed!")
