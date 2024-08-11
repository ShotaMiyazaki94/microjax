from scipy.special import ellipe as ellipe_s
from scipy.special import ellipk as ellipk_s
from microjax.fastlens.special import ellipe
from microjax.fastlens.special import ellipk
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from jax import config

config.update("jax_enable_x64", True)


def test_ellipe():
    """This function tests the ellipe function by comparing the results of the ellipe function from microjax and scipy."""
    random_m_values = np.random.uniform(low=-1, high=1, size=1000)
    random_m_values_jax = jnp.array(random_m_values)
    ellipe_vmap = jit(vmap(ellipe))
    ellipe_jax = ellipe_vmap(random_m_values_jax)
    ellipe_scipy = ellipe_s(random_m_values_jax)
    atol = 1e-14

    assert jnp.allclose(ellipe_jax, ellipe_scipy, atol=atol)


def test_ellipk():
    """This function tests the ellipk function by comparing the results of the ellipk function from microjax and scipy."""
    random_m_values = np.random.uniform(low=-1, high=1, size=1000)
    random_m_values_jax = jnp.array(random_m_values)
    ellipk_vmap = jit(vmap(ellipk))
    ellipk_jax = ellipk_vmap(random_m_values_jax)
    ellipk_scipy = ellipk_s(random_m_values_jax)
    atol = 1e-14

    assert jnp.allclose(ellipk_jax, ellipk_scipy, atol=atol)


if __name__ == "__main__":
    test_ellipe()
    test_ellipk()
    print("All tests passed!")
