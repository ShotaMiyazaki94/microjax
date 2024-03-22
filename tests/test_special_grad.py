from microjax.fastlens.special import ellipk, ellipe, gamma
#from jax.scipy.special import gamma
from microjax.fastlens.special import j0, j1, j2, j1p5
import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, vmap, grad
from jax.test_util import check_grads

def ellipk_sum(x):
    return jnp.mean(jit(vmap(ellipk))(x))

def ellipe_sum(x):
    return jnp.mean(jit(vmap(ellipe))(x))

def gamma_sum(x):
    return jnp.mean(gamma(x))

test = jnp.linspace(1e-4,1.0,1000)

check_grads(ellipk_sum, (test,), order=1, rtol=1e-5)
check_grads(ellipe_sum, (test,), order=1, rtol=1e-5)

test = jnp.linspace(0.5,1.0,1000)
check_grads(gamma_sum, (test,), order=1, rtol=1e-5)
