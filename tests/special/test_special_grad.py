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
    return jnp.sum(jit(vmap(ellipk))(x))

def ellipe_sum(x):
    return jnp.sum(jit(vmap(ellipe))(x))

def j0_sum(x):
    return jnp.sum(j0(x))

def j1_sum(x):
    return jnp.sum(j1(x))

def gamma_sum(x):
    return jnp.sum(gamma(x))

test = jnp.array(np.random.uniform(0, 0.99, 100000))

rtol = 1e-4
print("ellipk :",check_grads(ellipk_sum, (test,), order=1, rtol=rtol))
print("ellipe :",check_grads(ellipe_sum, (test,), order=1, rtol=rtol))
print("j0     :",check_grads(j0_sum, (test,),     order=1, rtol=rtol))
print("j1     :",check_grads(j1_sum, (test,),     order=1, rtol=rtol))

test = jnp.array(np.random.uniform(-100, 100, 100000) + 1.0j * np.random.uniform(-100, 100, 100000)) 
#test = jnp.linspace(0.5,1.0,1000)
print("gamma  :",check_grads(gamma_sum, (test,), order=1, rtol=rtol))
