import jax
from jax import test_util, grad
import numpy as np
import jax.numpy as jnp
from functools import partial
from microjax.poly_solver import poly_roots_EA_multi, poly_roots_EA
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_debug_nans", True)


def poly_sum(x):
    return jnp.mean(jnp.abs(poly_roots_EA(x)))

deg=11
test = jnp.array(np.random.uniform(-1,1,deg) + 1j * np.random.uniform(-1,1,deg))


rtol = 1e-1
print("poly_sum :",test_util.check_grads(poly_sum, (test,), order=1, rtol=rtol))