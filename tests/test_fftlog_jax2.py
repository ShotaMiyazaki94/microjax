import jax
import jax.numpy as jnp
import numpy as np
from microjax.fastlens.fftlog_jax2 import log_extrap as log_extrap_j, c_window as c_window_j
from microjax.fastlens.fftlog import log_extrap, c_window
jax.config.update("jax_enable_x64", True)

atol=1e-14

x   = np.logspace(-5,5,1000)
x_j = jnp.array(x) 
N_extrap_low = 512
N_extrap_high = 512
result_np  = log_extrap(x, N_extrap_low, N_extrap_high)
result_jax = log_extrap_j(x_j, N_extrap_low, N_extrap_high)
print("log_extrap:", jnp.allclose(result_np, result_jax,atol=atol))

n_cut = 100
result = c_window(x,n_cut)
result_j = c_window(x,n_cut)
print("c_window:", jnp.allclose(result, result_j,atol=atol))

