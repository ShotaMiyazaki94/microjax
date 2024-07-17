import jax
import jax.numpy as jnp
import numpy as np
from microjax.fastlens.fftlog_jax import log_extrap as log_extrap_j, c_window as c_window_j
from microjax.fastlens.fftlog_jax import g_m_vals as g_m_vals_j, g_m_ratio as g_m_ratio_j
from microjax.fastlens.fftlog import log_extrap, c_window, g_m_vals, g_m_ratio
from jax import jit
jax.config.update("jax_enable_x64", True)

atol=1e-14

x   = np.logspace(-3,3,1000)
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

mu=2.0
y   = 10**np.random.uniform(-2,2,1000) + 1.0j * 10**np.random.uniform(-2,2,1000)
y_j = jnp.array(y, dtype=complex)  
result   = g_m_vals(mu,y)
result_j = g_m_vals_j(mu,y_j)
idx_inf1 = jnp.isinf(result)
idx_inf2 = jnp.isinf(result_j)
print("g_m_vals:", jnp.allclose(result[~idx_inf1], result_j[~idx_inf2],atol=atol), np.sum(idx_inf1))

result   = g_m_ratio(y)
result_j = g_m_ratio_j(y_j)
idx_inf1 = jnp.isinf(result)
idx_inf2 = jnp.isinf(result_j)
print("g_m_ratio:", jnp.allclose(result[~idx_inf1], result_j[~idx_inf2],atol=atol), np.sum(idx_inf1))
