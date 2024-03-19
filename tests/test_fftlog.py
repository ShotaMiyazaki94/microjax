import numpy as np
import jax.numpy as jnp
from microjax.fastlens.fftlog import fftlog as fftlog_np, hankel as hankel_np, log_extrap as log_extrap_np, c_window as c_window_np, g_m_vals as g_m_vals_np, g_m_ratio as g_m_ratio_np, g_l as g_l_np, g_l_1 as g_l_1_np, g_l_2 as g_l_2_np, h_l as h_l_np
from microjax.fastlens.fftlog_jax import fftlog as fftlog_jax, hankel as hankel_jax, log_extrap as log_extrap_jax, c_window as c_window_jax, g_m_vals as g_m_vals_jax, g_m_ratio as g_m_ratio_jax, g_l as g_l_jax, g_l_1 as g_l_1_jax, g_l_2 as g_l_2_jax, h_l as h_l_jax
import jax
jax.config.update("jax_enable_x64", True)

# Set test input values
x = np.logspace(-2, 2, 100)
fx = np.sin(x)
nu = 1.1
ell = 2
N_extrap_low = 10
N_extrap_high = 10
c_window_width = 0.25
N_pad = 10
n = 1

# Convert numpy arrays to jax arrays
x_jax = jnp.array(x)
fx_jax = jnp.array(fx)

# Test fftlog
fftlog_np_obj = fftlog_np(x, fx, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)
fftlog_jax_obj = fftlog_jax(x_jax, fx_jax, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)

# Test hankel
hankel_np_obj = hankel_np(x, fx, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)
hankel_jax_obj = hankel_jax(x_jax, fx_jax, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)

# Test log_extrap
log_extrap_np_result = log_extrap_np(x, N_extrap_low, N_extrap_high)
log_extrap_jax_result = log_extrap_jax(x_jax, N_extrap_low, N_extrap_high)

# Test c_window
m = np.arange(0, len(x)//2+1)
m_jax = jnp.arange(0, len(x_jax)//2+1)
c_window_np_result = c_window_np(m, int(c_window_width * len(x)//2))
c_window_jax_result = c_window_jax(m_jax, int(c_window_width * len(x_jax)//2))

# Test g_m_vals
mu = 1.1
q = np.random.uniform(-10, 10, 100) + 1.0j * np.random.uniform(-10,10,100)
q_jax = jnp.array(q)
g_m_vals_np_result = g_m_vals_np(mu, q)
g_m_vals_jax_result = g_m_vals_jax(mu, q_jax)

# Test g_m_ratio
a = np.random.uniform(-10, 10, 100) + 1.0j * np.random.uniform(-10,10,100)
a_jax = jnp.array(a)
g_m_ratio_np_result = g_m_ratio_np(a)
g_m_ratio_jax_result = g_m_ratio_jax(a_jax)

# Test g_l
z_array = mu + 1j * np.linspace(-10, 10, 100)
z_array_jax = mu + 1j * jnp.linspace(-10, 10, 100)
g_l_np_result = g_l_np(ell, z_array)
g_l_jax_result = g_l_jax(ell, z_array_jax)

# Test g_l_1
g_l_1_np_result = g_l_1_np(ell, z_array)
g_l_1_jax_result = g_l_1_jax(ell, z_array_jax)

# Test g_l_2
g_l_2_np_result = g_l_2_np(ell, z_array)
g_l_2_jax_result = g_l_2_jax(ell, z_array_jax)

# Test h_l
h_l_np_result = h_l_np(ell, z_array)
h_l_jax_result = h_l_jax(ell, z_array_jax)

# Print results
atol = 1e-12
print("log_extrap results are close:", np.allclose(log_extrap_np_result, log_extrap_jax_result, atol=atol))
print("c_window results are close:", np.allclose(c_window_np_result, c_window_jax_result, atol=atol))
print("g_m_vals results are close:", np.allclose(g_m_vals_np_result, g_m_vals_jax_result, atol=atol))
print("g_m_ratio results are close:", np.allclose(g_m_ratio_np_result, g_m_ratio_jax_result, atol=atol))
print("g_l results are close:", np.allclose(g_l_np_result, g_l_jax_result, atol=atol))
print("g_l_1 results are close:", np.allclose(g_l_1_np_result, g_l_1_jax_result, atol=atol))
print("g_l_2 results are close:", np.allclose(g_l_2_np_result, g_l_2_jax_result, atol=atol))
print("h_l results are close:", np.allclose(h_l_np_result, h_l_jax_result, atol=atol))
