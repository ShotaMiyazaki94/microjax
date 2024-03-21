from microjax.fastlens.special import ellipk, ellipe, gamma
from microjax.fastlens.special import j0, j1, j2, j1p5
from scipy.special import ellipe as ellipe_s, ellipk as ellipk_s, gamma as gamma_s
from scipy.special import jv as jv_s, j0 as j0_s, j1 as j1_s
import jax.numpy as jnp
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import jit, vmap

random_complex = np.random.uniform(low=-100, high=100, size=1000) + 1.0j * np.random.uniform(low=-100, high=100, size=1000) 

gamma_jax = gamma(random_complex)
gamma_scipy = gamma_s(random_complex)

random_m_values = np.random.uniform(low=-1, high=1, size=1000)
random_m_values_jax = jnp.array(random_m_values)

ellipk_vmap = jit(vmap(ellipk))
ellipe_vmap = jit(vmap(ellipe))

ellipk_jax   = ellipk_vmap(random_m_values_jax)
ellipk_scipy = ellipk_s(random_m_values_jax)
ellipe_jax   = ellipe_vmap(random_m_values_jax)
ellipe_scipy = ellipe_s(random_m_values_jax)

random_real = np.random.uniform(low=-1000, high=1000, size=1000)  
random_real_jax = jnp.array(random_real)
j0_jax   = j0(random_real_jax)
j0_scipy = j0_s(random_real_jax) 
j1_jax   = j1(random_real_jax)
j1_scipy = j1_s(random_real_jax) 
j2_jax   = j2(random_real_jax)
j2_scipy = jv_s(2.0,random_real_jax) 
random_real = np.random.uniform(low=0, high=1000, size=1000)  
random_real_jax = jnp.array(random_real)
j1p5_jax   = j1p5(random_real_jax)
j1p5_scipy = jv_s(1.5,random_real_jax) 


atol = 1e-14
print("Accuracy: %.1e"%(atol))
print("gamma results are close :", jnp.allclose(gamma_jax, gamma_scipy, atol=atol))
print("ellipk results are close:", jnp.allclose(ellipk_jax, ellipk_scipy, atol=atol))
print("ellipe results are close:", jnp.allclose(ellipe_jax, ellipe_scipy, atol=atol))
print("j0 results are close    :", jnp.allclose(j0_jax, j0_scipy, atol=atol))
print("j1 results are close    :", jnp.allclose(j1_jax, j1_scipy, atol=atol))
print("j2 results are close    :", jnp.allclose(j2_jax, j2_scipy, atol=atol))
print("j1p5 results are close  :", jnp.allclose(j1p5_jax, j1p5_scipy, atol=atol))
print("-------------------------------------")
import timeit
n_runs = 1000
random_real = np.random.uniform(low=-1000, high=1000, size=1000)  
random_real_jax = jnp.array(random_real)

time_gamma_jax = timeit.timeit(lambda: gamma(random_m_values_jax), number=n_runs)
time_gamma_scipy = timeit.timeit(lambda: gamma_s(random_m_values), number=n_runs)

time_ellipk_jax = timeit.timeit(lambda: ellipk_vmap(random_m_values_jax), number=n_runs)
time_ellipk_scipy = timeit.timeit(lambda: ellipk_s(random_m_values), number=n_runs)

time_ellipe_jax = timeit.timeit(lambda: ellipe_vmap(random_m_values_jax), number=n_runs)
time_ellipe_scipy = timeit.timeit(lambda: ellipe_s(random_m_values), number=n_runs)

time_j0_jax = timeit.timeit(lambda: j0(random_real_jax), number=n_runs)
time_j0_scipy = timeit.timeit(lambda: j0_s(random_real), number=n_runs)

time_j1_jax = timeit.timeit(lambda: j1(random_real_jax), number=n_runs)
time_j1_scipy = timeit.timeit(lambda: j1_s(random_real), number=n_runs)

time_j2_jax = timeit.timeit(lambda: j2(random_real_jax), number=n_runs)
time_j2_scipy = timeit.timeit(lambda: jv_s(2.0,random_real), number=n_runs)

random_real = np.random.uniform(low=0, high=1000, size=1000)  
random_real_jax = jnp.array(random_real)
time_j1p5_jax = timeit.timeit(lambda: j1p5(random_real_jax), number=n_runs)
time_j1p5_scipy = timeit.timeit(lambda: jv_s(1.5,random_real), number=n_runs)

print(f"gamma JAX time   : {time_gamma_jax / n_runs:.6f} s")
print(f"gamma SciPy time : {time_gamma_scipy / n_runs:.6f} s")
print(f"ellipk JAX time  : {time_ellipk_jax / n_runs:.6f} s")
print(f"ellipk SciPy time: {time_ellipk_scipy / n_runs:.6f} s")
print(f"ellipe JAX time  : {time_ellipe_jax / n_runs:.6f} s")
print(f"ellipe SciPy time: {time_ellipe_scipy / n_runs:.6f} s")
print(f"j0 JAX time    : {time_j0_jax / n_runs:.6f} s")
print(f"j0 SciPy time  : {time_j0_scipy / n_runs:.6f} s")
print(f"j1 JAX time    : {time_j1_jax / n_runs:.6f} s")
print(f"j1 SciPy time  : {time_j1_scipy / n_runs:.6f} s")
print(f"j2 JAX time    : {time_j2_jax / n_runs:.6f} s")
print(f"j2 SciPy time  : {time_j2_scipy / n_runs:.6f} s")
print(f"j1p5 JAX time  : {time_j1p5_scipy / n_runs:.6f} s")
print(f"j1p5 SciPy time: {time_j1p5_scipy / n_runs:.6f} s")
