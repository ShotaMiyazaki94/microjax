from microjax.fastlens.special import j0, j1, j2, j1p5
from scipy.special import jv as jv_s
from scipy.special import j0 as j0_s
from scipy.special import j1 as j1_s
import numpy as np
import jax.numpy as jnp
from jax import jit

from jax import config

config.update("jax_enable_x64", True)

def get_random_real():
    random_real = np.random.uniform(low=-1000, high=1000, size=1000)  
    random_real_jax = jnp.array(random_real)
    return random_real_jax

def test_j0():
    random_real_jax = get_random_real()
    j0_jax   = j0(random_real_jax)
    j0_scipy = j0_s(random_real_jax) 
    atol = 1e-14

    assert jnp.allclose(j0_jax, j0_scipy, atol=atol)
    
    
def test_j1():
    random_real_jax = get_random_real()    
    j1_jax   = j1(random_real_jax)
    j1_scipy = j1_s(random_real_jax) 
    atol = 1e-14

    assert jnp.allclose(j1_jax, j1_scipy, atol=atol)
    
def test_j2():
    random_real_jax = get_random_real()    
    j2_jax   = j2(random_real_jax)
    j2_scipy = jv_s(2.0,random_real_jax) 
    atol = 1e-14
    
    assert jnp.allclose(j2_jax, j2_scipy, atol=atol)

def test_j1p5():    
    random_real = np.random.uniform(low=0, high=1000, size=1000)  
    random_real_jax = jnp.array(random_real)
    j1p5_jax   = j1p5(random_real_jax)
    j1p5_scipy = jv_s(1.5,random_real_jax) 
    atol = 1e-14

    assert jnp.allclose(j1p5_jax, j1p5_scipy, atol=atol)



if __name__ == "__main__":
    test_j0()
    test_j1()
    test_j2()
    test_j1p5()
    print("All tests passed!")
