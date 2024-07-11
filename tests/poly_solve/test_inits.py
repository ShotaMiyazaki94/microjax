import jax.numpy as jnp
from jax import grad, jit, vmap, random
import jax
from jax import config
config.update("jax_enable_x64", True)  # 高精度の計算を有効にする
from microjax.poly_solver import poly_roots_EA_multi_init  # 仮定のライブラリと関数

def generate_coeffs(key, num_coeffs, num_eqs, dtype=jnp.complex128):
    real_part = random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    imag_part = random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    return jnp.array(real_part + 1j * imag_part, dtype=dtype)

degree = 10  # number of polynomials 
length = 10   # number of equations 

key = random.PRNGKey(0)
coeffs = generate_coeffs(key, degree, length)
inits  = generate_coeffs(key, degree-1, length)
print(coeffs.shape)
print(inits.shape)

roots = poly_roots_EA_multi_init(coeffs, inits)

print(roots.shape)
