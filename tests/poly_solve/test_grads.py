import jax.numpy as jnp
from jax import grad, jit, vmap, random
import jax
from jax import config
config.update("jax_enable_x64", True)  # 高精度の計算を有効にする
from microjax.poly_solver import poly_roots_EA  # 仮定のライブラリと関数

def numerical_derivative(f, coeffs, h=1e-6):
    n = coeffs.size
    perturbations = jnp.eye(n) * h
    derivatives = []
    for i in range(n):
        coeffs_plus = coeffs.at[i].add(h)
        coeffs_minus = coeffs.at[i].add(-h)
        f_plus = f(coeffs_plus)
        f_minus = f(coeffs_minus)
        derivative = (f_plus - f_minus) / (2 * h)
        derivatives.append(derivative)
    return jnp.array(derivatives)

def test_polynomial(coeffs):
    roots = poly_roots_EA(coeffs)
    return jnp.sum(roots)  # 全ての根を返す

degree = 10  # number of polynomials 
length = 1   # number of equations

def generate_coeffs(key, num_coeffs, num_eqs, dtype=jnp.complex128):
    real_part = random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    imag_part = random.uniform(key, (num_eqs, num_coeffs), dtype=jnp.float64, minval=-1, maxval=1)
    return jnp.array(real_part + 1j * imag_part, dtype=dtype)

key = random.PRNGKey(0)
coeffs = generate_coeffs(key, degree, length)
coeffs = coeffs.ravel()
#coeffs = jnp.array([1, 0.1 + 0.3j, 0.8+0.2j], dtype=complex)

auto_stats = vmap(grad(test_polynomial, holomorphic=True))(coeffs.reshape(1, -1))
num_stats = vmap(numerical_derivative, in_axes=(None, 0))(test_polynomial, coeffs.reshape(1, -1))

# 出力フォーマットを改善
print("auto:",auto_stats)
print("num: ",num_stats)
