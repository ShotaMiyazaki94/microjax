import jax
import jax.numpy as jnp
from jax import jit, grad, lax, vmap
from functools import partial
from scipy.special import ellipk as ellipk_s, ellipe as ellipe_s
jax.config.update("jax_enable_x64", True)

def agm(a, b, num_iter=10):
    def agm_step(carry, _):
        a, b = carry
        return ((a + b) / 2, jnp.sqrt(a * b)), None    
    (a, b), _ = lax.scan(agm_step, (a, b), None, length=num_iter)
    return a

#@partial(jit, static_argnums=(1,))
def ellipk(m, num_iter=10):
    a, b = 1.0, jnp.sqrt(1 - m)
    c = agm(a, b, num_iter=num_iter)
    return jnp.pi / (2 * c)

def agm2(a0, b0, s_sum0, num_iter=10):
    def agm_step2(carry, _):
        a, b, s_sum, n = carry
        a_next, b_next = (a + b) / 2, jnp.sqrt(a * b)
        c_next = 0.5 * (a - b)
        s_sum_next = s_sum + 2**(n-1) * c_next**2
        n += 1
        return (a_next, b_next, s_sum_next, n), None
    n=1
    (a, b, s_sum, n), _ = lax.scan(agm_step2, (a0, b0, s_sum0, n), None, length=num_iter)
    return a, s_sum

#@partial(jit, static_argnums=(1,))
def ellipe(m, num_iter=10):
    a0, b0 = 1.0, jnp.sqrt(1 - m)
    c0     = jnp.sqrt(a0**2 - b0**2)
    s_sum0 = 0.5 * c0**2
    a, s_sum = agm2(a0, b0, s_sum0, num_iter=num_iter)
    return jnp.pi / (2 * a) * (1 - s_sum)

# vmapで並列化
ellipk_vmap = jit(vmap(ellipk, in_axes=(0,)))
ellipe_vmap = jit(vmap(ellipe, in_axes=(0,)))

# ベクトル入力の例
k_vec = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
K_vec = ellipk_vmap(k_vec)
print("第一種完全楕円積分 K:", K_vec)
K_vec = ellipk_s(k_vec)
print("第一種完全楕円積分 K:", K_vec)
E_vec = ellipe_vmap(k_vec)
print("第二種完全楕円積分 E:", E_vec)
E_vec = ellipe_s(k_vec)
print("第二種完全楕円積分 E:", E_vec)






# 第一種完全楕円積分の例
k = jnp.array([0.5])
ellipk_vmap = vmap(ellipk, in_axes=(0, None))
ellipe_vmap = vmap(ellipe, in_axes=(0, None))

K = ellipk_vmap(k)
print("第一種完全楕円積分 K(0.5):", K)
K = ellipk_s(k)
print("第一種完全楕円積分 K(0.5):", K)

# 第二種完全楕円積分の例
E = ellipe_vmap(k)
print("第二種完全楕円積分 E(0.5):", E)
E = ellipe_s(k)
print("第二種完全楕円積分 E(0.5):", E)

# 自動微分の例
#dK_dk = grad(ellipk)(k)
#print("dK/dk (k=0.5):", dK_dk)

#dE_dk = grad(ellipe)(k)
#print("dE/dk (k=0.5):", dE_dk)
