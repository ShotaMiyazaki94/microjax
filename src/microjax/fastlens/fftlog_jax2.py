
import jax.numpy as jnp
from microjax.fastlens.special import gamma
from jax.numpy.fft import rfft, irfft
from jax import jit, vmap, lax
from functools import partial

### Utility functions ####################
#@partial(jit, static_argnums=(1,2))
def log_extrap(x, N_extrap_low, N_extrap_high):
    if x.size < 2:
        raise ValueError("x must have at least 2 elements")

    dlnx_low = jnp.log(x[1] / x[0])
    low_x = x[0] * jnp.exp(dlnx_low * jnp.arange(-N_extrap_low, 0))
    low_x = lax.cond(N_extrap_low > 0, lambda: low_x, lambda: jnp.zeros_like(low_x))

    dlnx_high = jnp.log(x[-1] / x[-2])
    high_x = x[-1] * jnp.exp(dlnx_high * jnp.arange(1, N_extrap_high + 1))
    high_x = lax.cond(N_extrap_high > 0, lambda: high_x, lambda: jnp.zeros_like(high_x))

    x_extrap = jnp.concatenate((low_x, x, high_x))
    return x_extrap

#@partial(jit, static_argnums=(1,))
def c_window(n, n_cut):
    n_right = n[-1] - n_cut
    n_r = n[n > n_right]
    theta_right = (n[-1] - n_r) / (n[-1] - n_right - 1).astype(float)
    W = jnp.ones(n.size)
    W = jnp.where(n > n_right, theta_right - 1 / (2 * jnp.pi) * jnp.sin(2 * jnp.pi * theta_right), W)
    return W