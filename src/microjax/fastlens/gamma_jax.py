import jax.numpy as jnp
from jax import jit

@jit
def gamma_jax(z):
    """
    Custom gamma function with JAX similar to scipy.special.gamma.
    I see that the accuracy compared with scipy.special.gamma is met at 1e-14 within Re(z) < 3.
    I adopted the algorothm from https://ui.adsabs.harvard.edu/abs/2021arXiv210400697C/abstract.
    
    Arg   : z (complex)
    return: Gamma(z) (complex)
    """
    K = 5
    N = 16
    c = jnp.log(2 * jnp.pi) / 2
    a = jnp.array([1/12, -1/360, 1/1260, -1/1680, 1/1188])
    
    z = jnp.atleast_1d(z)
    original_shape = z.shape
    z = z.ravel()
    
    g = jnp.zeros_like(z, dtype=jnp.complex128)
    
    negative_real_mask = jnp.real(z) <= 0
    not_special_case = ~negative_real_mask

    def compute_case(z_val):
        t = z_val + N
        gam = a[0] / t
        for k in range(1, K):
            t *= (z_val + N) ** 2
            gam += a[k] / t
        u = jnp.prod(jnp.array([(z_val + n) for n in range(N)]), axis=0)
        lg = c - (z_val + N) + (z_val + N - 0.5) * jnp.log(z_val + N) - jnp.log(u) + gam
        return jnp.exp(lg)

    def compute_reflection(z_val):
        reflected_z = 1 - z_val
        sin_term = jnp.sin(jnp.pi * z_val)
        gamma_reflected = compute_case(reflected_z)
        return jnp.pi / (sin_term * gamma_reflected)

    # Compute results for all elements
    positive_results = compute_case(z)
    negative_results = compute_reflection(z)

    # Select the appropriate results based on the masks
    g = jnp.where(not_special_case, positive_results, g)
    g = jnp.where(negative_real_mask, negative_results, g)

    non_positive_integer_mask = negative_real_mask & (jnp.real(z) == jnp.floor(jnp.real(z)))
    g = jnp.where(non_positive_integer_mask, jnp.nan, g)
    
    return g.reshape(original_shape)
