import jax.numpy as jnp
from jax.scipy.special import gamma
from jax.numpy.fft import rfft, irfft

def log_extrap(x, N_extrap_low, N_extrap_high):
    low_x = high_x = []
    if(N_extrap_low):
        dlnx_low = jnp.log(x[1]/x[0])
        low_x = x[0] * jnp.exp(dlnx_low * jnp.arange(-N_extrap_low, 0) )
    if(N_extrap_high):
        dlnx_high= jnp.log(x[-1]/x[-2])
        high_x = x[-1] * jnp.exp(dlnx_high * jnp.arange(1, N_extrap_high+1) )
    x_extrap = jnp.hstack((low_x, x, high_x))
    return x_extrap

def c_window(n,n_cut):
    n_right = n[-1] - n_cut
    n_r=n[ n[:]  > n_right ] 
    theta_right=(n[-1]-n_r)/float(n[-1]-n_right-1) 
    W=jnp.ones(n.size)
    W=n.at[n[:] > n_right].set(theta_right - 1/(2*jnp.pi)*jnp.sin(2*jnp.pi*theta_right))
    return W

def g_m_vals(mu, q):
    '''
    This function calculates values of the gamma function used in the FFTLog algorithm for transforming functions with power-law behavior. 
    g_m_vals function adapted from FAST-PT
    '''

    if mu + 1 + q.real[0] == 0:
        raise ValueError("gamma(0) encountered. Please change another nu value! Try nu=1.1.")
    imag_q = jnp.imag(q)
    g_m = jnp.zeros(q.size, dtype=complex)
    cut = 200
    asym_q = q[jnp.abs(imag_q) + jnp.abs(mu) > cut]
    asym_plus = (mu + 1 + asym_q) / 2.
    asym_minus = (mu + 1 - asym_q) / 2.
    q_good = q[(jnp.abs(imag_q) + jnp.abs(mu) <= cut) & (q != mu + 1 + 0.0j)]
    alpha_plus = (mu + 1 + q_good) / 2.
    alpha_minus = (mu + 1 - q_good) / 2.
    g_m = g_m.at[(jnp.abs(imag_q) + jnp.abs(mu) <= cut) & (q != mu + 1 + 0.0j)].set(gamma(alpha_plus) / gamma(alpha_minus))
    # Asymptotic form
    g_m = g_m.at[jnp.abs(imag_q) + jnp.abs(mu) > cut].set(
        jnp.exp((asym_plus - 0.5) * jnp.log(asym_plus) - (asym_minus - 0.5) * jnp.log(asym_minus) - asym_q
                + 1. / 12 * (1. / asym_plus - 1. / asym_minus)
                + 1. / 360. * (1. / asym_minus**3 - 1. / asym_plus**3)
                + 1. / 1260 * (1. / asym_plus**5 - 1. / asym_minus**5)))
    g_m = g_m.at[q == mu + 1 + 0.0j].set(0. + 0.0j)
    return g_m

def g_m_ratio(a):
    '''g_m_ratio(a) = gamma(a) / gamma(a + 0.5)'''
    if a.real[0] == 0:
        raise ValueError("gamma(0) encountered. Please change another nu value! Try nu=1.1.")
    imag_a = jnp.imag(a)
    g_m = jnp.zeros(a.size, dtype=complex)
    cut = 100
    asym_a = a[jnp.abs(imag_a) > cut]
    asym_a_plus = asym_a + 0.5
    a_good = a[jnp.abs(imag_a) <= cut]
    g_m = g_m.at[jnp.abs(imag_a) <= cut].set(gamma(a_good) / gamma(a_good + 0.5))
    # Asymptotic form
    g_m = g_m.at[jnp.abs(imag_a) > cut].set(
        jnp.exp((asym_a - 0.5) * jnp.log(asym_a) - asym_a * jnp.log(asym_a_plus) + 0.5
                + 1. / 12 * (1. / asym_a - 1. / asym_a_plus)
                + 1. / 360. * (1. / asym_a_plus**3 - 1. / asym_a**3)
                + 1. / 1260 * (1. / asym_a**5 - 1. / asym_a_plus**5)))
    return g_m

def g_l(l, z_array):
    '''gl function for spherical Bessel transforms'''
    gl = 2.**z_array * g_m_vals(l + 0.5, z_array - 1.5)
    return gl


