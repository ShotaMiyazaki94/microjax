
"""JAX native FFTLog / Hankel utilities used by the fastlens magnification code.

The implementation mirrors the CPU reference in ``fastlens/_legacy`` but keeps
the code JIT-friendly and differentiable. It provides a minimal FFTLog class
for logarithmically spaced inputs and a thin ``hankel`` wrapper that evaluates
zero-th order Hankel transforms (spherical Bessel ``j_ell`` integrals) that
appear in the FFT-based microlensing algorithm.
"""

import jax.numpy as jnp
from microjax.fastlens.special import gamma
from jax.numpy.fft import rfft, irfft
from jax import jit
from functools import partial

class fftlog(object):
    """FFTLog on a log-spaced grid with optional edge extrapolation and padding."""

    def __init__(self, x, fx, nu=1.1, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
        """Prepare FFT coefficients for a biased input function ``f(x)/x^nu``.

        Parameters
        ----------
        x : array_like
            Monotonic, logarithmically spaced sample positions.
        fx : array_like
            Function values ``f(x)`` on the same grid.
        nu : float, optional
            Bias exponent applied before the FFT (default 1.1).
        N_extrap_low, N_extrap_high : int, optional
            Number of points to extrapolate on each side in log space to reduce
            ringing near the boundaries.
        c_window_width : float, optional
            Fraction of the upper half of coefficients to cosine-taper.
        N_pad : int, optional
            Zero-padding length (added symmetrically) to improve FFT stability.
        """
        self.x_origin = x # x is logarithmically spaced
        self.dlnx = jnp.log(x[1]/x[0])
        self.fx_origin= fx # f(x) array
        self.nu = nu
        self.N_extrap_low = N_extrap_low
        self.N_extrap_high = N_extrap_high
        self.c_window_width = c_window_width

        # extrapolate x and f(x) linearly in log(x), and log(f(x))
        self.x = log_extrap(x, N_extrap_low, N_extrap_high)
        self.fx = log_extrap(fx, N_extrap_low, N_extrap_high)
        self.N = self.x.size

        # zero-padding
        self.N_pad = N_pad
        if(N_pad):
            pad = jnp.zeros(N_pad)
            self.x = log_extrap(self.x, N_pad, N_pad)
            self.fx = jnp.hstack((pad, self.fx, pad))
            self.N += 2*N_pad
            self.N_extrap_high += N_pad
            self.N_extrap_low += N_pad
        
        if(self.N%2==1): # Make sure the array sizes are even
            self.x= self.x[:-1]
            self.fx=self.fx[:-1]
            self.N -= 1
            if(N_pad):
                self.N_extrap_high -=1 

        self.m, self.c_m = self.get_c_m()
        self.eta_m = 2*jnp.pi/(float(self.N)*self.dlnx) * self.m     
    #@jit
    def get_c_m(self):
        """Compute smoothed FFT coefficients of the biased function ``f/x^nu``."""
        f_b=self.fx * self.x**(-self.nu)
        c_m=rfft(f_b)
        m=jnp.arange(0,self.N//2+1) 
        c_m = c_m*c_window(m, int(self.c_window_width*self.N//2.) )
        return m, c_m
    #@jit 
    def fftlog(self, ell):
        r"""
        Calculate ``F(y) = ∫ dx/x f(x) j_ell(xy)`` on the reflected grid ``y``.

        Parameters
        ----------
        ell : int or float
            Order of the spherical Bessel ``j_ell``.

        Returns
        -------
        y : jnp.ndarray
            Output sample locations ordered like ``x[::-1]`` without padding.
        Fy : jnp.ndarray
            Transform values on ``y``.
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * g_l(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        #print(self.N_extrap_high,self.N,self.N_extrap_low)
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]

    def fftlog_dj(self, ell):
        r"""
        Same as :py:meth:`fftlog` but for the first derivative ``j'_ell``.
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * g_l_1(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]
    
    def fftlog_ddj(self, ell):
        r"""
        Same as :py:meth:`fftlog` but for the second derivative ``j''_ell``.
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * g_l_2(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]
    
    def fftlog_jsqr(self, ell):
        r"""
        Calculate ``F(y) = ∫ dx/x f(x) [j_ell(xy)]^2`` on the reflected grid.
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * h_l(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        #print(self.N_extrap_high,self.N,self.N_extrap_low)
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]

class hankel(object):
    """Zero-th order Hankel transform wrapper built on :class:`fftlog`."""

    def __init__(self, x, fx, nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
        #print('nu is required to be between (0.5-n) and 2.')
        #print("HANKEL!!!",x)
        self.myfftlog = fftlog(x, jnp.sqrt(x)*fx, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)
    
    def hankel(self, n):
        """Evaluate ``∫ x fx J_{n}(xy) dx`` via the FFTLog core."""
        y, Fy = self.myfftlog.fftlog(n-0.5)
        Fy *= jnp.sqrt(2*y/jnp.pi)
        return y, Fy

### Utility functions ####################
@partial(jit, static_argnums=(1,2))
def log_extrap(x, N_extrap_low, N_extrap_high):
    """Extrapolate a log-spaced grid ``x`` by a fixed number of steps on each side."""
    if N_extrap_low > 0:
        dlnx_low = jnp.log(x[1] / x[0])
        low_x = x[0] * jnp.exp(dlnx_low * jnp.arange(-N_extrap_low, 0))
    else:
        low_x = jnp.array([])

    if N_extrap_high > 0:
        dlnx_high = jnp.log(x[-1] / x[-2])
        high_x = x[-1] * jnp.exp(dlnx_high * jnp.arange(1, N_extrap_high + 1))
    else:
        high_x = jnp.array([])

    x_extrap = jnp.concatenate((low_x, x, high_x))
    return x_extrap    
@partial(jit, static_argnums=(1,))
def c_window(n, n_cut):
    """Cosine taper for the highest ``n_cut`` frequency bins to suppress ringing."""
    n_right = n[-1] - n_cut
    idx = n > n_right
    theta_right = (n[-1] - n) / (n[-1] - n_right - 1).astype(float)
    W = jnp.where(idx, theta_right - 1 / (2 * jnp.pi) * jnp.sin(2 * jnp.pi * theta_right), jnp.ones_like(n))
    return W

#@partial(jit, static_argnums=(0,))
@jit
def g_m_vals(mu, q):
    """Gamma ratio ``Gamma((mu+1+q)/2) / Gamma((mu+1-q)/2)`` with asymptotics."""
    #if (mu + 1 + q.real[0] == 0):
    #    raise ValueError("gamma(0) encountered. Please change another nu value! Try nu=1.1.")

    imag_q = jnp.imag(q)
    g_m = jnp.zeros_like(q, dtype=complex)
    # custom gamma_ becomes unstable for large |Im|; switch to asymptotic earlier
    cut = 80
    mask_asym = jnp.abs(imag_q) + jnp.abs(mu) > cut
    mask_good = (jnp.abs(imag_q) + jnp.abs(mu) <= cut) & (q != mu + 1 + 0j)

    alpha_plus  = (mu + 1 + q) / 2.
    alpha_minus = (mu + 1 - q) / 2.
    g_m_good    = gamma(alpha_plus) / gamma(alpha_minus)

    asym_q = q
    asym_plus = (mu + 1 + asym_q) / 2.
    asym_minus = (mu + 1 - asym_q) / 2.
    g_m_asym = jnp.exp((asym_plus - 0.5) * jnp.log(asym_plus) - (asym_minus - 0.5) * jnp.log(asym_minus) - asym_q
                       + 1. / 12 * (1. / asym_plus - 1. / asym_minus) + 1. / 360 * (1. / asym_minus ** 3 - 1. / asym_plus ** 3)
                       + 1. / 1260 * (1. / asym_plus ** 5 - 1. / asym_minus ** 5))

    g_m = jnp.where(mask_good, g_m_good, g_m)
    g_m = jnp.where(mask_asym, g_m_asym, g_m)
    g_m = jnp.where(q == mu + 1 + 0j, jnp.zeros_like(g_m), g_m)

    return g_m

@jit
def g_m_ratio(a):
    """Gamma ratio ``Gamma(a) / Gamma(a+0.5)`` with asymptotic fallback."""
    #if (a.real[0] == 0):
    #    raise ValueError("gamma(0) encountered. Please change another nu value! Try nu=1.1.")

    imag_a = jnp.imag(a)
    g_m = jnp.zeros_like(a, dtype=complex)
    cut = 80
    mask_asym = jnp.abs(imag_a) > cut
    mask_good = jnp.abs(imag_a) <= cut

    asym_a = a
    asym_a_plus = asym_a + 0.5
    g_m_asym = jnp.exp((asym_a - 0.5) * jnp.log(asym_a) - asym_a * jnp.log(asym_a_plus) + 0.5
                       + 1. / 12 * (1. / asym_a - 1. / asym_a_plus) + 1. / 360 * (1. / asym_a_plus ** 3 - 1. / asym_a ** 3)
                       + 1. / 1260 * (1. / asym_a ** 5 - 1. / asym_a_plus ** 5))

    a_good = a
    g_m_good = gamma(a_good) / gamma(a_good + 0.5)

    g_m = jnp.where(mask_good, g_m_good, g_m)
    g_m = jnp.where(mask_asym, g_m_asym, g_m)

    return g_m

def g_l(l,z_array):
    """Helper for ``fftlog``: gamma ratio for spherical Bessel ``j_l`` integrals."""
    gl = 2.**z_array * g_m_vals(l+0.5,z_array-1.5)
    return gl

def g_l_1(l,z_array):
    """Gamma prefactor for integrals involving ``j_l'``."""
    gl1 = -2.**(z_array-1) *(z_array -1) * g_m_vals(l+0.5,z_array-2.5)
    return gl1

def g_l_2(l,z_array):
    """Gamma prefactor for integrals involving ``j_l''``."""
    gl2 = 2.**(z_array-2) *(z_array -1)*(z_array -2)* g_m_vals(l+0.5,z_array-3.5)
    return gl2

def h_l(l,z_array):
    """Prefactor for integrals of ``(j_l)^2`` used in ``fftlog_jsqr``."""
    hl = g_m_vals(2*l+1., z_array - 2.) * g_m_ratio((2.-z_array)/2.)
    return hl
