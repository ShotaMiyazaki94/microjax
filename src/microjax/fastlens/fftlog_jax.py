import jax.numpy as jnp
from jax.numpy.fft import rfft, irfft
from jax import jit
#from jax.scipy.special import gamma
from microjax.fastlens.gamma_jax import gamma_jax as gamma

class fftlog(object):
    def __init__(self, x, fx, nu=1.1, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
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

    def get_c_m(self):
        """
        return m and c_m
        c_m: the smoothed FFT coefficients of "biased" input function f(x): f_b = f(x) / x^\nu
        number of x values should be even
        c_window_width: the fraction of c_m elements that are smoothed,
        e.g. c_window_width=0.25 means smoothing the last 1/4 of c_m elements using "c_window".
        """

        f_b=self.fx * self.x**(-self.nu)
        c_m=rfft(f_b)
        m=jnp.arange(0,self.N//2+1) 
        c_m = c_m*c_window(m, int(self.c_window_width*self.N//2.) )
        return m, c_m
    
    def fftlog(self, ell):
        """
        Calculate F(y) = \int_0^\infty dx / x * f(x) * j_\ell(xy),
        where j_\ell is the spherical Bessel func of order ell.
        array y is set as y[:] = (ell+1)/x[::-1]
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * g_l(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        print(self.N_extrap_high,self.N,self.N_extrap_low)
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]
    
    def fftlog_dj(self, ell):
        """
        Calculate F(y) = \int_0^\infty dx / x * f(x) * j'_\ell(xy),
        where j_\ell is the spherical Bessel func of order ell.
        array y is set as y[:] = (ell+1)/x[::-1]
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * g_l_1(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]

    def fftlog_ddj(self, ell):
        """
        Calculate F(y) = \int_0^\infty dx / x * f(x) * j''_\ell(xy),
        where j_\ell is the spherical Bessel func of order ell.
        array y is set as y[:] = (ell+1)/x[::-1]
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * g_l_2(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]

    def fftlog_jsqr(self, ell):
        """
        Calculate F(y) = \int_0^\infty dx / x * f(x) * (j_\ell(xy))^2,
        where j_\ell is the spherical Bessel func of order ell.
        array y is set as y[:] = (ell+1)/x[::-1]
        """
        z_ar = self.nu + 1j*self.eta_m
        y = (ell+1.) / self.x[::-1]
        h_m = self.c_m * (self.x[0]*y[0])**(-1j*self.eta_m) * h_l(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y**(-self.nu) * jnp.sqrt(jnp.pi)/4.
        print(self.N_extrap_high,self.N,self.N_extrap_low)
        return y[self.N_extrap_high:self.N-self.N_extrap_low], Fy[self.N_extrap_high:self.N-self.N_extrap_low]

class hankel(object):
    def __init__(self, x, fx, nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
        print('nu is required to be between (0.5-n) and 2.')
        self.myfftlog = fftlog(x, jnp.sqrt(x)*fx, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad)

    def hankel(self, n):
        y, Fy = self.myfftlog.fftlog(n-0.5)
        Fy *= jnp.sqrt(2*y/jnp.pi)
        return y, Fy


#utils
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
    """
    One-side window function of c_m,
    Adapted from Eq.(C1) in McEwen et al. (2016), arXiv:1603.04826
    """
    n_right = n[-1] - n_cut
    n_r=n[ n[:]  > n_right ] 
    theta_right=(n[-1]-n_r)/float(n[-1]-n_right-1) 
    W=jnp.ones(n.size)
    W=W.at[n[:] > n_right].set(theta_right - 1/(2*jnp.pi)*jnp.sin(2*jnp.pi*theta_right))
    return W

def g_m_vals(mu, q):
    '''
    This function calculates values of the gamma function used in the FFTLog algorithm 
    for transforming functions with power-law behavior. 
    g_m_vals function adapted from FAST-PT
    g_m_vals(mu,q) = gamma( (mu+1+q)/2 ) / gamma( (mu+1-q)/2 ) = gamma(alpha+)/gamma(alpha-)
    mu = (alpha+) + (alpha-) - 1
    q = (alpha+) - (alpha-)
    switching to asymptotic form when |Im(q)| + |mu| > cut = 200
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
    idx = (jnp.abs(imag_q) + jnp.abs(mu) <= cut) & (q != mu + 1 + 0.0j)

    g_m = g_m.at[idx].set(gamma(alpha_plus) / gamma(alpha_minus))

    # Asymptotic form
    idx = jnp.where(jnp.abs(imag_q) + jnp.abs(mu) > cut)
    exp_term1 = (asym_plus - 0.5) * jnp.log(asym_plus) - (asym_minus - 0.5) * jnp.log(asym_minus) - asym_q
    exp_term2 = 1. / 12 * (1. / asym_plus - 1. / asym_minus)
    exp_term3 = 1. / 360 * (1. / asym_minus ** 3 - 1. / asym_plus ** 3)
    exp_term4 = 1. / 1260 * (1. / asym_plus ** 5 - 1. / asym_minus ** 5)
    g_m = g_m.at[idx].set(jnp.exp(exp_term1 + exp_term2 + exp_term3 + exp_term4))
    g_m = g_m.at[q == mu + 1 + 0.0j].set(0. + 0.0j)
    return g_m

def g_m_ratio(a):
    '''
    g_m_ratio(a) = gamma(a)/gamma(a+0.5)
    switching to asymptotic form when |Im(a)| > cut = 100
    '''
    if a.real[0] == 0:
        print("gamma(0) encountered. Please change another nu value! Try nu=1.1.")
        exit()
    imag_a = jnp.imag(a)
    g_m = jnp.zeros(a.size, dtype=complex)
    cut = 100
    asym_a = a[jnp.abs(imag_a) > cut]
    asym_a_plus = asym_a + 0.5

    idx = (jnp.abs(imag_a) <= cut)
    a_good = a[idx]
    g_m = g_m.at[idx].set(gamma(a_good) / gamma(a_good + 0.5))

    # asymptotic form
    idx_asym = jnp.abs(imag_a) > cut
    term1 = (asym_a - 0.5) * jnp.log(asym_a)
    term2 = -asym_a * jnp.log(asym_a_plus)
    term3 = 0.5
    term4 = 1. / 12 * (1. / asym_a - 1. / asym_a_plus)
    term5 = 1. / 360. * (1. / asym_a_plus ** 3 - 1. / asym_a ** 3)
    term6 = 1. / 1260 * (1. / asym_a ** 5 - 1. / asym_a_plus ** 5)

    g_m = g_m.at[idx_asym].set(jnp.exp(term1 + term2 + term3 + term4 + term5 + term6))

    return g_m


def g_l(l, z_array):
    '''
    gl = 2^z_array * gamma((l+z_array)/2.) / gamma((3.+l-z_array)/2.)
    alpha+ = (l+z_array)/2.
    alpha- = (3.+l-z_array)/2.
    mu = (alpha+) + (alpha-) - 1 = l+0.5
    q = (alpha+) - (alpha-) = z_array - 1.5
    '''
    gl = 2.**z_array * g_m_vals(l + 0.5, z_array - 1.5)
    return gl

def g_l_1(l, z_array):
    '''
    for integral containing one first-derivative of spherical Bessel function
    gl1 = -2^(z_array-1) *(z_array -1)* gamma((l+z_array-1)/2.) / gamma((4.+l-z_array)/2.)
    mu = l+0.5
    q = z_array - 2.5
    '''
    gl1 = -2.**(z_array - 1) * (z_array - 1) * g_m_vals(l + 0.5, z_array - 2.5)
    return gl1

def g_l_2(l, z_array):
    '''
    for integral containing one 2nd-derivative of spherical Bessel function
    gl2 = 2^(z_array-2) *(z_array -1)*(z_array -2)* gamma((l+z_array-2)/2.) / gamma((5.+l-z_array)/2.)
    mu = l+0.5
    q = z_array - 3.5
    '''
    gl2 = 2.**(z_array - 2) * (z_array - 1) * (z_array - 2) * g_m_vals(l + 0.5, z_array - 3.5)
    return gl2

def h_l(l, z_array):
    '''
    hl = gamma(l+ z_array/2.) * gamma((2.-z_array)/2.) / gamma((3.-z_array)/2.) / gamma(2.+l -z_array/2.)
    first component is g_m_vals(2l+1, z_array - 2)
    second component is gamma((2.-z_array)/2.) / gamma((3.-z_array)/2.)
    '''
    hl = g_m_vals(2 * l + 1., z_array - 2.) * g_m_ratio((2. - z_array) / 2.)
    return hl
