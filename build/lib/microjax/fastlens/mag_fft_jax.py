import jax
import jax.numpy as jnp
from microjax.fastlens.fftlog_jax import fftlog, hankel
from microjax.fastlens.special import gamma, j0, j1, j2, j1p5
from microjax.fastlens.special import ellipk, ellipe, ellipk_, ellipe_
from jax import jit, lax, vmap

def A_point(u):
    return (u**2 + 2) / jnp.abs(u) / (u**2 + 4)**0.5

class mag_:
    def __init__(self, fft_logumin=-6, fft_logumax=3, N_fft=1024, normalize_sk=True, rho_switch=1e-4, u_switch=10):
        """
        Args:
            fft_logumin      (int): minimum of FFT bin in u space (default=-6)
            fft_logumax      (int): maximum of FFT bin in u space (default=3)
            N_fft            (int): number of FFT bin in u space  (default=1024)
            normalize_sk    (bool): flag to normalize source profile in fourier space
        """
        self.fft_logumin = fft_logumin
        self.fft_logumax = fft_logumax
        self.N_fft = N_fft
        self.init_Apk()

    def init_Apk(self):
        """
        Initializing the FFT counter part of point-source magnification
        """
        u = jnp.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        u2Au = ((u**2 + 2.0) / (u**2 + 4.0)**0.5 / u - 1) * u**2
        h = hankel(u, u2Au, nu=1.5)
        #h = hankel(u, u2Au, nu=1.5, N_extrap_high=512, N_extrap_low=512)
        self.k, apk = h.hankel(0)
        self.apk = apk * 2 * jnp.pi
    
    def sk(self, k, rho, a1):
        # Implement Fourier counter part of source profile.
        pass

    def A0(self, rho, a1):
        # Implement A(0, rho).
        pass

    def A(self, u, rho, a1=0.0):
        """
        Evaluates the extended-source magnification for rho large enough
        Aargs:
            u   (array)
            rho (float)
            a1  (float): limb darkening coefficient
        """
        u = jnp.atleast_1d(jnp.abs(u))
        a_base = jnp.ones(u.shape) * self.A0(rho, a1)
        # typical scale of source profile
        k_rho = 2 * jnp.pi / rho
        # dumping factor to avoid noisy result
        dump = jnp.exp(-(self.k / k_rho / 20)**2)
        # Fourier counter part of extended-source magnification
        cj = self.apk * self.k**2 * self.sk(self.k, rho, a1) * dump
        # Hankel back the extended-source magnification
        h = hankel(self.k, cj, nu=1.5, N_pad=512)
        u_fft, a_fft = h.hankel(0)
        a_fft = a_fft / 2.0 / jnp.pi
        a_fft = a_fft + 1.0
        # Truncate the result u>100 and append A(u=100)=1
        max_u_fft = 100
        u_fft_truncated = jnp.where(u_fft < max_u_fft, u_fft, max_u_fft)
        a_fft_truncated = jnp.where(u_fft < max_u_fft, a_fft, 1)
        log_u = jnp.log10(u)
        log_u_fft_truncated = jnp.log10(u_fft_truncated)
        interp_values = jnp.interp(log_u, log_u_fft_truncated, a_fft_truncated, right=1)
        a = jnp.where(u > 0, interp_values, a_base)

        return a 

class mag_disk(mag_):
    def sk(self, k, rho, a1):
        k = jnp.atleast_1d(k)
        x = k * rho
        idx = x > 0
        a = jnp.where(idx, 2 * j1(x) / x, jnp.ones_like(x))
        return a

    def A0(self, rho, a1):
        return (rho**2 + 4)**0.5 / rho

class mag_limb(mag_):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def su(self, u, rho, a1):
        su = jnp.zeros(u.size)
        u_rho = jnp.where(u < rho, u / rho, 1.0)
        #norm = (1.0 - a1) * jnp.pi * rho**2
        su = jnp.where(u < rho, 1.0 - a1 * (1.0 - jnp.sqrt(1.0 - u_rho**2)), su)
        #su = su / norm
        return su

    def sk(self, k, rho, a1):
        u = jnp.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        u2su = u**2 * self.su(u, rho, a1)
        h = hankel(u, u2su, nu=1.5)
        k, sk = h.hankel(0)
        sk = sk/sk[0]
        return sk

    def A0(self, rho, a1):
        A0_disk  = (rho**2 + 4)**0.5 / rho 
        A0_term1 = (2 + 1) * (2 * (rho**2 + 2) * ellipe(-rho**2 / 4) - (rho**2 + 4) * ellipk(-rho**2 / 4)) / 3.0 / rho**3 
        norm = 1.0 - a1
        return (A0_disk - a1 * A0_term1) / norm 

# FFT based magnification
class magnification:
    def __init__(self, fft_logumin=-6, fft_logumax=3, N_fft=1024, normalize_sk=True, rho_switch=1e-4, u_switch=10):
        """
        Args:
            fft_logumin      (int): minimum of FFT bin in u space (default=-6)
            fft_logumax      (int): maximum of FFT bin in u space (default=3)
            N_fft            (int): number of FFT bin in u space  (default=1024)
            normalize_sk    (bool): flag to normalize source profile in fourier space
        """
        # Defining FFT bin
        # Default choice is validated for rho > 1e-4 to ensure 0.3% precision
        self.fft_logumin = fft_logumin
        self.fft_logumax = fft_logumax
        self.N_fft = N_fft
        self.normalize_sk = normalize_sk

        # The scale for rho and u to switch to use the approximate solution.
        # When rho < rho_switch, we use the approximate solution.
        # For u < u_switch*rho, we use Eq. (13) which is precomputed by init_small_rho below.
        # For u > u_switch*rho, we use point-source magnification.
        self.rho_switch = rho_switch
        self.u_switch = u_switch

        # initialization
        self.init_Apk()
        self.init_Aext0()
        
    def init_Apk(self):
        """
        Initializing the FFT counter part of point-source magnification
        """
        u = jnp.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        u2Au = ((u**2 + 2.0) / (u**2 + 4.0)**0.5 / u - 1) * u**2
        h = hankel(u, u2Au, nu=1.5)
        #h = hankel(u, u2Au, nu=1.5, N_extrap_high=512, N_extrap_low=512)
        self.k, apk = h.hankel(0)
        self.apk = apk * 2 * jnp.pi

    def init_Aext0(self):
        """
        Implementation of A_ext0 in Eq. (13)
        """
        x = jnp.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        #x = jnp.logspace(-5, 5, 1024)
        dump = jnp.exp(-(x / 100)**2)
        #min_value = 1e-12
        #dump = jnp.where(dump < min_value, min_value, dump)
        fx = x * self.sk(x, 1) * dump
        #print("init_Aext0:", x, fx, self.sk(x, 1), dump)
        h = hankel(x, fx, nu=1.5, N_pad=self.N_fft)
        #h = hankel(x, fx, nu=1.5, N_pad=1024,)
        u, aext0 = h.hankel(0)
        self.Aext0 = lambda x: jnp.interp(x, u, aext0)

    def sk(self, k, rho):
        # Implement Fourier counter part of source profile.
        pass

    def A0(self, rho):
        # Implement A(0, rho).
        pass

    def _A_for_small_rho(self, u, rho):
        """
        Evaluates the extended-source magnification for small rho.
        Aargs:
            u   (array)
            rho (float)
        """
        u = jnp.atleast_1d(jnp.abs(u))
        # Assign approximated solution: Eq. (13)
        a = jnp.ones(u.shape)
        idx = u < self.u_switch * rho
        x = jnp.where(idx, u / rho, jnp.ones_like(u))
        val = jnp.where(idx, self.Aext0(x) / rho + 1, jnp.ones_like(u))
        # Assign approximated solution: point-source magnification
        a = jnp.where(idx, val, A_point(u))
        #a = jnp.where(~idx, A_point(u), a)
        return a

    def _A_for_large_rho(self, u, rho):
        """
        Evaluates the extended-source magnification for rho large enough
        Aargs:
            u   (array)
            rho (float)
        """
        u = jnp.atleast_1d(jnp.abs(u))
        a_base = jnp.ones(u.shape) * self.A0(rho)
        # typical scale of source profile
        k_rho = 2 * jnp.pi / rho
        # dumping factor to avoid noisy result
        dump = jnp.exp(-(self.k / k_rho / 20)**2)
        # Fourier counter part of extended-source magnification
        cj = self.apk * self.k**2 * self.sk(self.k, rho) * dump
        # Hankel back the extended-source magnification
        h = hankel(self.k, cj, nu=1.5, N_pad=512)
        u_fft, a_fft = h.hankel(0)
        a_fft = a_fft / 2 / jnp.pi
        a_fft = a_fft + 1
        # Truncate the result u>100 and append A(u=100)=1
        max_u_fft = 100
        u_fft_truncated = jnp.where(u_fft < max_u_fft, u_fft, max_u_fft)
        a_fft_truncated = jnp.where(u_fft < max_u_fft, a_fft, 1)
        log_u = jnp.log10(u)
        log_u_fft_truncated = jnp.log10(u_fft_truncated)
        interp_values = jnp.interp(log_u, log_u_fft_truncated, a_fft_truncated, right=1)
        a = jnp.where(u > 0, interp_values, a_base)

        return a
    
    def A(self, u, rho):
        """
        Returns the extended-source magnification of microlensing light-curve.
        Args:
            u   (float): impact parameter in the lens plane normalized by Einstein angle.
            rho (float): source scale parameter.
        Returns:
            a   (float): magnification for extended-source profile.
        """
        u = jnp.atleast_1d(jnp.abs(u))
        """
        def small(_):
            return self._A_for_small_rho(u, rho)
        def large(_):
           return self._A_for_large_rho(u, rho) 
        return lax.cond(rho < self.rho_switch, small, large, None) 
        """
        small_rho = self._A_for_small_rho(u, rho) 
        large_rho = self._A_for_large_rho(u, rho) 
        return jnp.where(rho < self.rho_switch, small_rho, large_rho) 

class magnification_disk(magnification):
    def sk(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        idx = x > 0
        a = jnp.where(idx, 2 * j1(x) / x, jnp.ones_like(x))
        return a

    def A0(self, rho):
        return (rho**2 + 4)**0.5 / rho

class mag_limb1(magnification):
    def __init__(self, a1=0.5, **kwargs):
        self.a1 = a1 #limb-darkening coeff
        super().__init__(**kwargs)

    def su(self, u, rho, a1=None):
        su = jnp.zeros(u.size)
        su = jnp.where(u < rho, 1.0 - self.a1 * (1.0 - jnp.sqrt(jnp.abs(1.0 - u**2 / rho**2))), su)
        su = su/su[0] 
        return su

    def sk(self, k, rho):
        u = jnp.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        u2su = u**2 * self.su(u, rho)
        h = hankel(u, u2su, nu=1.5)
        #h = hankel(u, u2su, nu=1.5)
        k, sk = h.hankel(0)
        sk = sk/sk[0]
        return sk

    def sk__(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        sk_disk  = 2.0 * j1(x) / x
        sk_term1 = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
        norm = 1.0 - self.a1
        sk = jnp.where(x > 0, (sk_disk - self.a1 * sk_term1) / norm, jnp.ones_like(x))
        return sk 


    def sk_(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        nu = 1.5
        a_base = jnp.ones(x.shape) * 1.0 / (1 + 2)
        norm = 1.0 - self.a1
        sk_disk  = 2 * j1(x) / x 
        sk_term1 = nu * 2**nu * gamma(nu) * j1p5(x) / x**nu 
        sk = jnp.where(x > 0, (sk_disk - self.a1 * sk_term1) / norm, a_base)
        #sk = sk / sk[0]
        return sk

    def A0(self, rho):
        A0_disk  = (rho**2 + 4)**0.5 / rho 
        A0_term1 = (2 + 1) * (2 * (rho**2 + 2) * ellipe(-rho**2 / 4) - (rho**2 + 4) * ellipk(-rho**2 / 4)) / 3.0 / rho**3 
        norm = 1.0 - self.a1
        return (A0_disk - self.a1 * A0_term1) / norm 


class magnification_limb1(magnification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sk(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        nu = 1.5
        a_base = jnp.ones(x.shape)*1.0 / (1 + 2)
        sk = jnp.where(x > 0, nu * 2**nu * gamma(nu) * j1p5(x) / x**nu, a_base)
        return sk

    def A0(self, rho): 
        return (2 + 1) * (2 * (rho**2 + 2) * ellipe(-rho**2 / 4) - (rho**2 + 4) * ellipk(-rho**2 / 4)) / 3.0 / rho**3

class magnification_limb2(magnification):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sk(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        nu = 2
        a_base = jnp.ones(x.shape)*1.0 / (2 + 2)
        a = jnp.where(x > 0, 2**nu * gamma(nu) * j2(x) / x**nu * nu, a_base)
        return a

    def A0(self, rho):
        return (2 + 2) * (rho * (rho**2 + 2) * (rho**2 + 4)**0.5 - 8*jnp.arcsinh(rho/2)) / 4/rho**4

class magnification_log(magnification):
    def su(self, u, rho):
        x = u / rho
        ans = jnp.zeros(x.shape)
        idx = x < 1
        sq = jnp.sqrt(1 - x[idx]**2)
        ans = ans.at[idx].set(sq * jnp.log(sq))
        return x * jnp.log(x)

    def sk(self, k, rho):
        pass

    def A0(self, rho):
        pass
