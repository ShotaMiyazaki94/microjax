"""
python module for calculating microlensing magnification with finite source size effect
by Sunao Sugiyama
Jun 13, 2023
"""

import numpy as np
from microjax.fastlens import fftlog
from scipy.special import gamma
from scipy.special import j0, j1, jn, ellipk, ellipe

def A_point(u):
    return (u**2+2)/np.abs(u)/(u**2+4)**0.5

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
        self.fft_logumin  = fft_logumin 
        self.fft_logumax  = fft_logumax
        self.N_fft        = N_fft
        # flag to normalize source profile in fourier space
        self.normalize_sk = normalize_sk

        # The scale for rho and u to switch to use the approximate solution.
        # When rho < rho_switch, we use the approximate solution.
        # For u < u_switch*rho, we use Eq. (13) which is precomputed by init_small_rho below.
        # For u > u_switch*rho, we use point-source magnification.
        self.rho_switch = rho_switch
        self.u_switch   = u_switch

        # initialization
        self.init_Apk()
        self.init_Aext0()
        
    def init_Apk(self):
        """
        Initializing the FFT counter part of point-source magnification
        """
        u = np.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        u2Au = ((u**2+2.0)/(u**2+4.0)**0.5/u - 1) * u**2
        h = fftlog.hankel(u, u2Au, nu=1.5, N_extrap_high=512, N_extrap_low=512)
        self.k, apk = h.hankel(0)
        self.apk = apk*2*np.pi

    def init_Aext0(self):
        """
        Implementation of A_ext0 in Eq. (13)
        """
        x    = np.logspace(-5, 5, 1024)
        dump = np.exp(-(x/100)**2)
        fx   = x * self.sk(x, 1) * dump
        h    = fftlog.hankel(x, fx, nu=1.5, N_pad=1024)
        u, aext0 = h.hankel(0)
        self.Aext0 = lambda x: np.interp(x, u, aext0)

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
        u = np.atleast_1d(np.abs(u))
        # Assign approximated solution: Eq. (13)
        a = np.ones(u.shape)
        idx = u < self.u_switch*rho
        x   = u[idx]/rho
        val = self.Aext0(x)/rho+1
        a[idx] = val
        # Assign approximated solution: point-source magnification
        idx = u >=self.u_switch*rho
        a[idx]= A_point(u[idx])
        return a

    def _A_for_large_rho(self, u, rho):
        """
        Evaluates the extended-source magnification for rho large enough

        Aargs:
            u   (array)
            rho (float)
        """
        u = np.atleast_1d(np.abs(u))
        # typical scale of source profile
        k_rho = 2*np.pi/rho
        # dumping factor to avoid noisy result
        dump = np.exp(-(self.k/k_rho/20)**2)
        # Fourier counter part of extended-source magnification
        cj = self.apk*self.k**2 * self.sk(self.k, rho) * dump
        # Hankel back the extended-source magnification
        h = fftlog.hankel(self.k, cj, nu=1.5, N_pad=512)
        u_fft, a_fft = h.hankel(0)
        a_fft = a_fft/2/np.pi
        a_fft = a_fft + 1
        
        # Truncate the result u>100 and append A(u=100)=1
        idx   = u_fft < 100
        u_fft = np.hstack([u_fft[idx], 100])
        a_fft = np.hstack([a_fft[idx], 1  ])

        # Assign values
        a     = np.ones(u.shape) * self.A0(rho)
        idx   = u>0
        a[idx]= np.interp(np.log10(u[idx]), np.log10(u_fft), a_fft, right=1)

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
        u = np.atleast_1d(np.abs(u))
        inputs = (u, rho)

        if rho < self.rho_switch:
            return self._A_for_small_rho(u, rho)
        else:
            return self._A_for_large_rho(u, rho)

        
class magnification_disk(magnification):
    def sk(self, k, rho):
        """
        Implementation of Eq. (14)
        """
        k = np.atleast_1d(k)
        x   = k*rho
        a   = np.ones(x.shape)
        idx = x>0
        a[idx] = 2*j1(x[idx])/x[idx]
        return a

    def A0(self, rho):
        """
        Returns A_disk(0, rho)
        """
        return (rho**2+4)**0.5/rho

class magnification_limb(magnification):
    def __init__(self, n_limb, **kwargs):
        """
        This is for limb darkening profile, Eq. (6).
        
        Args:
            n_limb (int): order of limb darkening
        """
        self.n_limb = n_limb
        super().__init__(**kwargs)

    def sk(self, k, rho):
        """
        Implementation of Eq. (15)
        """
        k = np.atleast_1d(k)
        x   = k*rho
        nu  = 1+self.n_limb/2
        a   = np.ones(x.shape)*1.0/(self.n_limb+2)
        idx = x>0
        a[idx] = 2**nu*gamma(nu)*jn(nu, x[idx])/x[idx]**nu * nu
        return a

    def A0(self, rho):
        """
        Returns A_limb(0, rho)
        """
        if self.n_limb == 1:
            return (2+1) * (2*(rho**2+2)*ellipe(-rho**2/4)-(rho**2+4)*ellipk(-rho**2/4)) / 3.0/rho**3
        elif self.n_limb == 2:
            return (2+2) * (rho*(2+rho**2)*(4+rho**2)**0.5 - 8*np.arcsinh(rho/2)) / 4/rho**4
        elif self.n_limb >= 3:
            raise NotImplementedError

class magnification_log(magnification):
    """
    This is for logarithmic profile from https://doi.org/10.1086/110960.     
    """
    def su(self, u, rho):
        x   = u/rho
        ans = np.zeros(x.shape)
        # assign log profile
        idx = x<1
        sq = np.sqrt(1-x[idx]**2)
        ans[idx] = sq*np.log(sq)
        return x*np.log(x)
    
    def sk(self, k, rho):
        """
        Implementation of Fourier counterpart
        """
        pass

    def A0(self, rho):
        """
        Returns A_log(0, rho)
        """
        pass