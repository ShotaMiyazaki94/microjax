"""JAX implementation of the FFT-based finite-source point-lens (FSPL) algorithm.

This mirrors the reference CPU version from Sugiyama (2022, arXiv:2203.06637)
but stays fully differentiable and JIT-friendly. The main entry points are
``fspl_*`` callables exposing ``A(u, rho)`` for different source profiles.
"""

import jax.numpy as jnp
from microjax.fastlens.fftlog_jax import fftlog, hankel
from microjax.fastlens.special import gamma, j1, j2, j1p5
from microjax.fastlens.special import ellipk, ellipe
from jax import lax


def fspl_point(u):
    """Point-source microlensing magnification ``A_p(u)``."""
    return (u**2 + 2) / jnp.abs(u) / (u**2 + 4)**0.5

# FFT based magnification
class fspl:
    """Base class for FFT-based extended-source microlensing magnification."""

    def __init__(self, fft_logumin=-6, fft_logumax=3, N_fft=1024, normalize_sk=True, rho_switch=1e-4, u_switch=10):
        """Configure FFT grids and small/large-ρ switching thresholds.

        Parameters
        ----------
        fft_logumin : int, optional
            Minimum ``log10(u)`` for the FFT grid (default -6).
        fft_logumax : int, optional
            Maximum ``log10(u)`` for the FFT grid (default 3).
        N_fft : int, optional
            Number of FFT grid points (default 1024).
        normalize_sk : bool, optional
            Placeholder flag for compatibility with the CPU API (kept for
            parity; no-op in the JAX path).
        rho_switch : float, optional
            Below this source size the code switches to the small-ρ asymptotic
            solution (Eq. 13 in the paper). Default 1e-4.
        u_switch : float, optional
            Boundary (in units of ρ) between small-ρ kernel and point-source
            approximation. Default 10.
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
        """Precompute Hankel of point-source magnification (shared core)."""
        u = jnp.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        u2Au = ((u**2 + 2.0) / (u**2 + 4.0)**0.5 / u - 1) * u**2
        h = hankel(u, u2Au, nu=1.5)
        self.k, apk = h.hankel(0)
        self.apk = apk * 2 * jnp.pi

    def init_Aext0(self):
        """Precompute small-ρ kernel A_ext0 (Eq. 13 in paper)."""
        x = jnp.logspace(self.fft_logumin, self.fft_logumax, self.N_fft)
        dump = jnp.exp(-(x / 100)**2)
        fx = x * self.sk(x, 1) * dump
        h = hankel(x, fx, nu=1.5, N_pad=self.N_fft)
        u, aext0 = h.hankel(0)
        # store interpolant as a callable to keep jit-friendliness
        self.Aext0 = lambda x_: jnp.interp(x_, u, aext0)

    def sk(self, k, rho):
        """Fourier counterpart of the source profile ``s̃(k; ρ)`` (override in subclasses)."""
        raise NotImplementedError

    def A0(self, rho):
        """Closed-form central magnification ``A(u=0, ρ)`` (override in subclasses)."""
        raise NotImplementedError

    def _A_for_small_rho(self, u, rho):
        """Extended-source magnification in the small-ρ regime (uses precomputed A_ext0)."""
        u = jnp.atleast_1d(jnp.abs(u))
        # Assign approximated solution: Eq. (13)
        a = jnp.ones(u.shape)
        idx = u < self.u_switch * rho
        x = jnp.where(idx, u / rho, jnp.ones_like(u))
        val = jnp.where(idx, self.Aext0(x) / rho + 1, jnp.ones_like(u))
        # Assign approximated solution: point-source magnification
        a = jnp.where(idx, val, fspl_point(u))
        #a = jnp.where(~idx, fspl_point(u), a)
        return a

    def _A_for_large_rho(self, u, rho):
        """Extended-source magnification in the large-ρ regime (FFT-based)."""
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
        """Return extended-source magnification ``A(u, ρ)``."""
        u = jnp.atleast_1d(jnp.abs(u))
        small_rho = self._A_for_small_rho(u, rho)
        large_rho = self._A_for_large_rho(u, rho)
        return jnp.where(rho < self.rho_switch, small_rho, large_rho)

class fspl_disk(fspl):
    """Uniform disk source profile."""

    def sk(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        idx = x > 1e-8
        core = jnp.where(idx, 2 * j1(x) / x, jnp.ones_like(x))
        return core

    def A0(self, rho):
        return (rho**2 + 4)**0.5 / rho

class fspl_ld1(fspl):
    """Linear limb-darkening profile."""

    def __init__(self, a1=0.5, **kwargs):
        """Create a linear limb-darkening magnification calculator.

        Parameters
        ----------
        a1 : float, optional
            Linear LD coefficient in the ``I(r)=1 - a1*(1 - sqrt(1 - (r/R)^2))``
            convention (default 0.5). Matches VBBinaryLensing ``u1``.
        **kwargs :
            Forwarded to :class:`fspl`.
        """
        self.a1 = a1
        super().__init__(**kwargs)

    def sk(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        small = 1e-8
        # disk term: 2 J1(x)/x
        sk_disk = jnp.where(x > small, 2 * j1(x) / x, jnp.ones_like(x))
        # linear LD term (n=1) from Eq. (15): nu = 1 + 1/2 = 1.5
        nu = 1.5
        pref = (2.0 ** nu) * gamma(nu + 1)  # = 2^nu * Gamma(nu+1)
        sk1 = jnp.where(
            x > small,
            pref * j1p5(x) / (x ** nu),
            jnp.ones_like(x),
        )
        # linear combination (no extra normalization; source is already normed)
        sk = (1.0 - self.a1) * sk_disk + self.a1 * sk1
        return sk

    def A0(self, rho):
        A0_disk = (rho**2 + 4) ** 0.5 / rho
        A0_lin = (2 * (rho**2 + 2) * ellipe(-rho**2 / 4) - (rho**2 + 4) * ellipk(-rho**2 / 4)) / rho**3
        return (1.0 - self.a1) * A0_disk + self.a1 * A0_lin

class fspl_ld2(fspl):
    """Quadratic limb-darkening (n=2) via unified pipeline."""

    def sk(self, k, rho):
        k = jnp.atleast_1d(k)
        x = k * rho
        nu = 2
        a_base = jnp.ones(x.shape) * 1.0 / (2 + 2)
        small = 1e-6
        const = 2**nu * gamma(nu) * nu / 8.0  # limit of j2(x) ~ x^2/8
        a_small = jnp.where(x < small, const, 0.0)
        a_main = jnp.where(x >= small, 2**nu * gamma(nu) * j2(x) / x**nu * nu, 0.0)
        a = a_small + a_main
        return a / a[0]

    def A0(self, rho):
        return (2 + 2) * (rho * (rho**2 + 2) * (rho**2 + 4) ** 0.5 - 8 * jnp.arcsinh(rho / 2)) / 4 / rho**4

class fspl_log(fspl):
    """Placeholder for logarithmic limb-darkening (not yet implemented)."""

    def su(self, u, rho):
        x = u / rho
        ans = jnp.zeros(x.shape)
        idx = x < 1
        sq = jnp.sqrt(1 - x[idx] ** 2)
        ans = ans.at[idx].set(sq * jnp.log(sq))
        return x * jnp.log(x)

    def sk(self, k, rho):
        # TODO: implement if/when needed
        raise NotImplementedError

    def A0(self, rho):
        # TODO: implement if/when needed
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------
# Older CPU code exposes ``magnification_disk`` in ``fastlens.mag_fft``.  Some
# downstream projects (e.g., jacscanomaly) still import that name from the JAX
# module.  The fspl_disk class is API compatible (constructor signature and
# ``A(u, rho)`` method), so we expose a lightweight alias here to avoid breaking
# those consumers.
magnification_disk = fspl_disk
