"""Configure a magnification solver once, then evaluate physical parameters."""

from functools import partial

import jax.numpy as jnp

from . import lightcurve
from .config import DEFAULT_BINARY_CONFIG, DEFAULT_TRIPLE_CONFIG


def _bind_solver(backend, config, return_info, cpu, gpu, default_config):
    if backend == "cpu":
        if config is not None:
            raise ValueError("config applies only to backend='gpu'")
        if return_info:
            return cpu

        def magnification(*args, **kwargs):
            return cpu(*args, **kwargs).magnification

        return magnification
    if backend != "gpu":
        raise ValueError("backend must be 'cpu' or 'gpu'")
    if return_info:
        raise ValueError("return_info is available only for backend='cpu'")
    return partial(gpu, config=default_config if config is None else config)


def make_mag_binary(*, backend="gpu", config=None, return_info=False):
    """Create a binary-lens magnification function for source positions.

    Choose ``backend="cpu"`` for the CPU integrator or ``"gpu"`` for the
    GPU implementation (the default). ``config`` accepts a
    :class:`~microjax.BinaryMagConfig` for the GPU implementation.

    The returned function has the signature ``mag(w_points, rho, *, s, q,
    u1=0.0)``. It accepts only physical parameters and supports ``jax.jit``
    and ``jax.jacfwd``. With the CPU backend, ``return_info=True`` returns a
    :class:`~microjax.CpuMagnificationResult` instead of the magnification array.
    """
    solve = _bind_solver(
        backend,
        config,
        return_info,
        lightcurve._mag_binary_cpu_impl,
        lightcurve._mag_binary_single_pass_impl,
        DEFAULT_BINARY_CONFIG,
    )

    def mag_binary(w_points, rho, *, s, q, u1=0.0):
        """Calculate binary magnification at complex source positions.

        ``w_points`` is a one-dimensional complex array, ``rho`` the source
        radius, ``s`` the binary separation and ``q = m2/m1`` the mass ratio.
        ``u1`` is the linear limb-darkening coefficient; zero gives a uniform
        source. The result has one magnification per source position.

        ``microjax.mag_binary`` uses the default GPU configuration. To select
        the CPU integrator or other settings, use :func:`make_mag_binary`.
        For event parameters and observation times, use
        :func:`make_binary_lightcurve`.
        """
        return solve(w_points, rho, s=s, q=q, u1=u1)

    return mag_binary


def make_mag_triple(*, backend="gpu", config=None, return_info=False):
    """Create a triple-lens magnification function for source positions.

    Settings follow :func:`make_mag_binary`, with a
    :class:`~microjax.TripleMagConfig` for the GPU implementation.
    The returned signature is ``mag(w_points, rho, *, s, q, q3, r3, psi,
    u1=0.0)``; see :func:`mag_triple` for the lens convention.
    """
    solve = _bind_solver(
        backend,
        config,
        return_info,
        lightcurve._mag_triple_cpu_impl,
        lightcurve._mag_triple_single_pass_impl,
        DEFAULT_TRIPLE_CONFIG,
    )

    def mag_triple(w_points, rho, *, s, q, q3, r3, psi, u1=0.0):
        """Calculate triple magnification at complex source positions.

        Arguments follow :func:`mag_binary`, with ``q3 = m3/m1`` and the
        third lens's distance ``r3`` and angle ``psi`` from the midpoint of
        lenses 1 and 2. Angles are in radians; source positions are relative
        to the centre of mass of lenses 1 and 2.

        ``microjax.mag_triple`` uses the default GPU configuration. Use
        :func:`make_mag_triple` to select the CPU integrator or other settings,
        or :func:`make_triple_lightcurve` to work with event parameters.
        """
        return solve(w_points, rho, s=s, q=q, q3=q3, r3=r3, psi=psi, u1=u1)

    return mag_triple


def _trajectory(times, t0, tE, u0, alpha):
    tau = (jnp.asarray(times) - t0) / tE
    return (tau + 1j * u0) * jnp.exp(1j * alpha)


def make_binary_lightcurve(*, backend="gpu", config=None, return_info=False):
    """Create ``lightcurve(parameters, times, *, u1=0.0)`` for a static binary.

    ``parameters`` is a seven-element array in the order
    ``[t0, tE, u0, q, s, alpha, rho]``. ``times`` is a one-dimensional array
    of observation times. The result has one magnification per time.
    ``t0`` and ``tE`` use the same unit as ``times``; ``alpha`` is in radians.
    ``u1`` is the linear limb-darkening coefficient; zero gives a uniform source.
    The trajectory is ``((times-t0)/tE + 1j*u0) * exp(1j*alpha)`` in the
    binary centre-of-mass frame.

    Backend settings follow :func:`make_mag_binary`. For example::

        lightcurve = microjax.make_binary_lightcurve(backend="cpu")
        A = lightcurve(parameters, times, u1=0.5)
        derivatives = jax.jacfwd(lightcurve, argnums=0)(parameters, times, u1=0.5)
    """
    mag = make_mag_binary(backend=backend, config=config, return_info=return_info)

    def lightcurve(parameters, times, *, u1=0.0):
        t0, tE, u0, q, s, alpha, rho = parameters
        return mag(_trajectory(times, t0, tE, u0, alpha), rho, s=s, q=q, u1=u1)

    return lightcurve


def make_triple_lightcurve(*, backend="gpu", config=None, return_info=False):
    """Create ``lightcurve(parameters, times, *, u1=0.0)`` for a static triple.

    ``parameters`` is a ten-element array in the order
    ``[t0, tE, u0, q, s, alpha, rho, q3, r3, psi]``, with the last three
    entries defined in :func:`mag_triple`. The source trajectory
    remains relative to the centre of mass of lenses 1 and 2. Backend settings
    follow :func:`make_mag_triple`.
    """
    mag = make_mag_triple(backend=backend, config=config, return_info=return_info)

    def lightcurve(parameters, times, *, u1=0.0):
        t0, tE, u0, q, s, alpha, rho, q3, r3, psi = parameters
        return mag(
            _trajectory(times, t0, tE, u0, alpha),
            rho,
            s=s,
            q=q,
            q3=q3,
            r3=r3,
            psi=psi,
            u1=u1,
        )

    return lightcurve


mag_binary = make_mag_binary()
mag_triple = make_mag_triple()
