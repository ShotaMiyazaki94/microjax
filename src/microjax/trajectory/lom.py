"""Linear, circular, and elliptic 3D orbital-motion helpers.

This module provides JAX-friendly utilities for three orbital-motion
parameterizations used in binary-lens models:

- First-order (linear) model:

  - ``s(t) = s0 + ds_dt * (t - tref)``
  - ``alpha(t) = alpha0 + dalpha_dt * (t - tref)``

- Circular 3D model (VBBinaryLensing-compatible):

  - parameters ``w1 = (1/s) ds/dt``, ``w2 = d alpha / dt``, ``w3 = (1/s) ds_z/dt``
  - returns projected separation ``s(t)``, lens-axis angle ``alpha(t)``, and
    line-of-sight separation ``s_z(t)``.

- Elliptic 3D model (VBBinaryLensing Kepler-compatible):

  - parameters ``w1``, ``w2``, ``w3``, ``szs``, ``ar`` as in
    ``VBBinaryLensing::BinaryLightCurveKepler``.
  - solves Kepler's equation with fixed-iteration Newton updates (JAX-friendly).

where ``s`` is the projected binary separation and ``alpha`` is the lens-axis
position angle (radians).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from ._kepler import solve_kepler_newton

Array = jnp.ndarray

__all__ = [
    "linear_orbital_motion",
    "linear_orbital_motion_jit",
    "linear_orbital_motion_state",
    "linear_orbital_motion_state_jit",
    "circular_orbital_motion_3d",
    "circular_orbital_motion_3d_jit",
    "circular_orbital_motion_3d_state",
    "circular_orbital_motion_3d_state_jit",
    "elliptic_orbital_motion_3d",
    "elliptic_orbital_motion_3d_jit",
    "elliptic_orbital_motion_3d_state",
    "elliptic_orbital_motion_3d_state_jit",
    "to_rotating_lens_frame",
]


def linear_orbital_motion(
    t: float | Array,
    s0: float,
    alpha0: float,
    ds_dt: float = 0.0,
    dalpha_dt: float = 0.0,
    tref: float = 0.0,
) -> Tuple[Array, Array]:
    """Evaluate linear orbital-motion parameters at time ``t``.

    Parameters
    ----------
    t : float or Array
        Time(s) at which to evaluate the model.
    s0 : float
        Binary separation at reference time ``tref``.
    alpha0 : float
        Lens-axis angle at reference time ``tref`` (radians).
    ds_dt : float, optional
        Time derivative of the separation.
    dalpha_dt : float, optional
        Time derivative of the lens-axis angle (radians per unit time).
    tref : float, optional
        Reference time where ``s = s0`` and ``alpha = alpha0``.

    Returns
    -------
    s_t : Array
        Separation evaluated at ``t``.
    alpha_t : Array
        Lens-axis angle evaluated at ``t``.
    """
    t = jnp.asarray(t)
    dt = t - jnp.asarray(tref, dtype=t.dtype)
    s_t = jnp.asarray(s0, dtype=t.dtype) + jnp.asarray(ds_dt, dtype=t.dtype) * dt
    alpha_t = jnp.asarray(alpha0, dtype=t.dtype) + jnp.asarray(
        dalpha_dt, dtype=t.dtype
    ) * dt
    return s_t, alpha_t


linear_orbital_motion_jit = jax.jit(linear_orbital_motion)


def to_rotating_lens_frame(w: complex | Array, alpha: float | Array) -> Array:
    """Rotate source-plane coordinates into the instantaneous lens frame.

    Parameters
    ----------
    w : complex or Array
        Source-plane coordinates in a fixed sky frame.
    alpha : float or Array
        Lens-axis angle(s) in the same frame (radians).

    Returns
    -------
    Array
        Rotated coordinates ``w * exp(-1j * alpha)`` aligned with the lens axis.
    """
    w = jnp.asarray(w)
    alpha = jnp.asarray(alpha)
    return w * jnp.exp(-1j * alpha)


def linear_orbital_motion_state(
    t: float | Array,
    w: complex | Array,
    s0: float,
    alpha0: float,
    ds_dt: float = 0.0,
    dalpha_dt: float = 0.0,
    tref: float = 0.0,
) -> Tuple[Array, Array, Array]:
    """Evaluate linear orbital motion and rotate a trajectory to lens frame.

    Parameters
    ----------
    t : float or Array
        Time(s) corresponding to ``w``.
    w : complex or Array
        Source-plane coordinates in a fixed sky frame.
    s0 : float
        Binary separation at reference time ``tref``.
    alpha0 : float
        Lens-axis angle at reference time ``tref`` (radians).
    ds_dt : float, optional
        Time derivative of separation.
    dalpha_dt : float, optional
        Time derivative of lens-axis angle (radians per unit time).
    tref : float, optional
        Reference time where ``s = s0`` and ``alpha = alpha0``.

    Returns
    -------
    w_lens : Array
        Coordinates rotated into the instantaneous lens frame.
    s_t : Array
        Time-dependent separation.
    alpha_t : Array
        Time-dependent lens-axis angle.
    """
    s_t, alpha_t = linear_orbital_motion(
        t=t, s0=s0, alpha0=alpha0, ds_dt=ds_dt, dalpha_dt=dalpha_dt, tref=tref
    )
    w_lens = to_rotating_lens_frame(w, alpha_t)
    return w_lens, s_t, alpha_t


linear_orbital_motion_state_jit = jax.jit(linear_orbital_motion_state)


def _vb_circular_3d_constants(
    s0: float,
    alpha0: float,
    w1: float,
    w2: float,
    w3: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Return constants for VBBinaryLensing-style circular 3D orbital motion."""
    branch_eps = jnp.asarray(1.0e-8, dtype=dtype)

    s0 = jnp.asarray(s0, dtype=dtype)
    alpha0 = jnp.asarray(alpha0, dtype=dtype)
    w1 = jnp.asarray(w1, dtype=dtype)
    w2 = jnp.asarray(w2, dtype=dtype)
    w3 = jnp.asarray(w3, dtype=dtype)

    c_alpha0 = jnp.cos(alpha0)
    s_alpha0 = jnp.sin(alpha0)

    w13_sq = w1 * w1 + w3 * w3
    w13 = jnp.sqrt(w13_sq)
    w123 = jnp.sqrt(w13_sq + w2 * w2)

    use_3d = w13 > branch_eps
    safe_w13 = jnp.where(use_3d, w13, 1.0)
    safe_w123 = jnp.where(w123 > 0.0, w123, 1.0)
    # Match VBBinaryLensing's branch behavior, which floors w3 to +eps.
    w3_eff = jnp.where(w3 > branch_eps, w3, branch_eps)

    w_orb_3d = w3_eff * safe_w123 / safe_w13
    cos_inc_arg = (w2 * w3_eff) / (safe_w13 * safe_w123)
    cos_inc_arg = jnp.clip(cos_inc_arg, -1.0, 1.0)
    inc_3d = jnp.arccos(cos_inc_arg)
    phi0_3d = jnp.arctan2(-w1 * w123, w3_eff * safe_w13)

    w_orb = jnp.where(use_3d, w_orb_3d, w2)
    inc = jnp.where(use_3d, inc_3d, 0.0)
    phi0 = jnp.where(use_3d, phi0_3d, 0.0)

    c_phi0 = jnp.cos(phi0)
    s_phi0 = jnp.sin(phi0)
    c_inc = jnp.cos(inc)
    s_inc = jnp.sin(inc)

    den0_arg = c_phi0 * c_phi0 + c_inc * c_inc * s_phi0 * s_phi0
    den0 = jnp.sqrt(jnp.where(den0_arg > 0.0, den0_arg, jnp.finfo(dtype).tiny))
    s_true = s0 / den0

    c_Om = (c_phi0 * c_alpha0 + c_inc * s_alpha0 * s_phi0) / den0
    s_Om = (c_phi0 * s_alpha0 - c_inc * c_alpha0 * s_phi0) / den0

    return w_orb, phi0, inc, c_inc, s_inc, s_true, c_Om, s_Om


def circular_orbital_motion_3d(
    t: float | Array,
    s0: float,
    alpha0: float,
    w1: float = 0.0,
    w2: float = 0.0,
    w3: float = 0.0,
    tref: float = 0.0,
) -> Tuple[Array, Array, Array]:
    """Evaluate circular 3D orbital motion (VBBinaryLensing parameterization).

    This follows the same circular-orbit mapping used in
    ``VBBinaryLensing::BinaryLightCurveOrbital``. The input velocity-like
    parameters are the standard VBBinaryLensing definitions:

    - ``w1 = (1/s) * ds/dt``
    - ``w2 = d(alpha)/dt``
    - ``w3 = (1/s) * ds_z/dt``

    where ``s`` is projected separation and ``s_z`` is the line-of-sight
    separation component.

    Notes
    -----
    For strict compatibility with the current VBBinaryLensing implementation,
    ``w3`` is internally floored to a small positive value in the full 3D
    branch (``w1^2 + w3^2 > 0``), matching the original C++ guard.

    Parameters
    ----------
    t : float or Array
        Time(s) at which to evaluate the model.
    s0 : float
        Projected binary separation at ``tref``.
    alpha0 : float
        Lens-axis angle at ``tref`` in radians.
    w1 : float, optional
        Fractional projected-separation rate ``(1/s) ds/dt``.
    w2 : float, optional
        Position-angle rate ``d(alpha)/dt`` in radians per unit time.
    w3 : float, optional
        Fractional line-of-sight-separation rate ``(1/s) ds_z/dt``.
    tref : float, optional
        Reference time at which ``s=s0`` and ``alpha=alpha0``.

    Returns
    -------
    s_t : Array
        Projected separation at each time.
    alpha_t : Array
        Projected lens-axis angle (radians).
    sz_t : Array
        Line-of-sight separation component ``s_z`` at each time.
    """
    t = jnp.asarray(t)
    dtype = t.dtype
    dt = t - jnp.asarray(tref, dtype=dtype)

    w_orb, phi0, _inc, c_inc, s_inc, s_true, c_Om, s_Om = _vb_circular_3d_constants(
        s0=s0, alpha0=alpha0, w1=w1, w2=w2, w3=w3, dtype=dtype
    )

    phi = w_orb * dt + phi0
    c_phi = jnp.cos(phi)
    s_phi = jnp.sin(phi)

    den_arg = c_phi * c_phi + c_inc * c_inc * s_phi * s_phi
    den = jnp.sqrt(jnp.where(den_arg > 0.0, den_arg, jnp.finfo(dtype).tiny))
    s_t = s_true * den

    # Effective projected lens-axis angle in the fixed sky frame.
    sin_alpha_t = (c_phi * s_Om + c_inc * s_phi * c_Om) / den
    cos_alpha_t = (c_phi * c_Om - c_inc * s_phi * s_Om) / den
    alpha_t = jnp.arctan2(sin_alpha_t, cos_alpha_t)

    sz_t = s_true * s_inc * s_phi
    return s_t, alpha_t, sz_t


circular_orbital_motion_3d_jit = jax.jit(circular_orbital_motion_3d)


def circular_orbital_motion_3d_state(
    t: float | Array,
    w: complex | Array,
    s0: float,
    alpha0: float,
    w1: float = 0.0,
    w2: float = 0.0,
    w3: float = 0.0,
    tref: float = 0.0,
) -> Tuple[Array, Array, Array, Array]:
    """Evaluate circular 3D orbital motion and rotate trajectory to lens frame.

    Parameters
    ----------
    t : float or Array
        Time(s) corresponding to ``w``.
    w : complex or Array
        Source-plane coordinates in a fixed sky frame.
    s0 : float
        Projected binary separation at ``tref``.
    alpha0 : float
        Lens-axis angle at ``tref`` (radians).
    w1, w2, w3 : float, optional
        Circular 3D orbital-motion rates in VBBinaryLensing convention.
    tref : float, optional
        Reference time where ``s=s0`` and ``alpha=alpha0``.

    Returns
    -------
    w_lens : Array
        Coordinates rotated into the instantaneous lens frame.
    s_t : Array
        Projected separation as a function of time.
    alpha_t : Array
        Time-dependent projected lens-axis angle.
    sz_t : Array
        Time-dependent line-of-sight separation component.
    """
    s_t, alpha_t, sz_t = circular_orbital_motion_3d(
        t=t,
        s0=s0,
        alpha0=alpha0,
        w1=w1,
        w2=w2,
        w3=w3,
        tref=tref,
    )
    w_lens = to_rotating_lens_frame(w, alpha_t)
    return w_lens, s_t, alpha_t, sz_t


circular_orbital_motion_3d_state_jit = jax.jit(circular_orbital_motion_3d_state)


def _solve_kepler_equation(M: Array, ecc: Array, n_iter: int = 10) -> Array:
    """Solve Kepler's equation ``M = E - e sin(E)`` with fixed Newton steps."""
    return solve_kepler_newton(M, ecc, n_iter=n_iter, init="m_plus_esinm")


def _vb_kepler_3d_constants(
    s0: float,
    alpha0: float,
    w1: float,
    w2: float,
    w3: float,
    szs: float,
    ar: float,
    tref: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return constants for VBBinaryLensing-style Keplerian 3D orbital motion."""
    eps = jnp.asarray(1.0e-8, dtype=dtype)

    s0 = jnp.asarray(s0, dtype=dtype)
    alpha0 = jnp.asarray(alpha0, dtype=dtype)
    w1 = jnp.asarray(w1, dtype=dtype)
    w2 = jnp.asarray(w2, dtype=dtype)
    w3 = jnp.asarray(w3, dtype=dtype)
    szs = jnp.asarray(szs, dtype=dtype)
    ar = jnp.asarray(ar, dtype=dtype) + eps
    tref = jnp.asarray(tref, dtype=dtype)

    smix = 1.0 + szs * szs
    sqsmix = jnp.sqrt(smix)

    w11 = w1 * w1
    w22 = w2 * w2
    w33 = w3 * w3
    w12 = w11 + w22
    wt2 = w12 + w33

    arm1 = ar - 1.0
    arm2 = 2.0 * ar - 1.0

    n = jnp.sqrt(wt2 / arm2 / smix) / ar

    Z = jnp.array([-szs * w2, szs * w1 - w3, w2], dtype=dtype)
    h = jnp.sqrt(jnp.sum(Z * Z))
    Z = Z / h

    X = jnp.array(
        [
            -ar * w11 + arm1 * w22 - arm2 * szs * w1 * w3 + arm1 * w33,
            -arm2 * w2 * (w1 + szs * w3),
            arm1 * szs * w12 - arm2 * w1 * w3 - ar * szs * w33,
        ],
        dtype=dtype,
    )
    Xnorm = jnp.sqrt(jnp.sum(X * X))
    X = X / Xnorm
    ecc = Xnorm / (ar * sqsmix * wt2)

    Y = jnp.cross(Z, X)

    conu = (X[0] + X[2] * szs) / sqsmix
    cos_E0 = (conu + ecc) / (1.0 + ecc * conu)
    cos_E0 = jnp.clip(cos_E0, -1.0, 1.0)
    E0 = jnp.arccos(cos_E0)
    sign = jnp.where((Y[0] + Y[2] * szs) > 0.0, 1.0, -1.0)
    E0 = E0 * sign
    sin_E0 = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_E0 * cos_E0)) * sign
    t_peri = tref - (E0 - ecc * sin_E0) / n

    a = ar * s0 * sqsmix
    return alpha0, n, ecc, t_peri, a, jnp.stack([X, Y], axis=0)


def elliptic_orbital_motion_3d(
    t: float | Array,
    s0: float,
    alpha0: float,
    w1: float = 0.0,
    w2: float = 0.0,
    w3: float = 0.0,
    szs: float = 0.0,
    ar: float = 1.0,
    tref: float = 0.0,
    kepler_newton_iter: int = 10,
) -> Tuple[Array, Array, Array]:
    """Evaluate Keplerian 3D orbital motion (VBBinaryLensing compatible).

    This matches the orbital geometry used by
    ``VBBinaryLensing::BinaryLightCurveKepler`` and returns projected
    separation, projected lens-axis angle, and line-of-sight separation.

    Parameters
    ----------
    t : float or Array
        Time(s) where the orbit is evaluated.
    s0 : float
        Projected binary separation parameter used by VBBinaryLensing Kepler model.
    alpha0 : float
        Baseline angle parameter used in VBBinaryLensing Kepler model.
    w1, w2, w3 : float, optional
        Orbital-motion rates as defined by VBBinaryLensing.
    szs : float, optional
        Line-of-sight separation ratio parameter used by VBBinaryLensing.
    ar : float, optional
        Keplerian shape parameter used by VBBinaryLensing.
    tref : float, optional
        Reference epoch equivalent to VBBinaryLensing ``t0_par``.
    kepler_newton_iter : int, optional
        Number of fixed Newton iterations for Kepler's equation.

    Returns
    -------
    s_t : Array
        Projected separation as a function of time.
    alpha_t : Array
        Projected lens-axis angle as a function of time.
    sz_t : Array
        Line-of-sight separation component as a function of time.
    """
    t = jnp.asarray(t)
    dtype = t.dtype

    alpha0_, n, ecc, t_peri, a, XY = _vb_kepler_3d_constants(
        s0=s0,
        alpha0=alpha0,
        w1=w1,
        w2=w2,
        w3=w3,
        szs=szs,
        ar=ar,
        tref=tref,
        dtype=dtype,
    )
    X, Y = XY[0], XY[1]

    M = n * (t - t_peri)
    E = _solve_kepler_equation(M, ecc, n_iter=kepler_newton_iter)

    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    one_minus_e2 = jnp.maximum(0.0, 1.0 - ecc * ecc)
    r0 = a * (cos_E - ecc)
    r1 = a * jnp.sqrt(one_minus_e2) * sin_E

    x0 = r0 * X[0] + r1 * Y[0]
    x1 = r0 * X[1] + r1 * Y[1]
    x2 = r0 * X[2] + r1 * Y[2]

    s_t = jnp.sqrt(x0 * x0 + x1 * x1)
    psi_t = jnp.arctan2(x1, x0)
    alpha_t = alpha0_ + psi_t
    sz_t = x2
    return s_t, alpha_t, sz_t


elliptic_orbital_motion_3d_jit = jax.jit(
    elliptic_orbital_motion_3d, static_argnames=("kepler_newton_iter",)
)


def elliptic_orbital_motion_3d_state(
    t: float | Array,
    w: complex | Array,
    s0: float,
    alpha0: float,
    w1: float = 0.0,
    w2: float = 0.0,
    w3: float = 0.0,
    szs: float = 0.0,
    ar: float = 1.0,
    tref: float = 0.0,
    kepler_newton_iter: int = 10,
) -> Tuple[Array, Array, Array, Array]:
    """Evaluate Keplerian 3D orbital motion and rotate trajectory to lens frame."""
    s_t, alpha_t, sz_t = elliptic_orbital_motion_3d(
        t=t,
        s0=s0,
        alpha0=alpha0,
        w1=w1,
        w2=w2,
        w3=w3,
        szs=szs,
        ar=ar,
        tref=tref,
        kepler_newton_iter=kepler_newton_iter,
    )
    w_lens = to_rotating_lens_frame(w, alpha_t)
    return w_lens, s_t, alpha_t, sz_t


elliptic_orbital_motion_3d_state_jit = jax.jit(
    elliptic_orbital_motion_3d_state, static_argnames=("kepler_newton_iter",)
)
