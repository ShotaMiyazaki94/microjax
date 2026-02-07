"""Inverse-ray style convenience wrapper around microlux contour integration.

This exposes a :func:`mag_binary` function with the same call signature as
``microjax.inverse_ray.lightcurve.mag_binary`` (first argument is a complex
trajectory ``w_points`` in the centre-of-mass frame). Internally it converts
that trajectory to the low-mass microlux frame and calls the existing contour
integrator.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple, Any

import jax
import jax.numpy as jnp

from .basic_function import to_lowmass
from .integrate import IntegratorOptions, integrate


@partial(
    jax.jit,
    static_argnames=(
        "tol",
        "retol",
        "default_strategy",
        "analytic",
        "return_info",
        "limb_darkening_coeff",
        "n_annuli",
    ),
)
def mag_binary(
    w_points: jax.Array,
    rho: float,
    *,
    s: float,
    q: float,
    tol: float = 1e-2,
    retol: float = 1e-3,
    default_strategy: Tuple[int, ...] = (30, 30, 60, 120, 240),
    analytic: bool = True,
    return_info: bool = False,
    limb_darkening_coeff: float | None = None,
    n_annuli: int = 10,
) -> jax.Array | tuple[jax.Array, Any]:
    """Finite-source binary magnification for a precomputed trajectory.

    Parameters
    ----------
    w_points : jax.Array
        Complex source-plane coordinates in the centre-of-mass frame (matching
        ``inverse_ray.lightcurve.mag_binary``). Shape ``(N,)``.
    rho : float
        Source radius in Einstein units.
    s, q : float
        Binary separation and mass ratio.
    tol, retol, default_strategy, analytic
        Microlux contour integration controls (see ``IntegratorOptions``).
    return_info : bool, optional
        If ``True``, also return the microlux debug info alongside magnification.
    limb_darkening_coeff : float or None, optional
        Linear limb-darkening coefficient. ``None`` for uniform brightness.
    n_annuli : int, optional
        Number of annuli used when limb darkening is enabled.

    Returns
    -------
    jax.Array or (jax.Array, Any)
        Magnification array (and optional info when ``return_info=True``).
    """

    trajectory_l = to_lowmass(s, q, jnp.asarray(w_points))

    opts = IntegratorOptions(
        tol=tol,
        retol=retol,
        default_strategy=default_strategy,
        analytic=analytic,
        return_info=return_info,
        limb_darkening_coeff=limb_darkening_coeff,
        n_annuli=n_annuli,
    )

    result = integrate(
        source={"trajectory": trajectory_l, "rho": rho},
        lens={"s": s, "q": q},
        options=opts,
    )

    return (result.mu, result.info) if return_info else result.mu
