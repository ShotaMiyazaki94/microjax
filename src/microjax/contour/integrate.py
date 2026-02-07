"""
Microlux-based, JAX-differentiable contour integration wrapper for microjax.

This module is a thin adapter around the original `microlux` implementation
(`CoastEgo/microlux`, MIT). It exposes a stable, minimal API that accepts
complex source positions and binary-lens parameters and returns
JAX-differentiable magnifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from .model import extended_light_curve


@dataclass
class IntegratorOptions:
    """Runtime configuration mirroring microlux defaults."""

    tol: float = 1e-2
    retol: float = 1e-3
    default_strategy: Tuple[int, ...] = (30, 30, 60, 120, 240)
    analytic: bool = True
    return_info: bool = False
    limb_darkening_coeff: float | None = None
    n_annuli: int = 10


@dataclass
class MagnificationResult:
    """Structured output for contour integration."""

    mu: jax.Array  # magnification array (same length as input trajectory)
    cond: jax.Array | None  # quadrupole-test validity flags (if available)
    info: Any | None  # optional debug info from microlux
    diagnostics: Dict[str, Any]


def integrate(
    source: Dict[str, Any],
    lens: Dict[str, Any],
    *,
    options: Optional[IntegratorOptions] = None,
) -> MagnificationResult:
    """
    Compute finite-source magnification via microlux contour integration.

    Parameters
    ----------
    source : dict
        Must contain either:
          - ``trajectory``: complex array of source positions (low-mass frame),
            shape (N,), or
          - ``w0``: single complex source position (will be wrapped to length-1).
        Also requires ``rho`` (source radius, Einstein units).
    lens : dict
        Requires binary-lens parameters ``s`` (separation) and ``q`` (mass ratio).
    options : IntegratorOptions, optional
        Tuning knobs; defaults mirror microlux.

    Returns
    -------
    MagnificationResult
        magnification array plus optional diagnostics.
    """
    opts = options or IntegratorOptions()

    rho = jnp.asarray(source["rho"])

    if "trajectory" in source:
        trajectory_l = jnp.asarray(source["trajectory"])
    elif "w0" in source:
        trajectory_l = jnp.asarray([source["w0"]])
    else:
        raise ValueError("source must provide 'trajectory' or 'w0'.")

    s = jnp.asarray(lens["s"])
    q = jnp.asarray(lens["q"])

    mag = extended_light_curve(
        trajectory_l,
        s,
        q,
        rho,
        tol=opts.tol,
        retol=opts.retol,
        default_strategy=opts.default_strategy,
        analytic=opts.analytic,
        return_info=opts.return_info,
        limb_darkening_coeff=opts.limb_darkening_coeff,
        n_annuli=opts.n_annuli,
    )

    info = None
    cond = None
    if opts.return_info:
        mag, info = mag
    # microlux returns (mag, cond) for quadrupole test inside info? keep placeholder

    diagnostics = {
        "tol": opts.tol,
        "retol": opts.retol,
        "default_strategy": opts.default_strategy,
        "analytic": opts.analytic,
        "rho": rho,
    }

    return MagnificationResult(mu=mag, cond=cond, info=info, diagnostics=diagnostics)
