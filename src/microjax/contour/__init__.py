"""
Microlux-based contour integration (JAX, binary lens) for microjax.

This subpackage vendors code from `CoastEgo/microlux` (MIT) and exposes a thin
wrapper `integrate` to fit microjax conventions.
"""

from __future__ import annotations

import warnings

from jax import config as _jax_config

if not _jax_config.read("jax_enable_x64"):
    warnings.warn(
        "microjax.contour recommends enabling 64-bit precision. "
        "Call jax.config.update('jax_enable_x64', True) before importing "
        "microjax.contour for best numerical stability.",
        RuntimeWarning,
        stacklevel=2,
    )

from .integrate import IntegratorOptions, MagnificationResult, integrate
from .model import binary_mag, extended_light_curve, point_light_curve
from .mag_binary import mag_binary
from .countour import contour_integral  # microlux spelling kept
from .utils import Error_State, Iterative_State
from .basic_function import to_lowmass, to_centroid

__all__ = [
    "integrate",
    "IntegratorOptions",
    "MagnificationResult",
    "binary_mag",
    "mag_binary",
    "extended_light_curve",
    "point_light_curve",
    "contour_integral",
    "Iterative_State",
    "Error_State",
    "to_lowmass",
    "to_centroid",
]
