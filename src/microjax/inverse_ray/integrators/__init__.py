"""Finite-source integration layer.

The modules follow the numerical flow rather than the public API surface:

``charts`` selects global or image-local polar coordinates; ``uniform`` and
``triple`` integrate uniform sources; ``profile`` adds radial brightness
profiles.  :mod:`microjax.inverse_ray.extended_source` is the public façade.
"""

from .common import BoundaryMagnificationResult
from .limb_dark import mag_limb_dark_boundary
from .profile import mag_radial_profile_boundary
from .triple import mag_uniform_triple_boundary
from .uniform import mag_uniform_boundary

__all__ = [
    "BoundaryMagnificationResult",
    "mag_limb_dark_boundary",
    "mag_radial_profile_boundary",
    "mag_uniform_boundary",
    "mag_uniform_triple_boundary",
]
