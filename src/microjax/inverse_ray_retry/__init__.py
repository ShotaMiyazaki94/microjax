"""Legacy boundary inverse-ray implementation with bounded retries.

Submodules
- boundary: smoothed membership and boundary factors
- limb_darkening: limb-darkening intensity profiles
- merge_area: region construction for polar integration
- extended_source: core integrators (uniform, limb-darkened)
- cond_extended: tests to select between multipole and full solve
- lightcurve: safety-first ``mag_binary_safe`` implementation
"""

from .lightcurve import mag_binary_safe

__all__ = [
    "boundary",
    "limb_darkening",
    "merge_area",
    "extended_source",
    "cond_extended",
    "lightcurve",
    "mag_binary_safe",
]
