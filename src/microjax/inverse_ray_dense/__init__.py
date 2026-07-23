"""Legacy dense polar-grid inverse-ray implementation.

Submodules
- boundary: smoothed membership and boundary factors
- limb_darkening: limb-darkening intensity profiles
- merge_area: region construction for polar integration
- extended_source: dense-grid integrators (uniform, limb-darkened)
- cond_extended: tests to select between multipole and full solve
- lightcurve: dense binary and triple light curves
"""

from .lightcurve import mag_binary_dense, mag_triple

__all__ = [
    "boundary",
    "limb_darkening",
    "merge_area",
    "extended_source",
    "cond_extended",
    "lightcurve",
    "mag_binary_dense",
    "mag_triple",
]
