"""Inverse-ray integration utilities for finite-source microlensing.

Submodules
- boundary: smoothed membership and boundary factors
- limb_darkening: limb-darkening intensity profiles
- merge_area: region construction for polar integration
- extended_source: core integrators (uniform, limb-darkened)
- cond_extended: tests to select between multipole and full solve
- lightcurve: adaptive lightcurve mixing multipole and full inverse-ray
"""

__all__ = [
    "boundary",
    "limb_darkening",
    "merge_area",
    "extended_source",
    "cond_extended",
    "lightcurve",
]
