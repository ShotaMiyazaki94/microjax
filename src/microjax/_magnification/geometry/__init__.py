"""Image-plane geometry used before numerical area integration.

Flow
----
``limb.calc_source_limb`` maps sampled source-boundary points to image roots.
``topology.track_limb_images`` joins those roots into branches, optionally
groups narrow planetary branches into local charts, and produces radial support
intervals.  ``mapping.distance_from_source`` maps a quadrature node back to its
source-centred radius for non-uniform brightness profiles.

This layer may depend on point-source lens geometry, but never on quadrature
rules or the high-level light-curve scheduler.
"""

from .lens import BinaryGeometry, binary_geometry

__all__ = ["BinaryGeometry", "binary_geometry"]
