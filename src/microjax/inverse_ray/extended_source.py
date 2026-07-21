"""Public finite-source boundary integrators.

This module is intentionally a small façade. Binary and triple calculations
flow as follows:

1. sample and solve the source limb;
2. build fixed-shape radial image topology;
3. optionally re-centre a certified binary planetary image component;
4. solve exact angular boundary roots at radial quadrature nodes;
5. integrate area or a radial intensity profile;
6. normalize the flux and return numerical status information.

Implementation is split by responsibility:

* ``integrators.charts`` — global versus image-local polar charts;
* ``integrators.uniform`` — uniform binary source;
* ``integrators.triple`` — uniform triple lens;
* ``integrators.profile`` — radial profiles and limb darkening.

All intermediate buffers have static capacity for JAX/XLA. A radial node is
always the physical point ``z = chart_center + r * exp(1j * theta)``. Triple
coordinates retain the centre of mass of the first two lenses, matching the
public point-source convention.
"""

from .integrators import (
    BoundaryMagnificationResult,
    mag_limb_dark_boundary,
    mag_radial_profile_boundary,
    mag_uniform_boundary,
    mag_uniform_triple_boundary,
)

__all__ = [
    "BoundaryMagnificationResult",
    "mag_limb_dark_boundary",
    "mag_radial_profile_boundary",
    "mag_uniform_boundary",
    "mag_uniform_triple_boundary",
]
