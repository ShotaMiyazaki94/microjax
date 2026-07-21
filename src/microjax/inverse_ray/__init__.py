"""Single-pass binary- and triple-lens boundary inverse-ray integration.

Public entry point
------------------
``mag_binary`` / ``mag_triple`` and their immutable configs form the public
API. Both solvers evaluate a complete trajectory with the following pipeline::

    source trajectory
          |
          v
    multipole prefilter                 lightcurve.py / selection.py
          |
          | rejected samples only
          v
    trace circular source limb          geometry/limb.py
          |
          v
    track image branches and choose     geometry/topology.py
    the polar chart
          |
          v
    construct exact Fourier level set   roots/level_set.py
    and solve angular boundary roots    roots/angular.py
          |
          v
    integrate angular brightness and    quadrature/angular.py
    radial image area once              quadrature/radial.py
          |
          v
    scatter boundary values into the trajectory

Layering
--------
``lightcurve`` owns batching and approximation selection. ``extended_source``
orchestrates one finite-source boundary solve. ``geometry`` discovers image
support, ``roots`` locates exact source-boundary crossings, and ``quadrature``
integrates already-discovered intervals. ``profiles`` contains brightness laws
only. Lower layers do not import the scheduler or finite-source orchestrator.

The retry-capable and legacy dense algorithms intentionally live in sibling
packages ``inverse_ray_retry`` and ``inverse_ray_dense``; this package contains
only the current retry-free single-pass implementation. Binary planetary
images and spatially isolated small-angle triple images may use local charts;
large Einstein-ring branches retain the public global chart.
"""

from .config import BinaryMagConfig, TripleMagConfig
from .lightcurve import mag_binary, mag_triple

__all__ = ["BinaryMagConfig", "TripleMagConfig", "mag_binary", "mag_triple"]
