API reference
=============

For event light curves, start with ``make_binary_lightcurve`` or
``make_triple_lightcurve`` in the finite-source API. The same page covers
source-position functions, lens coordinates, and CPU/GPU settings.
The point-source API includes magnifications, lens mappings, and critical
and caustic curves.

.. toctree::
   :maxdepth: 1

   api/magnification
   api/point_source
   api/trajectory
   api/likelihood

Lower-level routines
--------------------

These modules provide the finite-source approximations and numerical tools
used by the light-curve functions.

.. toctree::
   :maxdepth: 1

   api/multipole
   api/poly_solver
   api/utils
