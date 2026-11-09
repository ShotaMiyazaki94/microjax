API reference
=============

For most light-curve models, start with ``mag_binary`` or ``mag_triple`` in
the finite-source API. It also documents the lens coordinates, source profile,
and CPU/GPU settings. The point-source API includes magnifications, lens mappings, and
critical and caustic curves.

.. toctree::
   :maxdepth: 1

   api/inverse_ray_lightcurve
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
