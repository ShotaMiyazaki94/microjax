Polynomial roots
================

These routines find the complex roots used to locate lens images.
Coefficients follow NumPy's ``polyval`` convention, with the highest power
first. ``poly_roots`` is the general entry point.

.. autofunction:: microjax.poly_solver.poly_roots

.. autofunction:: microjax.poly_solver.poly_roots_EA

.. autofunction:: microjax.poly_solver.poly_roots_EA_multi

.. autofunction:: microjax.poly_solver.poly_roots_EA_self_inversive_fixed
