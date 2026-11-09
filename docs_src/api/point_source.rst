Point sources and caustics
==========================

Use ``mag_point_source`` when the source can be treated as a point. It takes
the same lens parameters and source coordinates as the finite-source model,
with ``nlenses=2`` or ``nlenses=3`` selecting the lens system.
``critical_and_caustic_curves`` returns the curves used in the example
geometry plots.

.. autofunction:: microjax.point_source.mag_point_source

.. autofunction:: microjax.point_source.critical_and_caustic_curves

Lens mapping
------------

These lower-level functions map image positions to the source plane and
evaluate the lens-mapping Jacobian. They use the midpoint coordinates and
mass fractions described in their arguments.

.. autofunction:: microjax.point_source.lens_eq

.. autofunction:: microjax.point_source.lens_eq_det_jac
