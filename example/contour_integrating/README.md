Caustics Contour Test
=====================

Light-curve benchmark that exercises `microjax.caustics.lightcurve.magnifications`
and compares the result with `VBBinaryLensing` for a binary lens.

Usage
-----
1. Install `VBBinaryLensing` (the reference solver)::

       pip install VBBinaryLensing

2. Run::

       python compare_vbbl.py

The script JIT-compiles the contour integrator, evaluates a 1000-point
trajectory, reports timings, and writes `compare_binary_uniform.png` with the
light curve and relative differences. Double-precision JAX is enabled inside
the script; CPU execution is sufficient for the default settings.
