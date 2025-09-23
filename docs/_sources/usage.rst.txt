Usage Guide
===========

This chapter walks through the most common workflows in microJAX and explains
what each knob does.  The goal is to provide copy-and-pasteable snippets along
with the context required to adapt them to your own microlensing problem.

Common setup
------------

Start every session by enabling 64-bit mode and importing the building blocks
you intend to use.  Keeping everything in one place makes it easier to reuse the
same configuration across notebooks or scripts::

   import jax
   import jax.numpy as jnp

   from microjax.point_source import mag_point_source
   from microjax.inverse_ray.lightcurve import mag_binary, mag_triple

   jax.config.update("jax_enable_x64", True)  # stabilises the polynomial solver

The snippets below assume this cell has already been run.  If you restart your
Python session, rerun it before continuing.

Point-source magnification
--------------------------

Use ``mag_point_source`` when the source can be treated as infinitesimally small
and you need fast magnifications for one to three lenses.

Step-by-step
~~~~~~~~~~~~

1. Assemble the complex source coordinates.  The real part is the x-position,
   the imaginary part is the y-position in Einstein radii.
2. Specify the lens configuration via ``nlenses`` and the associated parameters.
3. Call ``mag_point_source``; the function broadcasts across any leading axes of
   ``w`` so batches are handled automatically.

Example::

   w = jnp.array([
       0.00 + 0.10j,
       0.05 + 0.05j,
       -0.10 + 0.02j,
   ])

   mu = mag_point_source(w, nlenses=2, s=1.0, q=0.01)

   print("Magnification per sample:", mu)

``nlenses=3`` introduces a third body.  Provide the additional keywords ``q3``
(mass ratio of lens 3 to lens 1), ``r3`` (distance between lens 1 and 3), and
``psi`` (position angle of lens 3, in radians).  All other keyword arguments are
fully broadcastable and can be supplied as arrays if you want to sweep over a
grid of lens parameters.

.. list-table:: Common ``mag_point_source`` parameters
   :header-rows: 1

   * - Parameter
     - Meaning
     - Typical range
   * - ``s``
     - Lens separation in Einstein radii
     - 0.5–3.0 for planetary lenses
   * - ``q``
     - Secondary-to-primary mass ratio
     - 1e-4–1 for binary lenses
   * - ``q3`` / ``r3`` / ``psi``
     - Third-body configuration (only for triples)
     - Chosen per system

Finite-source binary lenses
---------------------------

``mag_binary`` computes finite-source light curves by combining a fast
hexadecapole approximation with full inverse-ray integrations when required.
The workflow is a little longer because you must supply a trajectory for the
source.

1. Build the trajectory
~~~~~~~~~~~~~~~~~~~~~~~

The helper below constructs a standard rectilinear trajectory.  Feel free to
replace it with your own sampler if you need orbital motion or parallax.

.. code-block:: python

   tE = 40.0                      # Einstein time (days)
   u0 = 0.05                      # impact parameter
   alpha = jnp.deg2rad(60.0)      # trajectory angle in radians
   t0 = 0.0                       # time of closest approach
   rho = 0.01                     # source radius in Einstein units

   t = t0 + jnp.linspace(-2 * tE, 2 * tE, 1024)
   tau = (t - t0) / tE
   y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
   y2 =  u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
   w = jnp.array(y1 + 1j * y2, dtype=complex)

2. Evaluate the magnification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``mag_binary`` with the trajectory, source radius, and lens parameters.  To
start, stick with the defaults for the optional arguments and only adjust them
if you hit performance limits.

.. code-block:: python

   s = 0.95                       # projected separation
   q = 5e-4                       # mass ratio (m2/m1)

   mags = mag_binary(
       w,
       rho,
       s=s,
       q=q,
       MAX_FULL_CALLS=200,        # budget for contour integrations
   )

``mag_binary`` returns magnifications aligned with the input trajectory.  If you
need fluxes, multiply by the intrinsic source flux and add blends or baselines
as appropriate.

Fine-tuning parameters
~~~~~~~~~~~~~~~~~~~~~~

- ``r_resolution`` / ``th_resolution`` control the radial and angular grid used
  by the inverse-ray solver.  Defaults (768) balance accuracy and memory usage.
- ``Nlimb`` sets the number of points in the limb-darkening lookup table.  Lower
  it (e.g. 256) if you run on memory-constrained GPUs.
- ``chunk_size`` dictates how many points are processed per device launch.
  Shrink it when working with long trajectories or lower-end GPUs.
- ``MAX_FULL_CALLS`` caps how many samples fall back to contour integration.  A
  higher value yields more accurate peaks at the cost of runtime.

.. note::

   If you see oscillations near caustics, increase ``MAX_FULL_CALLS`` and, if
   memory allows, raise ``r_resolution``/``th_resolution`` in tandem.

Triple lenses
-------------

Triple-lens finite-source calculations are handled by ``mag_triple``.  The
inputs mirror the binary API, but you must describe the third body explicitly.

.. code-block:: python

   mags_triple = mag_triple(
       w,
       rho,
       s=1.10,
       q=0.02,
       q3=0.50,
       r3=0.60,
       psi=jnp.deg2rad(210.0),
       MAX_FULL_CALLS=400,
   )

Guidelines:

- Start with the same trajectory used for the binary case; only the lens system
  changes.
- Triple lenses often have more intricate caustics.  Expect to raise
  ``MAX_FULL_CALLS`` and possibly lower ``chunk_size`` to avoid excessive memory
  pressure.
- ``psi`` is measured counter-clockwise from the lens 1–2 axis.

Autodiff and ``jit``
--------------------

All magnification routines are differentiable.  Wrapping them in ``jax.jit``
gives you compiled performance, and ``jax.grad`` / ``jax.jacrev`` provide
derivatives for inference.

.. code-block:: python

   from functools import partial
   from jax import grad, jacrev, jit

   data_flux = jnp.load("example_lightcurve.npy")

   def forward_model(q):
       mags = mag_binary(w, rho, s=s, q=q)
       return mags  # replace with instrument model if needed

   forward_jit = jit(forward_model)

   def neg_log_like(q):
       model = forward_jit(q)
       resid = model - data_flux
       return 0.5 * jnp.sum(resid ** 2)

   g = grad(neg_log_like)(q)
   J = jacrev(forward_jit)(q)

``jacrev`` is especially useful when fitting multiple parameters simultaneously
or when propagating uncertainties through a light-curve model.

Trajectory helpers
------------------

For trajectories beyond straight lines, the :mod:`microjax.trajectory` package
provides composable pieces:

- :mod:`microjax.trajectory.parallax` – annual parallax terms.
- :mod:`microjax.trajectory.keplerian` – Keplerian binary motion (work in
  progress; check docstrings for up-to-date status).
- :mod:`microjax.trajectory.utils` – utilities for resampling and interpolation.

These components return arrays compatible with the ``w`` input used above, so
you can drop them into ``mag_binary`` / ``mag_triple`` without further changes.

Best practices
--------------

- Keep 64-bit mode enabled for production runs; it significantly improves the
  stability of implicit differentiation through the polynomial solver.
- Batch trajectories by stacking them along a leading dimension and rely on JAX
  broadcasting to evaluate many light curves in a single call.
- Cache compiled callables (e.g. store ``forward_jit``) whenever you sweep over
  parameters; recompiling for every call erodes the benefit of JIT.
- Use :mod:`microjax.likelihood` to marginalise nuisance flux parameters instead
  of fitting them manually—this often reduces sampler autocorrelation.
