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

Finite-source binary lenses
---------------------------

``mag_binary`` computes finite-source light curves by combining a fast
hexadecapole approximation with full inverse-ray integrations when required.

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
   w_points = jnp.array(y1 + 1j * y2, dtype=complex)   # source trajectory

2. Evaluate the magnification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``mag_binary`` with the trajectory, source radius, and lens parameters.  To
start, stick with the defaults for the optional arguments and only adjust them
if you hit performance limits.

.. code-block:: python

   s = 0.95                       # projected separation
   q = 5e-4                       # mass ratio (m2/m1)
   mags = mag_binary(w_points, rho, s=s, q=q)

``mag_binary`` returns magnifications aligned with the input trajectory.  If you
need fluxes, multiply by the intrinsic source flux and add blends or baselines
as appropriate.

Fine-tuning parameters
~~~~~~~~~~~~~~~~~~~~~~

- ``r_resolution`` / ``th_resolution``  
  Set the number of grid divisions in the radial and angular directions for the 
  inverse-ray shooting method. Increasing these values improves the accuracy of 
  the magnification calculation, but also raises computational and memory costs 
  on GPUs. Users should adjust them according to their accuracy requirements 
  and hardware limits.
- ``MAX_FULL_CALLS``  
  Determines the maximum number of magnification points that are computed with 
  the image-centered ray-shooting (ICRS) method. It sets an upper limit on the 
  points that require finite-source calculations, with the remaining points 
  evaluated using the hexadecapole approximation.
- ``chunk_size``  
  Controls how many points are processed in parallel by the ICRS method via 
  ``jax.vmap``. A larger value can improve GPU utilization but may exceed 
  device memory, causing out-of-memory errors. Smaller values are safer but may 
  slow down the computation. Users should tune this parameter based on their 
  GPU capacity.
- ``Nlimb``  
  Sets the number of source limb points used to construct annular sectors on the 
  lens plane, where ray-shooting integrations are performed. In most cases, 
  users do not need to change this value. Adjust it only if catastrophic errors 
  appear in magnification calculations.

Triple lenses
-------------

Triple-lens finite-source calculations are handled by ``mag_triple``.  The
inputs mirror the binary API, but you must describe the third body explicitly.

.. code-block:: python

   mags_triple = mag_triple(w_points, rho, 
                            s=1.10,                 # separation between 1st and 2nd lenses
                            q=0.02,                 # mass ratio (m2/m1)
                            q3=0.50,                # mass ratio (m3/m1)
                            r3=0.60,                # separation between center of masss for m1/m2 and m3
                            psi=jnp.deg2rad(210.0)  # angle of 3rd lens axis in radians 
                            )

Guidelines:

- Start with the same trajectory used for the binary case; only the lens system
  changes.
- ``psi`` is measured counter-clockwise from the lens 1–2 axis.

Autodiff and ``jit``
--------------------

All magnification routines are differentiable.  Wrapping them in ``jax.jit``
gives you compiled performance, and ``jax.jacfwd`` provide derivatives for 
inference.

.. code-block:: python

   from functools import partial
   from jax import jacfwd, jit

   def forward_model(q):
       mags = mag_binary(w, rho, s=s, q=q)
       return mags  # replace with instrument model if needed

   forward_jit = jit(forward_model)
   J = jacfwd(forward_jit)(q)


Note: The reverse-mode automatic differentiation in ``microJAX`` is currently 
under development due to memory handling issues.


Trajectory helpers
------------------

For trajectories beyond straight lines, the :mod:`microjax.trajectory` package
provides composable pieces:

- :mod:`microjax.trajectory.parallax` – annual parallax terms.

These components return arrays compatible with the ``w`` input used above, so
you can drop them into ``mag_binary`` / ``mag_triple`` without further changes.

Best practices
--------------

- Keep 64-bit mode enabled for production runs; it significantly improves the
  stability of implicit differentiation through the polynomial solver.
- Use :mod:`microjax.likelihood` to marginalise nuisance flux parameters instead
  of fitting them manually—this often reduces sampler autocorrelation.
