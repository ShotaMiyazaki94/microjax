Usage Guide
===========

This guide introduces the public ``0.2`` workflow, the available settings, and
the checks recommended before using a result in an analysis.

Common setup
------------

Enable 64-bit mode before creating arrays and use the public package imports::

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from microjax.inverse_ray import (
       BinaryMagConfig,
       TripleMagConfig,
       mag_binary,
       mag_triple,
   )
   from microjax.point_source import mag_point_source

Point-source magnification
--------------------------

``mag_point_source`` evaluates magnification for one to three point lenses.
Source coordinates are complex numbers in Einstein-radius units: the real part
is x and the imaginary part is y.

.. code-block:: python

   w = jnp.array([0.00 + 0.10j, 0.05 + 0.05j, -0.10 + 0.02j])
   mu = mag_point_source(w, nlenses=2, s=1.0, q=0.01)

For a triple lens, add ``q3``, ``r3``, and ``psi``. Here ``q3`` is the third
mass relative to lens 1, ``r3`` is its separation parameter, and ``psi`` is its
position angle in radians.

Finite-source binary lenses
---------------------------

Construct a source trajectory and pass the circular-source radius ``rho`` to
``mag_binary``.

.. code-block:: python

   t0, tE, u0 = 0.0, 40.0, 0.05
   alpha = jnp.deg2rad(60.0)
   rho = 0.01

   t = t0 + jnp.linspace(-2 * tE, 2 * tE, 1024)
   tau = (t - t0) / tE
   w = (
       -u0 * jnp.sin(alpha)
       + tau * jnp.cos(alpha)
       + 1j * (u0 * jnp.cos(alpha) + tau * jnp.sin(alpha))
   )

   config = BinaryMagConfig(n_limb=500)
   mu = mag_binary(w, rho, s=0.95, q=5e-4, config=config)

Set ``u1`` to a non-zero value for the normalized linear limb-darkening law::

   mu_ld = mag_binary(w, rho, s=0.95, q=5e-4, u1=0.5, config=config)

Triple lenses
-------------

The triple-lens API uses the same trajectory and source profile.

.. code-block:: python

   triple_config = TripleMagConfig(n_limb=500)
   mu_triple = mag_triple(
       w,
       rho,
       s=1.10,
       q=0.02,
       q3=0.50,
       r3=0.60,
       psi=jnp.deg2rad(210.0),
       u1=0.5,
       config=triple_config,
   )

Configuration
-------------

``BinaryMagConfig`` and ``TripleMagConfig`` currently expose one setting:

``n_limb``
   Number of points placed on the circular source boundary before those points
   are mapped into the image plane. Larger values follow rapid changes of the
   image boundary more finely, at additional computational cost. The default
   value is recommended for normal use. This setting does not directly change
   the number of radial integration points.

microJAX automatically chooses the remaining integration and GPU execution
settings.

Boundary integration in outline
-------------------------------

The solver first tries a fast approximation. Source positions that require the
full finite-source calculation then follow this sequence:

1. Sample the circumference of the source and map those points through the
   lens equation.
2. Connect samples that form the same continuous image of the circumference.
   Such a connected sequence is called an *image branch* in the implementation
   report.
3. For every image branch, find the range of image-plane radius that it
   occupies. Combine ranges that overlap.
4. Divide a combined range wherever the number or arrangement of boundary
   crossings may change. Each resulting radial subinterval is called a
   *radial cell*.
5. At selected radii within each cell, calculate the angles where the radius
   circle crosses the image boundary. Adjacent crossing angles determine which
   angular arcs lie inside a lensed image.
6. Integrate the surface brightness along the inside arcs, then integrate over
   radius to obtain the total lensed flux.

The subdivision in step 4 is important: one image branch can contribute to
several cells, and one combined radial range can contain several cells. Within
one cell, the pattern of boundary crossings is expected to stay unchanged.

Differentiation
---------------

Forward-mode differentiation is the recommended route for full light-curve
Jacobians.

.. code-block:: python

   def forward_model(q):
       return mag_binary(w, rho, s=0.95, q=q, config=config)

   dmu_dq = jax.jacfwd(forward_model)(5e-4)

JAX differentiation does not guarantee that the numerical result is smooth or
accurate at every point. Caustic crossings can change the number and
arrangement of images, and the code switches between an approximation and the
full calculation where appropriate. Compare values and gradients against
independent calculations over the intended parameter range.

Failure behavior
----------------

The public finite-source functions return ``NaN`` if they cannot construct a
valid image boundary or integration region, or if a calculation becomes
non-finite. A finite result is still a numerical estimate without a guaranteed
error bound. Downstream likelihood code should check for non-finite values
explicitly and validate accuracy independently.

Performance and reproducibility
-------------------------------

- The first call includes compilation. Warm up and call ``block_until_ready``
  before timing.
- Finite-source calculations are intended for GPUs, although they also run on
  CPUs.
- Record microJAX, JAX, and JAXLIB versions; the accelerator model; precision;
  and the complete configuration object with reported results.
- Benchmark values committed under ``example/`` are records for their stated
  hardware and trajectories, not universal guarantees.
