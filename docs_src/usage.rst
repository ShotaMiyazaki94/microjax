Usage Guide
===========

This guide introduces the public ``0.2`` API through common calculations. See
:doc:`getting_started` first for installation and environment checks.

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

Finite-source point lens
------------------------

For a circular source magnified by one point lens, construct an FSPL source
profile and evaluate its ``A(u, rho)`` method. Here ``u`` is the lens-source
separation and ``rho`` is the source radius, both in Einstein-radius units.

.. code-block:: python

   from microjax.fspl import fspl_disk, fspl_ld1

   u = jnp.linspace(0.0, 1.0, 1000)
   mu_uniform = fspl_disk().A(u, rho=0.01)
   mu_limb_darkened = fspl_ld1(a1=0.5).A(u, rho=0.01)

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

   mu = mag_binary(w, rho, s=0.95, q=5e-4)

Set ``u1`` to a non-zero value for the normalized linear limb-darkening law::

   mu_ld = mag_binary(w, rho, s=0.95, q=5e-4, u1=0.5)

For the binary-lens CPU one-shot backend, select ``backend="cpu"``:

.. code-block:: python

   mu_cpu = mag_binary(
       w,
       rho,
       s=0.95,
       q=5e-4,
       u1=0.5,
       backend="cpu",
   )

The CPU path has fixed internal support and quadrature settings;
``BinaryMagConfig`` does not tune it. See :doc:`cpu_backend` for backend
selection, forward-mode differentiation, and validation.

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

For the accelerator backend, ``BinaryMagConfig`` and ``TripleMagConfig`` expose
source-boundary sampling and static scheduling:

``n_limb``
   Number of points placed on the circular source boundary before those points
   are mapped into the image plane. Larger values follow rapid changes of the
   image boundary more finely, at additional computational cost. The default
   is recommended for normal use.

``source_tile_size``
   Number of full-solve source positions in an outer vectorized tile. The
   binary default is 512 so ordinary GPU calls expose at least 500 positions
   concurrently when available.

``radial_chunk_size``
   Number of local-chart radial image regions in an inner vectorized chunk.
   The binary fast path uses this local chart across the full mass-ratio range
   with one fixed 19-point radial rule and no comparison retries.

Both scheduler sizes must be positive integers.

The scheduler settings produce separately compiled JAX executables. Keep them
fixed within a fit and benchmark changes on the actual analysis workload. See
:doc:`performance`.

Differentiation
---------------

Forward-mode differentiation is the recommended route for full light-curve
Jacobians. This example uses the accelerator backend; CPU-specific guidance is
given in :doc:`cpu_backend`.

.. code-block:: python

   def forward_model(q):
       return mag_binary(w, rho, s=0.95, q=q, config=config)

   dmu_dq = jax.jacfwd(forward_model)(5e-4)

Automatic differentiation does not certify numerical accuracy. Validate
values and derivatives over the intended parameter range; see :doc:`caveats`.

Failure behavior
----------------

The public finite-source functions return ``NaN`` for detected structural or
non-finite failures. Check the result before passing it to downstream inference
code and retain the complete configuration for rejected samples. A finite
result has no guaranteed error bound; :doc:`caveats` defines the numerical
contract.

For timing methodology and reproducibility metadata, see :doc:`performance`
and :doc:`citing`.
