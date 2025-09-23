Usage Guide
===========

This chapter presents common workflows, from point-source magnifications to
adaptive finite-source light curves and gradient-based inference.

Point-source magnification
--------------------------

``mag_point_source`` handles 1–3 lens configurations.  Pass complex source
coordinates and supply the relevant lens parameters::

   import jax
   import jax.numpy as jnp
   from microjax.point_source import mag_point_source

   jax.config.update("jax_enable_x64", True)

   w = jnp.array([0.0 + 0.1j, 0.05 + 0.05j])
   mu = mag_point_source(w, nlenses=2, s=1.0, q=0.01)

If ``nlenses`` is 3, provide ``q3``, ``r3`` and ``psi`` as additional keyword
arguments.  Outputs broadcast across leading axes so you can pass batches of
trajectories.

Finite-source binary lenses
---------------------------

The adaptive light-curve solver switches between the hexadecapole approximation
and full inverse-ray integrations::

   import jax
   import jax.numpy as jnp
   from microjax.inverse_ray.lightcurve import mag_binary

   jax.config.update("jax_enable_x64", True)

   s, q = 0.95, 5e-4
   rho = 0.01
   tE, u0, alpha, t0 = 40.0, 0.05, jnp.deg2rad(60.0), 0.0

   t = t0 + jnp.linspace(-2 * tE, 2 * tE, 1024)
   tau = (t - t0) / tE
   y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
   y2 =  u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
   w = jnp.array(y1 + 1j * y2, dtype=complex)

   mags = mag_binary(
       w,
       rho,
       s=s,
       q=q,
       r_resolution=768,
       th_resolution=768,
       Nlimb=400,
       MAX_FULL_CALLS=200,
       chunk_size=128,
   )

Tune ``MAX_FULL_CALLS`` to control the budget for contour integrations.  On
memory-limited devices, drop the resolutions to 512 × 512 and shrink
``chunk_size``.

Triple lenses
-------------

``mag_triple`` mirrors the binary API but currently upgrades a fixed number of
samples.  Increase ``MAX_FULL_CALLS`` and reduce ``chunk_size`` for challenging
caustic structures::

   from microjax.inverse_ray.lightcurve import mag_triple

   mags = mag_triple(
       w,
       rho,
       s=1.1,
       q=0.02,
       q3=0.5,
       r3=0.6,
       psi=jnp.deg2rad(210.0),
       MAX_FULL_CALLS=400,
   )

Autodiff and ``jit``
--------------------

All magnification routines support reverse- and forward-mode AD.  Use ``jax.jit``
to compile the forward model once, then differentiate::

   from functools import partial
   from jax import grad, jit

   def loglike(q):
       mags = mag_binary(w, rho, s=s, q=q)
       model_flux = mags * 1.0  # toy example
       return -0.5 * jnp.sum((model_flux - data_flux) ** 2)

   loglike_jit = jit(loglike)
   dloglike_dq = grad(loglike_jit)(q)

Trajectory helpers
------------------

The :mod:`microjax.trajectory.parallax` module offers building blocks for
annual-parallax trajectories.  Combine them with custom sampling strategies to
avoid missing peak magnification intervals.

Best practices
--------------

- Enable 64-bit mode for production runs.
- Batch trajectories to keep GPUs fully utilised.
- Cache compilation by reusing ``jit``-compiled callables for repeated runs.
- Use :mod:`microjax.likelihood` to marginalise over flux parameters instead of
  fitting them manually.
