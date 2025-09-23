Usage Guide
===========

This section highlights common operations in ``microJAX``. Each snippet is
ready to paste into a notebook or script once the package is installed.

Point-source magnification
--------------------------

Compute point-source magnification for binary or triple lens systems with
``mag_point_source``::

   import jax
   import jax.numpy as jnp
   from microjax.point_source import mag_point_source

   jax.config.update("jax_enable_x64", True)

   # Source trajectory in complex notation (x + i y)
   w = jnp.array([0.0 + 0.1j, 0.05 + 0.05j])

   mu = mag_point_source(w, nlenses=2, s=1.0, q=0.01)
   print(mu)

The function supports ``nlenses=1, 2, 3``. Parameters ``s`` (separation) and
``q`` (mass ratio) follow standard microlensing conventions.

Extended-source light curves
----------------------------

Finite-source magnification is implemented in ``microjax.inverse_ray.lightcurve``.
The ``mag_binary`` function uses image-centred ray shooting with a hexadecapole
fallback:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from microjax.inverse_ray.lightcurve import mag_binary

   jax.config.update("jax_enable_x64", True)

   # Binary-lens parameters
   s, q = 1.0, 0.01
   rho = 0.02
   tE, u0, alpha, t0 = 30.0, 0.0, jnp.deg2rad(10.0), 0.0

   npts = 800
   t = t0 + jnp.linspace(-2 * tE, 2 * tE, npts)
   tau = (t - t0) / tE
   y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
   y2 =  u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
   w_points = jnp.array(y1 + 1j * y2, dtype=complex)

   mu = mag_binary(
       w_points,
       rho,
       s=s,
       q=q,
       r_resolution=750,
       th_resolution=750,
       Nlimb=400,
   )

``mag_binary`` accepts many tuning parameters that trade accuracy for execution
speed. Start with the defaults above, then adjust ``r_resolution``,
``th_resolution`` and ``Nlimb`` to match your GPU's memory budget.

Trajectory helpers
------------------

The :mod:`microjax.trajectory` package contains helper routines for building the
source-plane path. For instance, ``microjax.trajectory.parallax`` exposes
functions to add annual parallax to straight-line trajectories. See the API
reference for full details.

Tips
----

- Enable 64-bit JAX globally when you care about numerical stability.
- To run just the GPU-accelerated tests, execute ``pytest -m gpu``; CPU-only
  test runs default to skipping those cases.
- Many functions are ``jax.jit`` compatible. Use JAX's ``jit`` and ``vmap`` to
  vectorise forward models in your inference code.
