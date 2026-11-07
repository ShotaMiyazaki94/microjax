Automatic differentiation
=========================

Use ``jax.jacfwd`` to differentiate a light curve with respect to event
parameters. The same forward-mode interface works with CPU and accelerator
backends.

A light-curve Jacobian
----------------------

This model takes seven parameters and returns a magnification at each
observation time:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from microjax.inverse_ray import mag_binary

   times = jnp.linspace(-5., 10., 32)
   parameters = jnp.array([0., 10., 0.1, 0.001, 1., jnp.deg2rad(50.), 0.005])

   def lightcurve(p):
       t0, t_e, u0, q, s, alpha, rho = p
       w = ((times - t0) / t_e + 1j * u0) * jnp.exp(1j * alpha)
       return mag_binary(w, rho, s=s, q=q, u1=0.5, backend="cpu")

   A = jax.jit(lightcurve)(parameters)
   jacobian = jax.jit(jax.jacfwd(lightcurve))(parameters)
   print(jacobian.shape)  # (32, 7)

Rows correspond to observation times; columns follow
``t0, t_e, u0, q, s, alpha, rho``. Angles are in radians.
The limb-darkening coefficient is fixed at ``u1=0.5`` here; include it in the
parameter vector to differentiate it as well.

For a triple lens, add ``q3``, ``r3``, and ``psi`` to the parameter vector
and call ``mag_triple``. The :doc:`examples/index` contain complete binary
and triple models and their Jacobian figures.

Values and derivatives together
-------------------------------

Return the light curve as auxiliary data to obtain both outputs in one call:

.. code-block:: python

   def with_value(p):
       A = lightcurve(p)
       return A, A

   jacobian, A = jax.jit(jax.jacfwd(with_value, has_aux=True))(parameters)

The first call includes compilation. Reuse the compiled function at the
same array shape for subsequent evaluations.

Directional derivatives
-----------------------

When only one parameter combination is needed, use ``jax.jvp``.
For example, the radius direction is:

.. code-block:: python

   direction = jnp.zeros_like(parameters).at[6].set(1.)
   A, dA_drho = jax.jvp(lightcurve, (parameters,), (direction,))

Forward mode is the supported route for CPU light curves; reverse mode
through its data-dependent loops is not supported. Numerical behavior near
caustics and measured derivative accuracy are described in :doc:`caveats`.
