microJAX
========

Differentiable binary- and triple-lens microlensing in JAX, with finite
sources, limb darkening, and light-curve Jacobians on CPUs and GPUs.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from microjax.inverse_ray import mag_binary

   times = jnp.linspace(-5., 10., 101)
   t0, t_e, u0 = 0., 10., 0.1
   alpha = jnp.deg2rad(50.)
   w = ((times - t0) / t_e + 1j * u0) * jnp.exp(1j * alpha)

   lightcurve = jax.jit(lambda rho: mag_binary(
       w, rho, s=1., q=0.001, u1=0.5, backend="cpu",
   ))
   A = lightcurve(0.005)

Supply source positions as a complex array in Einstein-radius units.
``mag_binary`` and ``mag_triple`` evaluate circular finite sources;
``u1`` sets the linear limb darkening. Use ``backend="cpu"`` for the
dedicated CPU integrator or ``backend="accelerator"`` for GPU batches.

Continue with :doc:`usage` and :doc:`autodiff`, or run the illustrated
:doc:`examples/index`. Single-lens models are available in
`singlens <https://github.com/ShotaMiyazaki94/singlens>`_.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   getting_started
   usage
   autodiff

.. toctree::
   :caption: Modeling
   :maxdepth: 1

   cpu_backend
   examples/index

.. toctree::
   :caption: Reference
   :maxdepth: 1

   modules
   caveats
   benchmarks
   performance
   citing
