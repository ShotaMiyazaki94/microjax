microJAX
========

`microJAX <https://github.com/ShotaMiyazaki94/microjax>`_ is a differentiable
microlensing modelling library built with JAX. It provides a GPU-oriented
boundary integrator and a separate one-shot CPU backend for finite-source
binary lenses, together with point-source and triple-lens calculations.

For source positions far enough from caustics, the current finite-source API
uses a fast approximation. Where a full calculation is needed, it traces the
lensed images of the source circumference and integrates the brightness
enclosed by those image boundaries. The implementation is compatible with JAX
transformations including ``jit``, ``vmap``, and forward-mode automatic
differentiation.

Release lineage
---------------

``v0.1.1`` is the archived implementation associated with the methods paper.
The ``0.2`` series is a substantial redesign and should not be treated as a
patch-level update. Record the exact microJAX version or Git commit together
with the JAX/JAXLIB versions, platform, and numerical configuration in
reproducible work.

Highlights
----------

- Point-source magnification and caustic curves for one to three lenses.
- Binary and triple finite-source boundary integration.
- Uniform and linear limb-darkened circular sources.
- Direct calculation of image-boundary crossing angles, without an angular
  sampling grid.
- GPU-oriented trajectory batching and forward-mode Jacobians.
- A differentiable CPU binary-lens backend with fixed Cartesian and polar
  full-solve routes and fail-closed behavior.
- Parallax and binary orbital-motion trajectory utilities.

Quick peek
----------

Enable double precision before creating arrays or compiling functions.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from microjax.inverse_ray import BinaryMagConfig, mag_binary
   from microjax.point_source import mag_point_source

   s, q, rho = 1.0, 0.01, 0.02
   t = jnp.linspace(-30.0, 30.0, 1000)
   tau = t / 30.0
   alpha = jnp.deg2rad(10.0)
   w = tau * jnp.cos(alpha) + 1j * tau * jnp.sin(alpha)

   config = BinaryMagConfig(n_limb=500)
   mu_point = mag_point_source(w, nlenses=2, s=s, q=q)
   mu_finite = mag_binary(w, rho, s=s, q=q, config=config)

The first call includes JAX compilation time. The example above uses the
accelerator backend. For the CPU binary-lens path, pass ``backend="cpu"`` and
see :doc:`cpu_backend`.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting_started
   usage
   cpu_backend
   performance
   troubleshooting
   citing

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules

Accuracy and limitations
------------------------

microJAX is research software under active development. The default
accelerator path and the production CPU one-shot path use bounded work; they do
not automatically retry a difficult calculation with increasingly expensive
settings. A returned finite value is a numerical estimate, not a value with a
guaranteed error bound. If microJAX cannot construct valid image boundaries or
integration regions, the magnification API returns ``NaN``. Validate
magnifications and derivatives over the parameter region used in an analysis.

Citing microJAX
---------------

See :doc:`citing` for the methods-paper and software citations, BibTeX, and the
environment information that should accompany reproducible numerical results.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
