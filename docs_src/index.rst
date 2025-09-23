Welcome to microJAX's documentation!
====================================

``microJAX`` is a GPU-aware, auto-differentiable microlensing toolkit built on
top of JAX.  The library combines fast multipole approximations with adaptive
inverse-ray contour integration to deliver accurate magnifications and
gradients for binary and triple lens systems.

Highlights
----------

- **Accelerated finite sources** – image-centred ray shooting with CUDA-ready
  batching.
- **Differentiable everywhere** – gradients flow through polynomial solvers and
  contour integrators for use in HMC/VI workflows.
- **Trajectory utilities** – helpers for parallax, limb darkening, and custom
  source motion.
- **Composable likelihoods** – analytic marginalisation utilities for fast
  photometric inference.

Quick peek
----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from microjax.inverse_ray.lightcurve import mag_binary

   jax.config.update("jax_enable_x64", True)

   t = jnp.linspace(-2.0, 2.0, 512)
   w = jnp.exp(1j * jnp.pi * t / 4)  # toy trajectory on the complex plane

   mags = mag_binary(w, rho=0.01, s=1.0, q=0.001)

Use the sections below to install the package, explore worked examples, and dig
into the API.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting_started
   usage
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Internals & Contribution

   concepts
   development

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
