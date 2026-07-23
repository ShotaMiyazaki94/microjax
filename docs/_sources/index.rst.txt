microJAX
========

`microJAX <https://github.com/ShotaMiyazaki94/microjax>`_ is a
differentiable, GPU-accelerated microlensing modelling library built with JAX.
It provides point-source calculations and finite-source light curves for
binary and triple lens systems.

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

The first call includes JAX compilation time. Finite-source calculations run
on CPUs but are intended primarily for GPU execution.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting_started
   usage
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules

Accuracy and limitations
------------------------

microJAX is research software under active development. The public
``mag_binary`` and ``mag_triple`` functions use a fixed amount of work for
each source position; they do not automatically repeat a difficult calculation
with increasingly expensive settings. A returned finite value is a numerical
estimate, not a value with a guaranteed error bound. If microJAX cannot
construct valid image boundaries or integration regions, it returns ``NaN``.
Validate magnifications and derivatives over the parameter region used in an
analysis.

Citing microJAX
---------------

If you use microJAX, cite the methods paper and the archived software version
actually used. The methods paper corresponds to the ``v0.1.1`` line; work
using the redesigned solver should additionally report the exact ``0.2.x``
release or Git commit.

- Miyazaki, S., & Kawahara, H. 2025, ApJ, 994, 144,
  `doi:10.3847/1538-4357/ae1005 <https://doi.org/10.3847/1538-4357/ae1005>`_
- microJAX software archive,
  `doi:10.5281/zenodo.17247892 <https://doi.org/10.5281/zenodo.17247892>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
