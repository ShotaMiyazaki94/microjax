microJAX
====================================

`microJAX <https://github.com/ShotaMiyazaki94/microjax>`_ is a GPU-aware, 
auto-differentiable microlensing toolkit built on top of JAX.  
The library combines GPU-optimized image-centered inverse-ray 
shooting method and JAX-enabled XLA-acceralation to deliver fast and accurate 
magnifications and gradients for binary and triple lens systems.

Highlights
----------

- **Accelerated finite sources** – image-centered ray shooting (ICRS) 
  with CUDA-ready batching.
- **Differentiable everywhere** – gradients flow through polynomial solvers 
  and ICRS for use in optimization and inference (e.g. HMC/VI) workflows.
- **Other Utilities** – helpers for higher-order microlensing effects 
  like orbital parallax, limb darkening, custom source motion, and more.
- **Composable likelihoods** – analytic marginalisation utilities for inference.

Quick peek
----------
Note: ``mag_binary`` also works on CPU but is very slow.

.. code-block:: python
  
  import jax
  import jax.numpy as jnp
  from microjax.point_source import mag_point_source
  from microjax.inverse_ray.lightcurve import mag_binary
  from microjax.point_source import critical_and_caustic_curves
  jax.config.update("jax_enable_x64", True)

  # Binary-lens parameters
  s, q = 1.0, 0.01            # separation and mass ratio (m2/m1)
  rho = 0.02                  # source radius (Einstein units)
  tE, u0 = 30.0, 0.0          # Einstein time [days], impact parameter
  alpha = jnp.deg2rad(10.0)   # trajectory angle in radian
  t0 = 0.0
  
  # Source trajectory
  N_points = 1000
  t = t0 + jnp.linspace(-tE, tE, N_points)
  tau = (t - t0)/tE
  y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
  y2 =  u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
  w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
  
  # Point-source and Extended-source magnifications (binary lens)
  mag_p   = mag_point_source(w_points, s=s, q=q, nlenses=2)
  mag_ext = mag_binary(w_points, rho, s=s, q=q)

  # Critical and caustic curves
  crit, cau = critical_and_caustic_curves(s=s, q=q, nlenses=2, npts=1000) 
  

Use the sections below to install the package, explore worked examples, and dig
into the API.

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References 
---------------------


License & Attribution
---------------------

Copyright 2025, Contributors

- `Shota Miyazaki <https://sites.google.com/view/shotamiyazaki/english>`_ (@ShotaMiyazaki94, maintainer)
- `Hajime Kawahara <http://secondearths.sakura.ne.jp/en/index.html>`_ (@HajimeKawahara, co-maintainer)
