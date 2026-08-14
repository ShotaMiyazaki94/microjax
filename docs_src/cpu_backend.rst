CPU Binary-Lens Backend
=======================

``mag_binary(..., backend="cpu")`` selects microJAX's CPU-oriented
finite-source binary-lens solver. It is a separate implementation from the
default accelerator backend, not an automatic fallback chosen from the
available JAX device.

The CPU backend supports binary lenses with uniform or linear limb-darkened
circular sources. Finite-source triple-lens calculations use the accelerator
backend.

Basic use
---------

Enable double precision before constructing arrays or compiling functions:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from microjax.inverse_ray import mag_binary

   w = jnp.asarray([0.10 + 0.20j, 0.60 - 0.20j])
   magnification = mag_binary(
       w,
       1.0e-2,
       s=1.0,
       q=0.3,
       u1=0.0,
       backend="cpu",
   )

The CPU solver has fixed internal scheduling and quadrature settings.
``BinaryMagConfig`` and ``n_limb`` configure the accelerator backend and do
not tune this path. ``"cpu-one-shot"`` remains an alias of ``"cpu"`` for
compatibility. The older ``"cpu-adaptive"`` backend is retained for research
comparisons and should not be used as a silent retry in production models.

Numerical contract
------------------

The CPU backend uses bounded work. It applies a multipole approximation where
its internal gate accepts it and otherwise performs one full finite-source
solve. It does not increase integration order or retry until a requested
tolerance is met.

A finite result is therefore a numerical estimate, not a certified error
bound. Detected geometry, capacity, root, or non-finite failures are returned
as ``NaN``. Before an inference run:

- validate values over the intended ``(q, s, rho, w, u1)`` region against an
  independent implementation;
- validate derivatives separately from values;
- retain the full configuration for every microJAX ``NaN``;
- record failures or non-convergence from the reference solver separately.

For investigation of an individual rejection, ``return_info=True`` exposes a
best-effort result and internal diagnostic state. Those fields are debugging
details rather than a stable scientific interface; routine modeling should
use the ordinary magnification result.

Forward-mode differentiation
----------------------------

``jax.jvp`` and ``jax.jacfwd`` are supported. Reverse-mode differentiation
through the CPU solver's data-dependent loops is not part of the public API.

.. code-block:: python

   parameters = jnp.asarray([0.10, 0.20, 1.0e-2, 1.0, 0.3])

   def model(values):
       source = jnp.asarray([values[0] + 1j * values[1]])
       return mag_binary(
           source,
           values[2],
           s=values[3],
           q=values[4],
           backend="cpu",
       )[0]

   value = jax.jit(model)(parameters)
   jacobian = jax.jit(jax.jacfwd(model))(parameters)

Caustic crossings and internal route changes can make the numerical graph
piecewise. A finite derivative does not by itself establish accuracy or
smoothness in a surrounding parameter region.

Timing
------

The first call includes tracing and compilation. Warm up the same array shape
and source profile, then synchronize each timed result with
``block_until_ready()``. Compare complete trajectories: runtime depends on how
many positions require a full finite-source calculation. See
:doc:`performance` for a timing example; its configuration controls apply
only to the accelerator backend.

Examples
--------

The repository includes two CPU workflows:

- `CPU/VBM value comparison
  <https://github.com/ShotaMiyazaki94/microjax/tree/main/example/cpu/compare-binary-vbml>`_
- `CPU forward Jacobian
  <https://github.com/ShotaMiyazaki94/microjax/tree/main/example/cpu/jacobian-binary>`_
