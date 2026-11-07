Performance
===========

Compile the complete light-curve model once, then reuse it during a fit.
JAX dispatch is asynchronous, so synchronize each result when timing a call.
The :doc:`benchmarks` page gives the reference workloads and reproduction
commands.

Time a light curve
------------------

This example measures a 128-position binary light curve on the CPU. The first
call compiles the model; the reported median covers five subsequent calls.

.. code-block:: python

   import time
   from statistics import median

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from microjax.inverse_ray import mag_binary

   with jax.default_device(jax.devices("cpu")[0]):
       times = jnp.linspace(-5., 10., 128)
       alpha = jnp.deg2rad(50.)
       w = (times / 10. + 0.1j) * jnp.exp(1j * alpha)

       model = jax.jit(lambda q: mag_binary(
           w, 0.005, s=1., q=q, u1=0.5, backend="cpu",
       ))
       jax.block_until_ready(model(0.001))

       seconds = []
       for _ in range(5):
           start = time.perf_counter()
           values = jax.block_until_ready(model(0.001))
           seconds.append(time.perf_counter() - start)
       print(f"Median warm time: {1e3 * median(seconds):.2f} ms")

For a Jacobian, time ``jax.jit(jax.jacfwd(model))`` with its own warmup.
Keep the trajectory length, lens model, source profile, and differentiation
mode the same when comparing implementations. A new array shape or static
configuration triggers a separate compilation.

GPU batching
------------

``BinaryMagConfig`` and ``TripleMagConfig`` control accelerator batching.
Start with their defaults. The solver applies a multipole approximation to
the trajectory and batches the positions that need full finite-source
integration.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Effect
   * - ``source_tile_size``
     - Full-solve positions in each outer batch; the final tile is padded.
   * - ``radial_chunk_size``
     - Radial image regions evaluated together inside each position.
   * - ``n_limb``
     - Sampling of the source circumference, affecting geometric resolution.

For example, change the outer batch size while keeping the other defaults:

.. code-block:: python

   from microjax.inverse_ray import BinaryMagConfig

   config = BinaryMagConfig(source_tile_size=256)
   model_gpu = jax.jit(lambda q: mag_binary(
       w, 0.005, s=1., q=q, u1=0.5, config=config,
   ))

Run this model on the selected JAX accelerator. Compare a few tile sizes on
the event being fitted; the number of full solves and final-tile padding
determine their cost. Larger tiles and chunks use more device memory. If a
light curve and its Jacobian exceed available memory, split the time grid
into fixed-size batches.

Batch sizes control throughput. ``n_limb`` controls resolution; see
:doc:`caveats` for accuracy checks. The CPU backend chooses its internal
schedule automatically and does not use these configuration objects.
