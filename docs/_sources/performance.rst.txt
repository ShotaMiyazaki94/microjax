Accelerator Performance Tuning
==============================

This page covers the accelerator backend configured by
``BinaryMagConfig`` and ``TripleMagConfig``. The binary-lens CPU backend uses
fixed internal scheduling; see :doc:`cpu_backend` for CPU guidance.

The accelerator solver first applies a fast approximation over the complete
trajectory. Only rejected source positions enter the full finite-source
calculation. Their number, rather than the total trajectory length, largely
determines the best scheduler configuration.

Static scheduler settings
-------------------------

``source_tile_size``
   Maximum number of full-solve positions evaluated in one outer batch. A
   partially occupied final tile is evaluated in full.

``radial_chunk_size``
   Number of radial image regions evaluated in one inner batch.

These settings change static JAX shapes and produce separately compiled
executables. Choose them before a fit rather than changing them between
likelihood evaluations. They control batching, not numerical accuracy.

The defaults were selected from synchronized, compilation-excluded
measurements on an NVIDIA A100 using the repository's binary and triple-lens
trajectories. They are useful starting points, not hardware-independent
optima. In particular, the best source tile depends on the number of full
solves and padding in its final tile.

Benchmark the analysis workload
-------------------------------

Benchmark the complete forward model with the same trajectory length, lens
type, source profile, array shapes, and JAX transformation used by the
analysis. Warm up the exact configuration and synchronize before stopping the
timer:

.. code-block:: python

   import time
   import jax

   model = jax.jit(lambda: mag_binary(w, rho, s=s, q=q, config=config))
   model().block_until_ready()  # trace and compile

   start = time.perf_counter()
   result = model()
   result.block_until_ready()
   elapsed = time.perf_counter() - start

When tuning ``source_tile_size``, compare values near the expected number of
full solves and account for the padding in the final tile. For larger
trajectories, compare more than one tile size: lens type and the number of
executed tiles can change the winner. Report the selected configuration and
user-level batch size with benchmark results.

Memory and accuracy
-------------------

Larger tiles and chunks can increase peak device memory. If a complete light
curve and its derivatives do not fit, split the trajectory into stable
user-level batches and benchmark that same batching scheme.

``n_limb`` is different from the scheduler settings: it controls how finely
the accelerator backend follows the source circumference. Reducing it may
lower cost but can miss rapidly changing image geometry, so changes require
independent accuracy validation. It does not configure the CPU backend.

The complete A100 sweep, including exact timings, memory measurements, and
workload definitions, is retained as a development record rather than a
portable performance promise.
