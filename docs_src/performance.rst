Accelerator Performance Tuning
==============================

This page describes the accelerator backend controlled by
``BinaryMagConfig`` and ``TripleMagConfig``. The binary-lens CPU backend has
fixed internal scheduling and does not use these settings; see
:doc:`cpu_backend` for CPU timing and warm-up guidance.

The finite-source light-curve functions first evaluate a fast approximation
over the complete trajectory. Only source positions rejected by that
approximation enter the full image-boundary calculation. In this guide,
*full-solve count* means the number of rejected positions, not the total
trajectory length.

``BinaryMagConfig`` and ``TripleMagConfig`` expose two static scheduler
settings:

``source_tile_size``
   Maximum number of full-solve source positions evaluated by one outer
   vectorized batch. A partially occupied final tile is evaluated in full.

``radial_chunk_size``
   Number of radial image regions evaluated by one inner vectorized batch.
   The fixed-capacity region buffer contains 64 entries, so a value of 64
   evaluates the complete buffer together.

Both settings change static JAX shapes. A new value therefore produces a
separately compiled executable. Select a configuration before a fit rather
than changing it between likelihood evaluations.

Recommended A100 configurations
-------------------------------

The following table reports recommendations at the full-solve counts included
in the sweep. It prioritizes execution speed when accelerator memory is
available. ``n_limb=500`` is omitted from the examples for clarity.

.. list-table::
   :header-rows: 1
   :widths: 22 18 22 38

   * - Measured full-solve count
     - Lens
     - Recommended scheduler
     - Measured interpretation
   * - 1 or 8
     - Binary or triple
     - ``8 / 64``
     - One 8-position outer tile is evaluated. At count 1, seven padded
       positions are evaluated.
   * - 32
     - Binary or triple
     - ``32 / 64``
     - One 32-position outer tile is evaluated without padding.
   * - 64
     - Binary or triple
     - ``64 / 64``
     - One 64-position outer tile is evaluated without padding.
   * - 100
     - Binary
     - ``100 / 64``
     - One 100-position outer tile is evaluated without padding.
   * - 100
     - Triple
     - ``100 / 8``
     - At this outer width, radial chunks 8, 32, and 64 were within 0.3 ms for
       uniform and limb-darkened sources. The 8-region setting had the lowest
       combined time and is the triple default.

Here ``tile / radial`` abbreviates
``source_tile_size / radial_chunk_size``.

For an expected full-solve count up to 100 that was not measured directly,
use the smallest tile in ``8, 16, 32, 64, 100`` that is not smaller than the
count, then benchmark the two adjacent tile sizes. For example, a count of 23
starts with tile 32 and compares it with tile 16. This is a deterministic
starting rule, not a claim that an unmeasured count has a known optimum.

For counts greater than 100, no single threshold rule matched every
measurement. The relevant quantities are
``ceil(full_solve_count / source_tile_size)`` executed tiles and the padding
in the final tile. The measured results were:

.. list-table::
   :header-rows: 1
   :widths: 22 18 22 38

   * - Full-solve count
     - Lens
     - Fastest measured scheduler
     - Compared source tiles
   * - 256
     - Binary
     - ``100 / 64``
     - 8, 16, 32, 64, and 100
   * - 256
     - Triple
     - ``64 / 64``
     - 8, 16, 32, 64, and 100
   * - 544
     - Binary benchmark
     - ``100 / 64``
     - 8, 16, 32, 64, and 100
   * - 888
     - Triple example
     - ``100 / 8``
     - 8, 16, 32, 64, and 100

The 544-position binary benchmark used ``s=0.85``, ``q=0.03``, and
``rho=5e-3`` on the 1000-position VBBL-comparison trajectory. The
888-position triple benchmark used the 1000-position triple-Jacobian
trajectory. These are the workloads used to choose the defaults. For example:

.. code-block:: python

   from microjax.inverse_ray import BinaryMagConfig, TripleMagConfig

   binary_544_full_solves = BinaryMagConfig(
       n_limb=500,
       source_tile_size=100,
       radial_chunk_size=64,
   )
   triple_888_full_solves = TripleMagConfig(
       n_limb=500,
       source_tile_size=100,
       radial_chunk_size=8,
   )
   binary_1_to_8_full_solves = BinaryMagConfig(
       n_limb=500,
       source_tile_size=8,
       radial_chunk_size=64,
   )

Measurement scope
-----------------

The recommendations come from an NVIDIA A100-PCIE-40GB sweep with JAX 0.10.2,
double precision, and ``n_limb=500``. Timings excluded compilation and used
synchronized runs. Under the measured parameters above, the
default-selection trajectories sent 544 of 1000 binary positions and 888 of
1000 triple positions to the full calculation. A changed source radius or
prefilter changes these counts and is a different benchmark workload.

With JAX device-memory preallocation disabled, changing the radial chunk from
8 to 64 raised the measured 100-position-tile primal peak from 160 MiB to
256 MiB for binary lenses and from 144 MiB to 256 MiB for triple lenses.
Scheduler settings produced identical magnification arrays in the reference
binary and triple trajectories.

Benchmark your actual trajectory
--------------------------------

The best outer tile depends on how many positions enter the full calculation
and on padding in the final tile. Benchmark the complete forward model,
including the same trajectory length, lens type, limb-darkening law, and JAX
transformation used by the analysis. Always warm up first and synchronize
before stopping the timer:

.. code-block:: python

   import time
   import jax

   model = jax.jit(lambda: mag_binary(w, rho, s=s, q=q, config=config))
   model().block_until_ready()  # compile

   start = time.perf_counter()
   result = model()
   result.block_until_ready()
   elapsed = time.perf_counter() - start

Scheduler settings change batching only; they are not accuracy controls.
