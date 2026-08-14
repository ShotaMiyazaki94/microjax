CPU Binary-Lens Backend
=======================

microJAX provides a CPU-oriented finite-source solver for binary lenses through
the public :func:`microjax.inverse_ray.mag_binary` function. It is a separate
execution path from the accelerator boundary integrator: it uses a fixed
one-shot Cartesian or polar image-plane calculation and is designed for
forward-mode differentiation on CPUs.

The CPU backend is currently available for binary lenses only. Triple-lens
finite-source calculations continue to use the accelerator-oriented backend.

Quick start
-----------

Enable double precision before creating arrays or compiling functions:

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

The ordinary API returns the magnification array directly and replaces detected
structural failures with ``NaN``. Check for non-finite values before passing a
light curve to downstream inference code.

Backend choices
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - ``backend``
     - Intended use
     - Behaviour
   * - ``"accelerator"``
     - GPU-oriented binary calculation
     - Fixed-work boundary integrator configured by ``BinaryMagConfig``.
       ``"gpu"`` is an alias.
   * - ``"cpu"``
     - Production CPU calculation
     - Multipole prefilter followed, when necessary, by one fixed one-shot
       full solve. No retry or order escalation is performed.
   * - ``"cpu-one-shot"``
     - Compatibility
     - Exact alias of ``"cpu"``.
   * - ``"cpu-adaptive"``
     - Research and compatibility checks
     - Older coverage-oriented CPU scheduler with adaptive chart logic. It is
       slower and is not the default CPU path.

``BinaryMagConfig`` controls static shapes in the accelerator backend. It does
not tune the production CPU path. In particular, the public CPU backend has no
``n_limb`` or accuracy-tolerance argument: it uses a fixed 64-point source-limb
support trace and a calibrated internal multipole gate.

Execution model
---------------

For each source position the CPU scheduler performs the following operations:

1. Evaluate the finite-source multipole approximation and geometric guards.
2. Return the multipole value when the fixed internal gate accepts it.
3. Otherwise trace the lensed source circumference once with 64 support
   samples.
4. Measure image topology and radial/tangential image motion.
5. Select one Cartesian, source-radial, or angle-first polar chart from that
   state.
6. Evaluate one fixed high-order quadrature and return immediately.

The full-solve graph does not compare a coarse and fine answer. On a detected
root, support, topology, capacity, or non-finite failure it returns ``NaN``
instead of starting a rescue chart, retracing the limb, or increasing the
quadrature order. This fail-closed design keeps the compiled graph bounded.

Uniform sources use root-free Bernstein strip isolation for Cartesian charts.
Linear limb darkening (``u1 > 0``) integrates the normalized brightness weight
over the same image geometry. Nearly annular images use angle-first polar
radial moments.

Advanced diagnostics
--------------------

Routine modeling does not require diagnostic flags. For debugging a rejected
configuration, ``return_info=True`` returns a
:class:`microjax.inverse_ray.cpu.CpuMagnificationResult` containing the
best-effort value and internal routing information. A non-zero ``status`` is
invalid regardless of whether that best-effort value is finite.

Individual status bits and exact tier numbers are implementation diagnostics,
not a stable scientific interface. Do not branch an analysis on them. Full
one-shot solves also report ``estimated_error=NaN`` because this path does not
perform a coarse/fine convergence comparison.

Accuracy contract
-----------------

The CPU backend returns numerical estimates, not guaranteed error bounds. A
finite result means only that the solver did not detect a structural failure;
it does not mean that the relative error is below ``1e-3`` or any other target.

Before using the backend in an inference run:

- validate values over the intended ``(q, s, rho, w, u1)`` region against an
  independent implementation;
- validate derivatives separately from values;
- retain configurations that return ``NaN`` rather than silently discarding
  them;
- record the microJAX Git commit, JAX/JAXLIB versions, platform, and x64 mode.

When running an external validation sweep, retain microJAX misses and reference
solver failures separately. Store the complete lens and source configuration
for each miss so that it can be replayed independently.

Forward-mode differentiation
----------------------------

The production CPU graph supports forward-mode transformations such as
``jax.jvp`` and ``jax.jacfwd``. Reverse-mode differentiation through its
data-dependent sequential loops is not part of the API.

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

Route-selection boundaries are discrete. A finite forward derivative does not
prove that the selected numerical route is accurate or smooth over a larger
neighbourhood.

Compilation and performance
---------------------------

The first call includes JAX tracing and compilation. Warm up the exact array
shape and source profile before timing, then synchronize the result:

.. code-block:: python

   import time

   solve = jax.jit(
       lambda points: mag_binary(
           points,
           1.0e-2,
           s=1.0,
           q=0.3,
           backend="cpu",
       )
   )
   solve(w).block_until_ready()

   start = time.perf_counter()
   solve(w).block_until_ready()
   elapsed = time.perf_counter() - start

Runtime depends strongly on how many positions pass the multipole gate and on
which full-solve charts are selected. Compare warmed end-to-end trajectories,
not isolated unsynchronised calls. The CPU backend can be competitive on
caustic-heavy batches, while smooth trajectories dominated by the multipole
path may favour other implementations.

Examples
--------

The repository includes two CPU-specific workflows:

- `CPU/VBM value comparison
  <https://github.com/ShotaMiyazaki94/microjax/tree/main/example/cpu/compare-binary-vbbl>`_
- `CPU forward Jacobian
  <https://github.com/ShotaMiyazaki94/microjax/tree/main/example/cpu/jacobian-binary>`_

Use the comparison workflow to establish value accuracy for a trajectory and
the Jacobian workflow to exercise the same public backend under
``jax.jacfwd``.
