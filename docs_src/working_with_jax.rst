Working with JAX
================

You can use microJAX much like an array-based NumPy model. JAX adds two
particularly useful tools: ``jit`` compiles a model for repeated evaluation,
and ``jacfwd`` calculates its parameter derivatives. The tutorials introduce
both; this page collects the details that matter when using them in a fit.

Precision and compilation
-------------------------

Enable double precision before creating arrays:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

Use ``jax.numpy`` inside your model so JAX can trace its calculations.
Ordinary NumPy is fine for loading data or plotting results outside the model.
If you are new to JAX, its
`introduction to automatic differentiation <https://docs.jax.dev/en/latest/automatic-differentiation.html>`_
is a useful companion to the :doc:`autodiff` tutorial.

Create a compiled function once and reuse it. For example, using
``lightcurve`` and ``parameters`` from that tutorial:

.. code-block:: python

   model = jax.jit(lightcurve)
   A = model(parameters, times, u1=0.5)

The first call compiles the function. Changing parameter or time values
reuses the compiled model when the array shapes and dtypes stay the same.
Changing the number of observation times triggers a new compilation.

JAX may return before the calculation has finished. When measuring runtime,
wait for each result and keep compilation outside the timed section:

.. code-block:: python

   import time

   A.block_until_ready()
   start = time.perf_counter()
   A = model(parameters, times, u1=0.5).block_until_ready()
   print(f"{1e3 * (time.perf_counter() - start):.2f} ms")

Repeat the timed call to compare typical runtimes on your machine.
The :doc:`benchmarks` use this approach for magnifications and Jacobians.

CPU and GPU
-----------

Select the implementation when creating the light-curve function:

.. code-block:: python

   import microjax as mj

   lightcurve = mj.make_binary_lightcurve(backend="cpu")

Use ``backend="gpu"`` for the GPU implementation; this is the
default. The returned function takes event parameters and times. To tune
GPU batching, pass a configuration when you create it:

.. code-block:: python

   lightcurve = mj.make_binary_lightcurve(
       backend="gpu",
       config=mj.BinaryMagConfig(source_tile_size=256),
   )

JAX chooses the device separately from this setting. On a machine with a
GPU, you can force a CPU example to run on the CPU with:

.. code-block:: sh

   JAX_PLATFORMS=cpu python example/cpu/lightcurve.py

To use a GPU, install GPU-enabled JAX and select the GPU backend.
The :doc:`examples/index` includes ready-to-run GPU scripts. Start with the
default settings; if memory or throughput becomes an issue, adjust the
:ref:`GPU batch sizes <gpu-configuration>` or split the
observation times into fixed-size batches.

Numerical accuracy
------------------

Both implementations use a multipole approximation away from caustics and
integrate the lensed images where a full finite-source calculation is
needed. The :doc:`benchmark maps <benchmarks>` show how measured accuracy
and speed vary with lens parameters and magnification.

Near a caustic, derivatives can be more sensitive to numerical resolution
than magnifications. For a fit dominated by a caustic crossing, compare a
few model values and derivatives with an independent calculation at your
event's parameters. The solver uses fixed work limits, rather than a
requested relative-error tolerance. CPU status flags and GPU
resolution settings are described in the
:doc:`finite-source reference <api/magnification>`.
