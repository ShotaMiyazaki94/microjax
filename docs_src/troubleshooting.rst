Troubleshooting
===============

The most common issues reported by new users are summarised here together with
suggested fixes.

JAX cannot see my GPU
---------------------

- Ensure you installed a CUDA/ROCm build of ``jaxlib`` that matches your driver
  version.  Follow the `official installation matrix
  <https://jax.readthedocs.io/en/latest/installation.html>`_.
- Double-check the environment variables ``XLA_PYTHON_CLIENT_PREALLOCATE`` and
  ``JAX_PLATFORMS``; temporarily set ``JAX_PLATFORMS=cuda`` to force GPU usage.
- On multi-user systems, confirm that you have access to the GPU (``nvidia-smi``
  or ROCm equivalents).

mag_binary is slow or runs out of memory
----------------------------------------

- Finite-source calculations run on a CPU but are intended primarily for a
  GPU. A CPU run can be substantially slower.
- The first call includes JAX compilation. Measure later calls only after
  waiting for the first result with ``block_until_ready()``.
- Process a very long light curve in several user-level batches if the complete
  output and its derivatives do not fit in device memory.
- ``n_limb`` controls how finely the source circumference is followed. Reducing
  it can lower cost, but it can also miss rapidly changing image geometry.
  Validate against the default before using a smaller value.

Gradient computations stall
---------------------------

- Confirm that ``jax_enable_x64`` is turned on; implicit differentiation through
  the polynomial solver is numerically sensitive in single precision.
- Use ``jax.jit`` to compile the forward pass before taking gradients; this
  shortens trace lengths and avoids repeated recompilations.

Import errors for optional dependencies
---------------------------------------

``microJAX`` only depends on JAX and NumPy at runtime, but some examples pull in
``matplotlib`` or ``seaborn``.  Install the plotting stack you need manually—for
example ``python -m pip install matplotlib seaborn``—before running the demo
scripts.

Still stuck?
------------

Open an issue on GitHub with the following information:

- microJAX version (``python -c "import microjax; print(microjax.__version__)"``)
- JAX/JAXLIB versions and platform (CPU, CUDA, ROCm)
- A minimal code snippet reproducing the issue

We are happy to help debug problems and improve the documentation.
