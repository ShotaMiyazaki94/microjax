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

- ``mag_binary`` also works on CPU but is very slow.
- Trim the inverse-ray grid via ``r_resolution`` / ``th_resolution`` when you
  do not need the default 1000×1000 sampling; smaller grids cut both runtime and
  memory pressure.  Increase them only when accuracy demands it.
- Adjust ``chunk_size`` to fit your device.  Lower values avoid out-of-memory
  crashes; raise it gradually if the GPU remains underutilised.
- Use ``MAX_FULL_CALLS`` to cap how many samples fall back to the full
  image-centred ray shooting routine.  Lowering it keeps runtimes bounded, but
  expect a trade-off in accuracy if many points revert to the hexadecapole
  approximation.

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
