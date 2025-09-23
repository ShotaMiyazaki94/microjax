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

- Reduce ``r_resolution``/``th_resolution`` and the limb-darkening table size
  ``Nlimb``.  Start with 512×512 and scale up only if accuracy requires it.
- Increase ``chunk_size`` so that more samples are processed per device launch.
- Decrease ``MAX_FULL_CALLS`` if the adaptive trigger upgrades too many points.

Gradient computations stall
---------------------------

- Confirm that ``jax_enable_x64`` is turned on; implicit differentiation through
  the polynomial solver is numerically sensitive in single precision.
- Use ``jax.jit`` to compile the forward pass before taking gradients; this
  shortens trace lengths and avoids repeated recompilations.

Import errors for optional dependencies
---------------------------------------

``microJAX`` only depends on JAX and NumPy at runtime, but some examples pull in
``matplotlib`` or ``seaborn``.  Install extras with ``pip install microjaxx[plot]``
(or install the packages manually) if you intend to run the demo scripts.

Still stuck?
------------

Open an issue on GitHub with the following information:

- microJAX version (``python -c "import microjax; print(microjax.__version__)"``)
- JAX/JAXLIB versions and platform (CPU, CUDA, ROCm)
- A minimal code snippet reproducing the issue

We are happy to help debug problems and improve the documentation.
