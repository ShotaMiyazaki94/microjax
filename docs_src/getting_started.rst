Getting Started
===============

Use this guide to set up a microJAX environment, verify that JAX can see your
accelerator, and run a sanity check.

Prerequisites
-------------

- Python 3.9 or newer (3.10/3.11 recommended for the latest JAX wheels)
- ``jax``/``jaxlib`` compiled for your target platform (CPU, CUDA, or ROCm)
- Optionally: ``matplotlib`` and ``seaborn`` for the plotting utilities
- A recent ``pip`` (``python -m pip install --upgrade pip``)

Installation
------------

Stable releases (PyPI)::

   python -m pip install --upgrade microjaxx

Development version (from source)::

   git clone https://github.com/ShotaMiyazaki94/microjax.git
   cd microjax
   python -m pip install -e ".[dev]"

The import name is ``microjax`` even though the wheel on PyPI is published as
``microjaxx``.

GPU-aware JAX wheels must be installed separately; follow the `official JAX
installation matrix <https://jax.readthedocs.io/en/latest/installation.html>`_
and pick the CUDA/ROCm wheel that matches your drivers.

Verify the environment
----------------------

Check that JAX detects your devices and that microJAX imports cleanly::

   python - <<'PY'
   import jax
   import microjax

   print("microJAX", microjax.__version__)
   print("JAX devices", jax.devices())
   PY

Enable 64-bit mode for improved numerical stability::

   import jax
   jax.config.update("jax_enable_x64", True)

Smoke test: point-source magnification::

   python - <<'PY'
   import jax
   import jax.numpy as jnp
   from microjax.point_source import mag_point_source

   jax.config.update("jax_enable_x64", True)

   w = jnp.linspace(-0.3, 0.3, 5) + 1j * 0.1
   mu = mag_point_source(w, nlenses=2, s=1.0, q=1e-3)
   print(mu)
   PY

Up next
-------

- :doc:`usage` walks through binary and triple lens examples.
- :doc:`concepts` explains how the modules fit together.
- :doc:`troubleshooting` lists common pitfalls and quick fixes.
