Getting Started
===============

Use this guide to prepare an environment, confirm that JAX detects your
accelerator, and run a quick smoke test.

Prerequisites
-------------

- Python 3.9 or newer with matching ``jax``/``jaxlib`` wheels for your platform
  (CPU, CUDA, ROCm).  Follow the `official JAX installation matrix
  <https://jax.readthedocs.io/en/latest/installation.html>`_.
- Optional plotting stack: ``matplotlib`` or ``seaborn`` if you plan to run the
  visualization examples.

Installation
------------

Install the latest release from PyPI::

   python -m pip install microjaxx

Or work from source::

   git clone https://github.com/ShotaMiyazaki94/microjax.git
   cd microjax
   python -m pip install -e ".[dev]"

The import name remains ``microjax`` even though the published wheel is
``microjaxx``.

Verify the environment
----------------------

Run the snippet below to confirm that microJAX imports cleanly, JAX can see your
devices, and 64-bit mode is enabled for better numerical stability::

   python - <<'PY'
   import jax
   import jax.numpy as jnp
   import microjax
   from microjax.point_source import mag_point_source

   jax.config.update("jax_enable_x64", True)  # recommended for microlensing

   print("microJAX", microjax.__version__)
   print("Devices", jax.devices())

   w = jnp.linspace(-0.3, 0.3, 5) + 0.1j
   print("Sample magnification", mag_point_source(w, nlenses=2, s=1.0, q=1e-3))
   PY

Up next
-------

- :doc:`usage` walks through binary and triple lens examples.
- :doc:`concepts` explains how the modules fit together.
- :doc:`troubleshooting` lists common pitfalls and quick fixes.
