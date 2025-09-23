Getting Started
===============

Use this section to set up a working ``microJAX`` environment and ensure the
package can talk to your available hardware.

Requirements
------------

- Python 3.9 or newer
- ``jax`` / ``jaxlib`` matching your accelerator (CPU-only or CUDA/ROCm build)
- ``numpy`` and ``scipy`` (installed automatically via ``pip``)

Installation
------------

From PyPI (recommended)::

   pip install microjaxx

Notes:

- Import name is ``microjax`` but the package on PyPI is ``microjaxx``.
- Upgrade to the latest release with ``pip install --upgrade microjaxx``.

Development install::

   git clone https://github.com/ShotaMiyazaki94/microjax.git
   cd microjax
   pip install -e ".[dev]"

GPU support depends on your system. Follow the
`official JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
to install the correct ``jaxlib`` wheel for CUDA or ROCm.

First checks
------------

After installation, verify that JAX detects the expected devices and that core
modules import correctly::

   python - <<'PY'
   import jax
   import microjax

   print("JAX devices:", jax.devices())
   print("microJAX version:", microjax.__version__)
   PY

For numerical robustness we recommend enabling 64-bit precision in JAX::

   import jax
   jax.config.update("jax_enable_x64", True)

Next steps
----------

- Continue with :doc:`usage` for worked examples and snippets.
- Review :doc:`development` if you plan to contribute or run the test suite.
