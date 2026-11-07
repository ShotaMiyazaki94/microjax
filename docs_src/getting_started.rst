Installation
============

Install microJAX in your Python environment:

.. code-block:: sh

   python -m pip install microjaxx

The distribution is named ``microjaxx``; import it as ``microjax``.
Package requirements are Python 3.9+, JAX 0.4+, and NumPy 1.26+.
For GPU support, follow the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

To use the latest development version:

.. code-block:: sh

   python -m pip install "microjaxx @ git+https://github.com/ShotaMiyazaki94/microjax.git@develop"

The examples use Matplotlib:

.. code-block:: sh

   python -m pip install matplotlib

First calculation
-----------------

Enable float64 before creating arrays, then calculate a binary light curve:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from microjax.inverse_ray import mag_binary

   w = jnp.linspace(-0.3, 0.3, 5) + 0.1j
   A = mag_binary(w, 0.01, s=1., q=0.01, u1=0.5, backend="cpu")
   print(A.shape)  # (5,)

Continue with :doc:`usage`, :doc:`autodiff`, or the
:doc:`examples/index`.

Source checkout
---------------

Clone the development branch to run or edit the examples:

.. code-block:: sh

   git clone --branch develop https://github.com/ShotaMiyazaki94/microjax.git
   cd microjax
   python -m pip install -e ".[dev]" matplotlib
   python example/cpu/lightcurve.py

The `contribution guide <https://github.com/ShotaMiyazaki94/microjax/blob/develop/CONTRIBUTING.md>`_
has the test and documentation-build commands.
