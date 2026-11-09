Installation
============

The examples in this guide use the development version of microJAX.
Install it together with Matplotlib for plotting:

.. code-block:: sh

   python -m pip install "microjaxx @ git+https://github.com/ShotaMiyazaki94/microjax.git@develop" matplotlib

The package name is ``microjaxx`` on pip and ``microjax`` in Python.
This installation is enough to follow the tutorials on a CPU. To use a
GPU, install the appropriate JAX package for your machine using the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

You are ready to :doc:`calculate your first light curve <usage>`.
The examples introduce the JAX features they use as they go; you do not
need to learn JAX before starting.

Installing from a checkout
--------------------------

A checkout is useful if you want to run the full example scripts or change
the code:

.. code-block:: sh

   git clone --branch develop https://github.com/ShotaMiyazaki94/microjax.git
   cd microjax
   python -m pip install -e . matplotlib

You can then run the :doc:`examples/index` from the repository root.
The `contribution guide <https://github.com/ShotaMiyazaki94/microjax/blob/develop/CONTRIBUTING.md>`_
explains how to run tests and build these pages.

The published release is also available with
``python -m pip install microjaxx``. These pages follow ``develop``, where
new CPU and GPU features are added between releases.
