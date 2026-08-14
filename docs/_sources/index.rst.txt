microJAX
========

`microJAX <https://github.com/ShotaMiyazaki94/microjax>`_ is a differentiable
microlensing library built with JAX. It provides point-source calculations for
one to three lenses, finite-source calculations for binary and triple lenses,
and trajectory utilities.

The finite-source API includes a GPU-oriented accelerator backend and a
separate CPU backend for binary lenses. Both support uniform and linear
limb-darkened circular sources. Use :doc:`getting_started` to install the
package and verify the JAX environment, then continue with :doc:`usage` for
public API examples.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   getting_started
   usage
   cpu_backend
   performance
   caveats
   citing

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules

Choosing a guide
----------------

- :doc:`cpu_backend` defines the binary-lens CPU interface, supported
  differentiation, and numerical contract.
- :doc:`performance` explains accelerator scheduler settings and sound timing
  practice.
- :doc:`caveats` collects microJAX-specific limitations that can affect a
  scientific result.
- :doc:`citing` gives citations and the reproducibility information to report.

The methods-paper implementation is archived as ``v0.1.1``. The ``0.2``
series is a substantial solver redesign, so reproducible work should record
the exact microJAX release or Git commit.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
