Finite-source magnification
===========================

Evaluate binary and triple circular sources with linear limb darkening.
Source positions are a one-dimensional complex array in Einstein-radius
units; each function returns one magnification per position. See
:doc:`../usage` for lens coordinates and :doc:`../autodiff` for Jacobians.

Light-curve functions
---------------------

.. autofunction:: microjax.inverse_ray.mag_binary

.. autofunction:: microjax.inverse_ray.mag_triple

Configuration
-------------

These configuration objects tune the default accelerator implementation.
``mag_binary`` and ``mag_triple`` select the dedicated CPU integrator with
``backend="cpu"``; CPU support and quadrature settings are internal.

.. autoclass:: microjax.inverse_ray.BinaryMagConfig
   :members:

.. autoclass:: microjax.inverse_ray.TripleMagConfig
   :members:

CPU result
----------

.. autoclass:: microjax.inverse_ray.cpu.CpuMagnificationResult
   :members:

Returned by either lens model with ``backend="cpu", return_info=True``;
contains only magnification and status. See
:doc:`../cpu_backend` for status interpretation.
