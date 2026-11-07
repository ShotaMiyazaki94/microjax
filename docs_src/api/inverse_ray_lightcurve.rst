Inverse-Ray Lightcurve
======================

Configuration
-------------

These configuration objects tune the default accelerator implementation.
``mag_binary`` and ``mag_triple`` select the dedicated CPU integrator with
``backend="cpu"``; CPU support and quadrature settings are internal.

.. autoclass:: microjax.inverse_ray.BinaryMagConfig
   :members:

.. autoclass:: microjax.inverse_ray.TripleMagConfig
   :members:

Light-curve functions
---------------------

.. automodule:: microjax.inverse_ray.lightcurve
   :members:
   :show-inheritance:

CPU result
----------

.. autoclass:: microjax.inverse_ray.cpu.CpuMagnificationResult
   :members:

Returned by either lens model with ``backend="cpu", return_info=True``;
contains only magnification and status. See
:doc:`../cpu_backend` for the CPU backend contract and debugging guidance.
