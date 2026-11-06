Inverse-Ray Lightcurve
======================

Configuration
-------------

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

Returned with ``return_info=True``; contains only magnification and status. See
:doc:`../cpu_backend` for the CPU backend contract and debugging guidance.
