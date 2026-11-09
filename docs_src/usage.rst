A binary light curve
====================

Let's calculate a microlensing event produced by a star and a low-mass
companion. We will choose its event parameters, evaluate the magnification
at a set of observation times, and plot the result. The same steps apply to
a stellar binary; only the lens parameters change.

Setting up the event
--------------------

We use ``jax.numpy`` for arrays and Matplotlib for the plot. Enable double
precision at the start: the lens calculations need more precision than
JAX's default 32-bit arithmetic.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import microjax as mj

Create a function for CPU calculations. It takes event
parameters and observation times, and returns the magnification:

.. code-block:: python

   times = jnp.linspace(-5.0, 10.0, 500)
   lightcurve = mj.make_binary_lightcurve(backend="cpu")

The source follows a straight-line trajectory with parameters ``t0``, ``tE``,
and ``u0``, relative to the binary centre of mass. The angle ``alpha`` is
measured from the binary axis in radians. Here, all times are in days:

.. code-block:: python

   # t0, tE, u0, q, s, alpha, rho
   parameters = jnp.array([
       0.0, 10.0, 0.1, 0.001, 1.0, jnp.deg2rad(50.0), 0.005,
   ])

The companion has mass ratio ``q=0.001`` relative to the primary and
separation ``s=1`` Einstein radius. The source radius is ``rho=0.005`` in
the same units, with linear limb darkening ``u1=0.5``. Setting ``u1=0``
gives a uniform disk.

The trajectory passes 0.1 Einstein radii from the binary's centre of mass
at day zero. The light-curve peak need not occur at this time.

Calculating the magnification
-----------------------------

.. code-block:: python

   A = lightcurve(parameters, times, u1=0.5)

``A`` contains one magnification for each of the 500 observation times.
You can call ``lightcurve`` again with different parameters or times;
:doc:`working_with_jax` explains how to compile it for repeated use or
select the GPU implementation.

.. code-block:: python

   fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
   ax.plot(times, A, color="C0")
   ax.set(xlabel="Time (days)", ylabel="Magnification")
   plt.show()

.. figure:: _static/binary_lightcurve.png
   :alt: Binary magnification from day minus five to day ten, with a short dip and sharp rise before the broad peak
   :width: 100%

   The light curve from the code above.

The broad brightening is interrupted by a short feature before the peak.
This is where the source passes near the binary's caustic. Increasing the
source radius smooths the feature; try changing ``rho``, the last entry
of the parameter array:

.. code-block:: python

   A_larger_source = lightcurve(parameters.at[6].set(0.01), times, u1=0.5)

``parameters.at[6].set(0.01)`` makes a copy with the new radius; the original
parameter array is unchanged.

Magnification to observed flux
------------------------------

microJAX returns the magnification. To compare it with photometry, combine
it with the source flux :math:`F_s` and blend flux :math:`F_b`:

.. math::

   F(t) = F_s A(t) + F_b.

The unlensed baseline is :math:`F_s+F_b`. This lets you use the same
magnification curve with different flux scales for different data sets.
The :doc:`flux-fitting functions <api/likelihood>` can solve for these two
fluxes given a magnification curve and observed photometry.

Next, :doc:`autodiff` shows how to calculate derivatives of this event.
For a third lens, continue to the :ref:`triple example <triple-example>`.
If you already have source positions, you can pass them directly to the
:ref:`position-based functions <source-position-api>`.
