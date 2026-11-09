A binary light curve
====================

Let's calculate a microlensing event produced by a star and a low-mass
companion. We will choose its event parameters, evaluate the magnification
at a set of observation times, and plot the result. The same steps apply to
a stellar binary; only the lens parameters change.

Setting up the event
--------------------

We use ``jax.numpy`` for arrays and Matplotlib for the plot. Enable double
precision at the start: resolving the small images near a caustic needs more
precision than JAX's default 32-bit arithmetic.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import microjax as mj

Choose the observation times and the CPU integrator once. This gives us a
function that takes the event parameters and returns the magnification:

.. code-block:: python

   times = jnp.linspace(-5.0, 10.0, 500)
   lightcurve = mj.make_binary_lightcurve(times, backend="cpu")

The source follows a straight line. Its time of closest approach is ``t0``,
its Einstein crossing time is ``tE``, and its impact parameter is ``u0``.
The trajectory angle ``alpha`` is measured from the binary axis in radians.
Here, all times are in days:

.. code-block:: python

   parameters = dict(
       t0=0.0, tE=10.0, u0=0.1, alpha=jnp.deg2rad(50.0),
       q=0.001, s=1.0, rho=0.005, u1=0.5,
   )

The companion has mass ratio ``q=0.001`` relative to the primary and
separation ``s=1`` Einstein radius. The source has radius ``rho=0.005`` and
a linear limb-darkening coefficient ``u1=0.5``, so its edge is darker than
its centre. Setting ``u1=0`` gives a uniform disk.

The trajectory passes 0.1 Einstein radii from the binary's centre of mass
at day zero. This defines the geometry, rather than the time of peak
magnification, which also depends on the lens.

Calculating the magnification
-----------------------------

.. code-block:: python

   A = lightcurve(parameters)

``A`` contains one magnification for each of the 500 observation times.
You can call ``lightcurve`` again with different parameter values;
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
This is where the source passes near the binary's caustic, a curve along
which the magnification of a point source diverges. Giving the source a
finite radius smooths the feature. Try increasing ``rho`` to see how it
changes, or changing ``q`` to explore a more massive companion:

.. code-block:: python

   A_larger_source = lightcurve({**parameters, "rho": 0.01})

Magnification to observed flux
------------------------------

microJAX returns the magnification. To compare it with photometry, combine
it with the source flux :math:`F_s` and any constant blended light
:math:`F_b`:

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
