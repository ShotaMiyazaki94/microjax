Differentiating a light curve
=============================

A derivative tells us how the model changes when we adjust a parameter.
For example, the source-radius derivative shows which parts of the light
curve are sensitive to the star's size. Here we calculate derivatives for
the parameters of the :doc:`binary event <usage>`.

Writing a model function
------------------------

The light-curve function accepts a parameter dictionary, which JAX can
differentiate directly. The observation times and calculation settings stay
fixed while the event parameters change:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   import microjax as mj

   times = jnp.linspace(-5.0, 10.0, 500)
   lightcurve = mj.make_binary_lightcurve(times, backend="cpu")
   parameters = dict(
       t0=0.0, tE=10.0, u0=0.1, alpha=jnp.deg2rad(50.0),
       q=0.001, s=1.0, rho=0.005, u1=0.5,
   )

The parameter names carry through to the derivatives, so there is no need
to remember a column order.

Calculating the Jacobian
------------------------

``jax.jacfwd`` constructs a function that evaluates the Jacobian. For this
dictionary input, it returns a dictionary of arrays: ``jacobian["rho"]``
contains the source-radius derivative at every observation time.
Wrapping it in ``jax.jit`` compiles the calculation for repeated use:

.. code-block:: python

   derivative = jax.jit(jax.jacfwd(lightcurve))
   jacobian = derivative(parameters)
   print(jacobian["rho"].shape)

.. code-block:: text

   (500,)

The first call includes compilation. Subsequent calls with the same
parameter keys and new values reuse it. All values in this dictionary are
floating point; the derivatives include the limb-darkening coefficient
``u1``. This forward-mode calculation works for both CPU and GPU models;
use ``jacfwd`` for the CPU integrator,
whose adaptive loops do not support reverse-mode ``grad`` or ``jacrev``.

Looking at one derivative
-------------------------

Let's plot :math:`\partial A/\partial\rho` below the magnification so we
can see where the source size matters:

.. code-block:: python

   A = jax.jit(lightcurve)(parameters)

   fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                            layout="constrained")
   axes[0].plot(times, A, color="C0")
   axes[0].set_ylabel("Magnification")
   axes[1].axhline(0, color="0.8", linewidth=0.8)
   axes[1].plot(times, jacobian["rho"], color="C1")
   axes[1].set(xlabel="Time (days)", ylabel=r"$\partial A / \partial\rho$")
   plt.show()

.. figure:: _static/binary_radius_derivative.png
   :alt: The binary light curve and its source-radius derivative, which is largest around the short caustic feature
   :width: 100%

   Sensitivity to source radius is concentrated around the caustic feature.

Where the derivative is positive, a small increase in ``rho`` raises the
magnification; where it is negative, the magnification falls. The nearly
flat derivative far from the feature shows why those observations alone
provide little information about the source radius.

For a flux model :math:`F=F_s A+F_b`, the derivative with respect to an event
parameter is simply :math:`F_s\,\partial A/\partial p`. These are the
quantities a least-squares fit uses to update the event parameters.

Values and derivatives in one call
----------------------------------

A fit usually needs both the light curve and its Jacobian. Return the curve
as auxiliary data to get them from the same calculation:

.. code-block:: python

   def with_value(p):
       A = lightcurve(p)
       return A, A

   value_and_jacobian = jax.jit(jax.jacfwd(with_value, has_aux=True))
   jacobian, A = value_and_jacobian(parameters)

For a triple lens, the same pattern applies with three additional parameters
and ``make_triple_lightcurve``. The :doc:`examples/index` keeps ``u1`` fixed
and plots the seven binary and ten triple event derivatives, with complete
scripts for both CPU and GPU.
