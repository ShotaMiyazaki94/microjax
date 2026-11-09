Differentiating a light curve
=============================

A derivative tells us how the model changes when we adjust a parameter.
For example, the source-radius derivative shows which parts of the light
curve are sensitive to the star's size. Here we calculate derivatives for
all seven parameters of the :doc:`binary event <usage>`.

Writing a model function
------------------------

Put the parameters you want to vary into an array, then write a function
that maps this array to a light curve. The times stay fixed while the
parameters change:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   from microjax.inverse_ray import mag_binary

   times = jnp.linspace(-5.0, 10.0, 500)
   parameters = jnp.array([
       0.0, 10.0, 0.1, 0.001, 1.0, jnp.deg2rad(50.0), 0.005,
   ])

   def lightcurve(p):
       t0, t_e, u0, q, s, alpha, rho = p
       w = ((times - t0) / t_e + 1j * u0) * jnp.exp(1j * alpha)
       return mag_binary(w, rho, s=s, q=q, u1=0.5, backend="cpu")

The parameter order is ``t0, t_e, u0, q, s, alpha, rho``. We keep limb
darkening fixed here; you can include ``u1`` in the array if you want its
derivative too.

Calculating the Jacobian
------------------------

``jax.jacfwd`` constructs a function that evaluates the Jacobian. Each row
corresponds to an observation time and each column to a model parameter.
Wrapping it in ``jax.jit`` compiles the calculation for repeated use:

.. code-block:: python

   derivative = jax.jit(jax.jacfwd(lightcurve))
   jacobian = derivative(parameters)
   print(jacobian.shape)

.. code-block:: text

   (500, 7)

The first call includes compilation. Subsequent calls with another
seven-element parameter array reuse it. This forward-mode calculation
works for both CPU and GPU models; use ``jacfwd`` for the CPU integrator,
whose adaptive loops do not support reverse-mode ``grad`` or ``jacrev``.

Looking at one derivative
-------------------------

The last column is :math:`\partial A/\partial\rho`. Let's plot it below the
magnification so we can see where the source size matters:

.. code-block:: python

   A = jax.jit(lightcurve)(parameters)

   fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                            layout="constrained")
   axes[0].plot(times, A, color="C0")
   axes[0].set_ylabel("Magnification")
   axes[1].axhline(0, color="0.8", linewidth=0.8)
   axes[1].plot(times, jacobian[:, 6], color="C1")
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
and ``mag_triple``. The :doc:`examples/index` plots all seven binary and
all ten triple derivatives and includes the complete scripts.
