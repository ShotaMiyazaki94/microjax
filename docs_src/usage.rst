Light curves
============

Calculate binary and triple light curves from a trajectory of complex source
positions. The finite-source functions integrate a circular source with linear
limb darkening; ``mag_point_source`` evaluates the point-source model.

Binary lens
-----------

Enable float64, construct a trajectory, and evaluate its magnification:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from microjax.inverse_ray import mag_binary, mag_triple

   def trajectory(times, t0, t_e, u0, alpha):
       tau = (times - t0) / t_e
       return (tau + 1j * u0) * jnp.exp(1j * alpha)

   times = jnp.linspace(-5., 10., 500)
   w = trajectory(times, t0=0., t_e=10., u0=.1,
                  alpha=jnp.deg2rad(50.))

   binary = jax.jit(lambda w: mag_binary(
       w, rho=.005, s=1., q=.001, u1=.5, backend="cpu",
   ))
   A = binary(w)
   print(A.shape)  # (500,)

``w`` is a one-dimensional array of source positions; the result contains one
magnification per position. ``t0`` is the closest-approach time, ``t_e`` is the
Einstein crossing time, ``u0`` is the signed impact parameter, and ``alpha``
rotates the trajectory from the binary axis. Use the same time unit for
``times``, ``t0``, and ``t_e``; these examples use days.

To plot the result:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.plot(times, A)
   plt.xlabel("Time (day)")
   plt.ylabel("Magnification")
   plt.show()

The :doc:`examples/index` includes this light curve, its lens geometry, and
derivatives with respect to all seven event parameters.

Triple lens
-----------

Add a third lens with mass ratio ``q3`` and position ``r3 * exp(1j * psi)``.
Using the trajectory above:

.. code-block:: python

   triple = jax.jit(lambda w: mag_triple(
       w, rho=.01, s=1.1, q=.1, q3=.01,
       r3=jnp.abs(.3 + 1.2j), psi=jnp.angle(.3 + 1.2j),
       u1=.5, backend="cpu",
   ))
   A_triple = triple(w)

Here the third lens lies at ``0.3 + 1.2j`` relative to the first two lenses'
midpoint. See :doc:`examples/index` for the ten-parameter triple Jacobian.

Coordinates and masses
----------------------

Lengths are in Einstein-radius units for the total lens mass. The source
coordinate is ``w = x + 1j*y``. Both lens models place the origin at the
centre of mass of lenses 1 and 2, with their axis along the real direction:

.. math::

   z_1 = -\frac{q}{1+q}s, \qquad
   z_2 = \frac{1}{1+q}s.

.. list-table:: Lens and source parameters
   :header-rows: 1
   :widths: 18 82

   * - Parameter
     - Meaning
   * - ``s``
     - Separation of lenses 1 and 2.
   * - ``q``
     - Mass ratio :math:`m_2/m_1`.
   * - ``rho``
     - Angular source radius :math:`\theta_*/\theta_E`.
   * - ``u1``
     - Linear limb-darkening coefficient.
   * - ``q3``
     - Third-lens mass ratio :math:`m_3/m_1`.
   * - ``r3``, ``psi``
     - Third-lens distance and angle from the binary midpoint; ``psi`` is in radians.

In source coordinates, the third lens is at

.. math::

   z_3 = r_3 e^{i\psi} + \frac{s(1-q)}{2(1+q)}.

The source origin remains fixed when changing ``q3``, ``r3``, or ``psi``.

Source brightness
-----------------

The linear brightness law is

.. math::

   I(r) \propto 1-u_1+u_1\sqrt{1-(r/\rho)^2},
   \qquad 0 \le r \le \rho.

The disk-average brightness is proportional to :math:`1-u_1/3`.
``u1=0`` gives a uniform disk; ``0 < u1 <= 1`` darkens the limb. On CPU,
these profiles share the same compiled model, including differentiation with
respect to ``u1`` at zero.

Point sources
-------------

Use the point-source API when the source radius is negligible:

.. code-block:: python

   from microjax.point_source import mag_point_source

   A_point = mag_point_source(w, nlenses=2, s=1., q=.001)
   A_point_triple = mag_point_source(
       w, nlenses=3, s=1.1, q=.1, q3=.01,
       r3=jnp.abs(.3 + 1.2j), psi=jnp.angle(.3 + 1.2j),
   )

CPU and GPU
-----------

The examples above select the dedicated :doc:`cpu_backend`. For GPU execution,
use ``backend="accelerator"`` (the default) with a GPU-enabled JAX installation.
``backend="gpu"`` is an alias. See :doc:`performance` for accelerator batching
and :doc:`autodiff` for derivatives of either backend.
