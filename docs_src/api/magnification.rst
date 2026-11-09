Finite-source magnification
===========================

Choose an integrator when creating a light-curve function, then call
``lightcurve(parameters, times, u1=...)``. See :doc:`../usage` for
a worked binary event and :doc:`../autodiff` for parameter derivatives.
The position-based functions below support custom source trajectories.
CPU is the default; use ``backend="gpu"`` to select the GPU implementation.

Event light curves
------------------

.. autofunction:: microjax.make_binary_lightcurve

.. autofunction:: microjax.make_triple_lightcurve

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Event parameter
     - Meaning
   * - ``t0``
     - Time of closest approach to the centre of mass of lenses 1 and 2.
   * - ``tE``
     - Einstein crossing time, in the same units as ``times`` and ``t0``.
   * - ``u0``
     - Signed impact parameter in Einstein-radius units.
   * - ``alpha``
     - Trajectory angle from the binary axis, in radians.

The static binary model uses seven parameters in the order
``[t0, tE, u0, q, s, alpha, rho]``. The static triple model appends
``[q3, r3, psi]``, giving ten parameters. Values are in physical units, with
angles in radians. Pass ``u1`` separately; it is not part of the array.
The lens and source conventions are given below.

With a one-dimensional parameter array, ``jax.jacfwd`` returns a Jacobian
whose columns follow that order and whose rows correspond to the times.

.. _source-position-api:

Source-position functions
-------------------------

For trajectories including parallax, or source positions calculated
elsewhere, use the position-based functions. For example:

.. code-block:: python

   import microjax as mj

   mag = mj.make_mag_binary(backend="cpu")
   A = mag(w, rho=0.005, s=1.0, q=0.001, u1=0.5)

Here ``w`` is a one-dimensional complex array. The result has one
magnification per position. The module-level ``mj.mag_binary`` and
``mj.mag_triple`` use the CPU integrator. The factories select either
backend once, with the same physical arguments and array output.

.. autofunction:: microjax.mag_binary

.. autofunction:: microjax.mag_triple

.. autofunction:: microjax.make_mag_binary

.. autofunction:: microjax.make_mag_triple

.. _lens-coordinates:

Coordinates and source profile
------------------------------

Lengths are in angular Einstein-radius units for the total lens mass.
The source coordinate is ``w = x + 1j*y``. The origin is the centre of mass
of lenses 1 and 2, whose positions on the real axis are

.. math::

   z_1 = -\frac{q}{1+q}s, \qquad z_2 = \frac{s}{1+q}.

.. list-table::
   :header-rows: 1
   :widths: 22 78

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

For a triple lens, the source origin stays at the centre of mass of the
first two lenses. The third lens is at

.. math::

   z_3 = r_3 e^{i\psi} + \frac{s(1-q)}{2(1+q)}.

Use positive source radius, separation, and mass ratios, with distinct lens
positions. For a two-lens model, use ``mag_binary`` directly. For a point
source, use :doc:`mag_point_source <point_source>`.

The source brightness follows

.. math::

   I(r) \propto 1-u_1+u_1\sqrt{1-(r/\rho)^2},
   \qquad 0 \le r \le \rho.

Its disk-average brightness is proportional to :math:`1-u_1/3`, which is
included in the magnification normalization. ``u1=0`` gives a uniform disk;
``0 < u1 <= 1`` darkens the limb.

.. _gpu-configuration:

GPU configuration
-----------------

Pass a configuration object when creating a function, for example
``make_binary_lightcurve(backend="gpu", config=BinaryMagConfig(source_tile_size=256))``.
The CPU integrator uses internal settings and does not accept ``config``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Effect
   * - ``n_limb``
     - Source-boundary sampling, which affects geometric resolution.
   * - ``source_tile_size``
     - Number of full finite-source positions evaluated together.
   * - ``radial_chunk_size``
     - Number of radial image regions evaluated together within each position.

The two batch sizes trade memory for throughput; increasing them does not
increase accuracy. Changing a configuration triggers a new JAX compilation.

.. autoclass:: microjax.BinaryMagConfig
   :no-members:

.. autoclass:: microjax.TripleMagConfig
   :no-members:

CPU result and status
---------------------

Set ``backend="cpu", return_info=True`` when creating a function to return
a ``CpuMagnificationResult``. Its ``magnification`` and ``status`` arrays
have one entry per observation time or source position.

.. autoclass:: microjax.CpuMagnificationResult
   :members: magnification, status

A zero status means no structural problem was detected. Nonzero flags
indicate problems with image roots, geometry, integration capacity, or
non-finite arithmetic. The calculated magnification is retained; the
status is a diagnostic flag, not a flux-error estimate. The GPU
implementation returns ``NaN`` for detected calculation failures.
