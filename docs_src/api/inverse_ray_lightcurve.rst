Finite-source magnification
===========================

``mag_binary`` and ``mag_triple`` integrate a circular source with linear
limb darkening. Both take a one-dimensional complex array of source positions
and return one magnification per position. See :doc:`../usage` for a worked
binary event and :doc:`../autodiff` for parameter derivatives.

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

Light-curve functions
---------------------

.. autofunction:: microjax.inverse_ray.mag_binary

.. autofunction:: microjax.inverse_ray.mag_triple

.. _accelerator-configuration:

Accelerator configuration
-------------------------

Pass a configuration object to tune the accelerator implementation, for
example ``config=BinaryMagConfig(source_tile_size=256)``. The CPU integrator
uses internal settings and ignores ``config``.

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

.. autoclass:: microjax.inverse_ray.BinaryMagConfig

.. autoclass:: microjax.inverse_ray.TripleMagConfig

CPU result and status
---------------------

With ``backend="cpu", return_info=True``, either function returns a
``CpuMagnificationResult``. Its ``magnification`` and ``status`` arrays
have one entry per source position.

.. autoclass:: microjax.inverse_ray.cpu.CpuMagnificationResult
   :members: magnification, status

A zero status means no structural problem was detected. Nonzero flags
indicate problems with image roots, geometry, integration capacity, or
non-finite arithmetic. The calculated magnification is retained; the
status is a diagnostic flag, not a flux-error estimate. The accelerator
implementation returns ``NaN`` for detected calculation failures.
