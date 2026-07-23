Finite-source Point Lens
========================

The FSPL classes calculate a circular finite source magnified by a single
point lens. Construct the source profile once, then call its ``A(u, rho)``
method with the lens-source separation ``u`` and source radius ``rho``.

.. py:class:: fspl_disk()

   Uniform-brightness circular source.

   .. py:method:: A(u, rho)

      Evaluate the finite-source magnification.

.. py:class:: fspl_ld1(a1=0.5)

   Circular source with linear limb darkening. ``a1`` is the linear
   limb-darkening coefficient.

   .. py:method:: A(u, rho)

      Evaluate the finite-source magnification.

.. py:class:: fspl_ld2()

   Circular source with the quadratic radial profile implemented by the FSPL
   solver.

   .. py:method:: A(u, rho)

      Evaluate the finite-source magnification.

.. py:function:: fspl_point(u)

   Evaluate point-source, point-lens magnification.
