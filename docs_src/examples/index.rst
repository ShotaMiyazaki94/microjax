Examples
========

Compute binary and triple light curves and their parameter derivatives.
The examples use linear limb darkening with ``u1=0.5`` and float64.
CPU and GPU versions share the event parameters and observation times.

Run from a source checkout with Matplotlib installed:

.. code-block:: bash

   python -m pip install -e . matplotlib

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Example
     - Backend
     - Result
   * - `Binary light curve`_
     - CPU or GPU
     - 500 observations, seven derivatives
   * - `Triple light curve`_
     - CPU or GPU
     - 1000 observations, ten derivatives
   * - :doc:`Magnification benchmarks <../benchmarks>`
     - CPU
     - Accuracy and synchronized runtime

Binary light curve
------------------

.. code-block:: bash

   python example/cpu/lightcurve.py

The event has ``q=0.001``, ``s=1``, ``rho=0.005``, ``u0=0.1``,
``tE=10`` days, and ``alpha=50°``. The upper panel shows magnification,
with the source trajectory, caustics, and critical curves in the inset.
The remaining panels show derivatives with respect to
``t0, tE, u0, q, s, alpha, rho``.

.. figure:: ../../example/cpu/lightcurve.png
   :alt: Binary-lens light curve, lens geometry, and seven parameter derivatives
   :width: 720px

   CPU binary-lens magnification and forward Jacobian, sampled from −5 to 10 days.

The GPU version runs the same event through the accelerator backend:

.. code-block:: bash

   python example/gpu/jacobian-binary/code/grads_limb_dark_binary.py

See the `GPU binary example <https://github.com/ShotaMiyazaki94/microjax/tree/develop/example/gpu/jacobian-binary>`_
for its timing report and output options.

Triple light curve
------------------

.. code-block:: bash

   python example/cpu/lightcurve.py --lens triple

The triple event has ``q=0.1``, ``s=1.1``, ``q3=0.01``, and ``rho=0.01``.
The third lens lies at ``0.3+1.2j`` relative to the midpoint of the first two.
Its position is supplied as ``r3=abs(0.3+1.2j)`` and
``psi=angle(0.3+1.2j)``. The trajectory uses ``u0=0.1``, ``tE=10`` days,
and ``alpha=50°``. The Jacobian adds ``q3, r3, psi`` to the binary parameters.

.. figure:: ../../example/cpu/triple_jacobian.png
   :alt: Triple-lens light curve, lens geometry, and ten parameter derivatives
   :width: 720px

   CPU triple-lens magnification and forward Jacobian at 1000 observation times.

To evaluate the same trajectory on GPU:

.. code-block:: bash

   python example/gpu/jacobian-triple/code/grads_limb_dark_triple.py

See the `GPU triple example <https://github.com/ShotaMiyazaki94/microjax/tree/develop/example/gpu/jacobian-triple>`_
for its timing report and output options.

Source and outputs
------------------

Download the :download:`CPU example script <../../example/cpu/lightcurve.py>`
or open it `on GitHub <https://github.com/ShotaMiyazaki94/microjax/blob/develop/example/cpu/lightcurve.py>`_.
Edit ``params`` and ``times`` in ``main()`` to change the event.
The script obtains magnification and derivatives together using
``jax.jacfwd(..., has_aux=True)``; see :doc:`../autodiff` to build your own model.

CPU output files are ``lightcurve.png``, ``lightcurve.csv``, and
``jacobian_forward.npy``. They go to ``results/cpu/example/`` for the binary
and ``results/cpu/example/triple/`` for the triple. GPU scripts write to their
``outputs/limb_dark/`` directories and accept ``--output-dir PATH``.
