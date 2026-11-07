CPU backend
===========

Use ``backend="cpu"`` for the CPU finite-source integrator. Binary and triple
lenses share image tracing and quadrature; the solver applies a multipole
approximation away from caustics and integrates the resolved images nearby.

Evaluate magnification
----------------------

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from microjax.inverse_ray import mag_binary, mag_triple

   w = jnp.asarray([.10 + .20j, .60 - .20j])
   A = mag_binary(w, rho=.01, s=1., q=.3, u1=.5, backend="cpu")
   A_triple = mag_triple(
       w, rho=.01, s=1.1, q=.1, q3=.01,
       r3=jnp.abs(.3 + 1.2j), psi=jnp.angle(.3 + 1.2j),
       u1=.5, backend="cpu",
   )

Use positive source radius, separation, and mass ratios, with distinct lens
positions. For a two-lens model, call ``mag_binary`` directly. The source and
lens coordinate conventions are described in :doc:`usage`.

``backend`` selects the algorithm; JAX selects its execution device. On a
machine with both CPU and GPU devices, choose the CPU before launching Python:

.. code-block:: sh

   JAX_PLATFORMS=cpu python example/cpu/lightcurve.py

Alternatively, place the calculation inside
``with jax.default_device(jax.devices("cpu")[0]):``.

The CPU settings are internal. ``BinaryMagConfig`` and ``TripleMagConfig``
configure the accelerator integrator only.

Forward derivatives
-------------------

Compile the model and differentiate a physical parameter:

.. code-block:: python

   curve = jax.jit(lambda rho: mag_binary(
       w, rho, s=1., q=.3, u1=.5, backend="cpu",
   ))
   A = curve(.01)
   dA_drho = jax.jacfwd(curve)(.01)

CPU differentiation uses forward mode: ``jax.jacfwd`` for a Jacobian and
``jax.jvp`` for a directional derivative. Its adaptive loops do not support
reverse-mode ``jax.grad`` or ``jax.jacrev``. The :doc:`autodiff` guide builds
a full event Jacobian and obtains values and derivatives together.

The runnable examples plot seven binary or ten triple derivatives:

.. code-block:: sh

   python example/cpu/lightcurve.py
   python example/cpu/lightcurve.py --lens triple

See the figures in :doc:`examples/index`.

Calculation status
------------------

Request the magnification and a per-position status together:

.. code-block:: python

   result = mag_binary(
       w, rho=.01, s=1., q=.3, u1=.5,
       backend="cpu", return_info=True,
   )
   A, status = result.magnification, result.status

``status == 0`` means that no structural problem was detected. Nonzero values
flag image roots, geometry, integration capacity, or non-finite arithmetic.
The calculated magnification is retained in either case. Status is a
diagnostic flag, not an estimated flux error; :doc:`caveats` describes the
accuracy measurements.
