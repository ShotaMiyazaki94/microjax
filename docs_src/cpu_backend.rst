CPU Binary-Lens Backend
=======================

``mag_binary(..., backend="cpu")`` selects the CPU finite-source binary-lens
solver. Uniform and linear limb-darkened circular sources share the same
implementation. Finite-source triple lenses use the accelerator backend.

Basic use
---------

Enable double precision before constructing arrays or compiling functions:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from microjax.inverse_ray import mag_binary

   w = jnp.asarray([0.10 + 0.20j, 0.60 - 0.20j])
   magnification = mag_binary(
       w, 1.0e-2, s=1.0, q=0.3, u1=0.5, backend="cpu",
   )

``BinaryMagConfig`` and ``n_limb`` configure the accelerator backend and do
not tune the CPU path. ``"cpu"`` is the sole CPU backend name. The old
``"cpu-one-shot"`` alias and ``"cpu-adaptive"`` implementation are retired.

Numerical contract
------------------

The solver uses a multipole approximation where its internal gate accepts it.
Other positions receive one finite-source calculation. A source-limb trace
selects Cartesian or theta-first integration; the solver does not retry another
chart. The fixed setup uses a 112-point limb trace, GL5/GK11 outer panels,
and a common 12-node inner Jacobi rule. Adaptive allocation shares a bounded
84-ray budget per active support cell; it does not add another solver route.

Uniform area and the linear brightness moment share support and quadrature.
The brightness coefficient is a regular JAX input: ``u1=0`` uses the same
compiled CPU model as nonzero coefficients, and forward differentiation with
respect to ``u1`` is supported. There is no uniform-specific CPU dispatch.
For image integrals :math:`M_0` and :math:`M_1`, the magnification is

.. math::

   A(u) = \frac{(1-u)M_0 + u M_1}{\pi\rho^2(1-u/3)}.

The internal quadrature indicator controls work; it is not an error bound.
Finite traces and polynomial root searches can miss support or roots, and
bounded integration can exhaust its allocation. Validate both values and
derivatives over the parameter region used in an analysis.

``return_info=True`` returns only ``magnification`` and ``status``:

.. code-block:: python

   result = mag_binary(w, 1.0e-2, s=1.0, q=0.3, backend="cpu", return_info=True)
   magnification, status = result.magnification, result.status

A nonzero status flags suspicious geometry, roots, capacity or non-finite
calculation. The best-effort magnification is preserved; it may be non-finite.
A zero status does not certify accuracy or quadrature convergence. Detailed
route and work counters are no longer part of the CPU result.

Forward differentiation
-----------------------

``jax.jvp`` and ``jax.jacfwd`` are supported. Reverse differentiation through
the data-dependent CPU loops is outside the public API.

.. code-block:: python

   parameters = jnp.asarray([0.10, 0.20, 1.0e-2, 1.0, 0.3])

   def model(values):
       source = jnp.asarray([values[0] + 1j * values[1]])
       return mag_binary(
           source, values[2], s=values[3], q=values[4], backend="cpu",
       )[0]

   jacobian = jax.jit(jax.jacfwd(model))(parameters)

AD follows the selected support and subdivision schedule. Caustic crossings,
route changes and discrete panel choices can make this numerical graph
piecewise. Agreement with finite differences of that graph does not itself
certify physical derivative accuracy.

Development checks and timing
-----------------------------

Run the focused CPU contract checks from the repository root:

.. code-block:: sh

   pytest -q tests/unittests/inverse_ray/test_cpu.py
   python tools/cpu_benchmark.py --output results/cpu-benchmark.json

The benchmark compares linear limb-darkened values (``u1=0.5``) with a small
fixed set of difficult reference positions. It also times a forward Jacobian,
checking finiteness only; this is not a derivative accuracy comparison.
Independent numerical AD checks and the uniform coefficient limit remain in
the focused CPU tests. The benchmark excludes compilation from synchronized
warm timings. Compare the same inputs, precision,
JAX version and machine; trajectory cost depends on how many positions require
a full finite-source calculation.

The former implementations and tests remain available in Git history. Local
archives and measurement results are excluded from version control. Historical
timings belong to their recorded implementations and are not measurements of
the current solver.
