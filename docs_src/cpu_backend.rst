CPU Binary/Triple Backend
=========================

``mag_binary(..., backend="cpu")`` and ``mag_triple(..., backend="cpu")``
select the dedicated CPU finite-source integrator. Binary and triple lenses
share support construction and quadrature. Uniform and linear limb-darkened
circular sources use the same implementation.

Basic use
---------

Enable double precision before constructing arrays or compiling functions:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from microjax.inverse_ray import mag_binary, mag_triple

   w = jnp.asarray([0.10 + 0.20j, 0.60 - 0.20j])
   magnification = mag_binary(
       w, 1.0e-2, s=1.0, q=0.3, u1=0.5, backend="cpu",
   )
   triple_magnification = mag_triple(
       w, 1.0e-2, s=1.1, q=0.1, q3=0.01,
       r3=jnp.abs(0.3 + 1.2j), psi=jnp.angle(0.3 + 1.2j),
       u1=0.5, backend="cpu",
   )

Use positive mass ratios (``q`` and, for triples, ``q3``), a positive source
radius, and distinct lens positions. Triple source coordinates retain the
centre of mass of the first two lenses. ``r3`` and ``psi`` place the third lens
relative to the midpoint of those two lenses; changing its mass does not
translate the public source frame.

Both APIs default to the accelerator implementation. ``BinaryMagConfig`` and
``TripleMagConfig``, including ``n_limb``, configure only that implementation;
they do not tune the CPU path. ``"cpu"`` is the sole CPU backend name. The old
``"cpu-one-shot"`` alias and ``"cpu-adaptive"`` implementation are retired.

The backend selects the algorithm. On a machine with multiple JAX platforms,
run with ``JAX_PLATFORMS=cpu`` or use
``jax.default_device(jax.devices("cpu")[0])`` to select the CPU device.
Running the default accelerator implementation on a CPU does not select this
CPU integrator.

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

For either CPU lens model, ``return_info=True`` returns only ``magnification``
and ``status``:

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

The same forward transformations apply to ``mag_triple`` and its additional
``q3``, ``r3``, and ``psi`` parameters. The runnable CPU examples reproduce the
corresponding binary and triple GPU trajectories and plot all seven or ten
physical/trajectory derivatives:

.. code-block:: sh

   python example/cpu/lightcurve.py
   python example/cpu/lightcurve.py --lens triple

The examples hold ``u1=0.5`` fixed. The underlying CPU API also supports
differentiation with respect to ``u1``, including at zero.

Development checks and timing
-----------------------------

Run the focused CPU contract checks from the repository root:

.. code-block:: sh

   pytest -q tests/unittests/inverse_ray/test_cpu.py tests/unittests/inverse_ray/test_cpu_triple.py
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
