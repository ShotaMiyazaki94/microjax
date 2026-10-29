CPU Binary-Lens Backend
=======================

``mag_binary(..., backend="cpu")`` selects microJAX's CPU-oriented
finite-source binary-lens solver. It is a separate implementation from the
default accelerator backend, not an automatic fallback chosen from the
available JAX device.

The CPU backend supports binary lenses with uniform or linear limb-darkened
circular sources. Finite-source triple-lens calculations use the accelerator
backend.

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
       w,
       1.0e-2,
       s=1.0,
       q=0.3,
       u1=0.0,
       backend="cpu",
   )

The CPU solver has internal scheduling and bounded quadrature settings.
``BinaryMagConfig`` and ``n_limb`` configure the accelerator backend and do
not tune this path. ``"cpu-one-shot"`` remains an alias of ``"cpu"`` for
compatibility. The older ``"cpu-adaptive"`` backend is retained for research
comparisons and should not be used as a silent retry in production models.

Numerical contract
------------------

The CPU backend uses bounded work. It applies a multipole approximation where
its internal gate accepts it and otherwise performs one full finite-source
solve. Its Cartesian and theta-first integrals use an internal
relative stopping indicator and a fixed maximum amount of work. It does not
retry another chart or keep increasing work until a user-requested tolerance
is met.

Uniform and linear limb-darkened sources share one full solver with two
integration routes: Cartesian (with x/y chosen as an internal axis) and polar
theta-first. One 112-sample source-limb trace selects the route before
integration. Each
source position then integrates that chart within a fixed work budget: it does not
evaluate a second coordinate chart. The
route name is an execution diagnostic, not an accuracy ranking.

Route selection uses only the traced image state: physical image
counts and transitions, ghost-root residuals and angles, projected extrema,
and radial versus tangential image motion. There are no ``q``, ``s``, ``rho``,
or source-position regions that select a support family. Cartesian orientation
is chosen by comparing the x and y projections of the same traced image limb.

Cartesian integration cells are bounded by refined
projection tangencies and source-limb critical crossings. Fixed Newton seeds
come from the image trace, observed image-count transitions, and the closest
rejected image pair. A single source probe between hidden critical crossings
can reveal an image pair absent from the original limb samples. A physical
trace point inside a cell proves occupancy; cells without that evidence use
one midpoint ray. Finite event searches are not a completeness certificate.

Theta-first integration refines every candidate angular endpoint and adds
critical-crossing angles and four common polar charts. Root-slot changes can
otherwise resemble branch endings and leave an occupied angular sector out of
the support. Converged tangencies and critical positions use implicit forward
derivatives of their defining lens equations. When an angular
tangency fails to converge and the fitted trace endpoint is returned, AD
follows that fitted endpoint instead of freezing its parameter dependence.

Both charts share the two-moment interface. Each ray evaluates
the uniform area and the linear-profile moment together; the final flux mixes
them using ``u1``. If these image integrals are :math:`M_0` and :math:`M_1`,
the magnification is

.. math::

   A(u) = \frac{(1-u)M_0 + u M_1}{\pi\rho^2(1-u/3)}.

At ``u1=0``, this is exactly the uniform image-area integral. There is no
separate uniform full-solve dispatch. Both moment indicators remain active at
zero coefficient, so support, chart choice, and quadrature work are independent
of the brightness coefficient. Each support cell starts with two panels of five
Gauss-Legendre nodes in a sine-squared coordinate, which regularizes endpoint
singularities. A positive three-node embedded rule reuses samples to rank the panels.
Cartesian and theta-first integration then retain all five values when extending
a panel to an eleven-node Gauss-Kronrod rule: only six new rays are evaluated.
An already extended panel can be bisected into two fresh five-node panels at a
cost of ten rays. Both moments must separately meet the internal relative
indicator ``1e-4`` in two consecutive allocations, with at least one refinement
for nonempty support. The total work is bounded by 84 times the number of active
cells, shared across the cells rather than capped separately per cell.
These are work bounds, not error bounds.

Cartesian and theta-first rays retain eight inner profile nodes. Embedded
discrepancies never trigger another route, a larger work budget, or a retry.
All default CPU full solves use this same cached allocator. The older
uniform-only three-chart helper and ``cpu-adaptive`` remain available for
explicit research comparisons; the default scheduler does not call them.

The cache lives within one integration call and retains moving-endpoint and
parameter dependencies for forward AD. It stores only the five reusable Gauss
samples per panel; the eleven-node extension is temporary. Reduced panel
statuses compose with the six new ray statuses, avoiding a second status cache.
It is not reused between different
source positions or lens parameters. The internal result tracks convergence,
budget exhaustion, and all evaluated rays, including discarded parent panels.
Public structural status remains separate: a zero status does not imply that
the embedded relative stopping condition was met. The standard Cartesian and
theta-first profile kernels do not supply an independent inner-profile error
estimate, and neither embedded difference detects missing support or roots.

The Cartesian and theta-first kernels form their line-intersection
sextics in a locally centered coordinate before solving them. The origin
``s * (1-q) / (1+q)`` changes sign under lens exchange and reduces cancellation
near the lower-mass lens. Each line uses one real companion solve and a fixed
polynomial polishing schedule. During moment integration, accepted real roots
receive one further Newton correction in the unexpanded physical lens equation.
It must reduce the physical residual and stay within a quarter of the nearest
real-root separation; complex pairs keep their original classification.
Real roots carry the implicit derivative of the physical equation. Support
occupancy checks retain the cheaper polynomial-only solve. This correction
does not recover missing roots or establish root completeness. Integration
remains an inverse-ray area integral.

The common solver and denser trace improve numerical consistency at a measured
execution cost, especially for uniform sources. The unification is not a speed
claim. The compact validation inputs and their reference limitations are
included in ``tests/data/cpu_accuracy/``; the full local research archive is
not distributed with the repository.

A result is therefore a numerical estimate, not a certified error bound.
``return_info=True`` exposes a diagnostic ``status`` bit field when traced
geometry, support, capacity, or roots look suspicious. Status does not itself
replace the best-effort magnification with ``NaN`` and is not an accuracy
failure definition. Before an inference run:

- validate values over the intended ``(q, s, rho, w, u1)`` region against an
  independent implementation;
- validate derivatives separately from values;
- inspect nonzero status for unexpectedly inaccurate or non-finite values;
- record failures or non-convergence from the reference solver separately.

For investigation of an individual discrepancy, ``return_info=True`` exposes
the same best-effort result and its internal diagnostic state. Those fields are
debugging details rather than a stable scientific interface; benchmark failure
rates are defined from the final magnification relative to a reference solver.

Forward-mode differentiation
----------------------------

``jax.jvp`` and ``jax.jacfwd`` are supported. Reverse-mode differentiation
through the CPU solver's data-dependent loops is not part of the public API.

.. code-block:: python

   parameters = jnp.asarray([0.10, 0.20, 1.0e-2, 1.0, 0.3])

   def model(values):
       source = jnp.asarray([values[0] + 1j * values[1]])
       return mag_binary(
           source,
           values[2],
           s=values[3],
           q=values[4],
           backend="cpu",
       )[0]

   value = jax.jit(model)(parameters)
   jacobian = jax.jit(jax.jacfwd(model))(parameters)

Caustic crossings and internal route changes can make the numerical graph
piecewise. Forward AD follows the selected support and subdivision schedule;
it does not differentiate discrete panel selection. A finite derivative or
agreement with finite differences of that schedule does not by itself establish accuracy or
smoothness in a surrounding parameter region.

Value-regression checks
-----------------------

For routine checks, run from the repository root:

.. code-block:: sh

   python tools/cpu_accuracy.py --output-dir results/cpu-accuracy/compact-01

This checks 1,144 frozen difficult and broadly sampled positions for both
uniform brightness and linear LD with ``u1=0.5``. A separate 512-position
supplemental check is available through ``--suite holdout``. The collections
use disjoint lens configurations from the same caustic-centered grid; they
are not samples of a general observational input distribution.

The saved full-grid comparison contains 443,700 positions. There is one
difference above 0.1% among 440,781 usable LD references; 2,919 LD references
are missing or invalid. Uniform has 115 raw differences above 0.1%: 114 have
unresolved reference disagreements, and one is a known bad reference that is
replaced by an independently checked value in the compact fixtures. These
counts do not establish an unconditional accuracy bound.

Missing or unresolved references do not count as accuracy passes. The checker
reports their values against the saved baseline separately and detects new
regressions against the comparable references. See
``docs_src/internal/cpu_accuracy_validation.md`` for the policy, commands and
optional full-grid input requirements. These checks do not replace derivative
tests or measure execution speed.

Timing
------

The first call includes tracing and compilation. Warm up the same array shape
and source profile, then synchronize each timed result with
``block_until_ready()``. Compare complete trajectories: runtime depends on how
many positions require a full finite-source calculation. See
:doc:`performance` for a timing example; its configuration controls apply
only to the accelerator backend.

A local x64 CPU measurement on 2026-10-14 covered 12,325 configurations of
36 positions each, with LD coefficient 0.5. Median warm full-solve time was
1.3128 ms/point for microJAX, 1.7372 ms/point for VBMicrolensing 5.3.5 with
``RelTol=1e-4``, and 0.4367 ms/point with ``RelTol=1e-3``; both reference
settings used ``Tol=1e-4``. The faster VBM setting remains faster.

Each value is a configuration's group-mean time, not a separately timed
position. The measurement excludes JIT compilation, worker startup, IPC and
the microJAX multipole shortcut. The two VBM settings exclude 14 and 9
configurations respectively because of timeout or invalid group results.
These one-machine, one-pass measurements are workload-specific and do not
provide uniform-source timings. The full timing data and generated figures
remain in the optional local research archive.

Examples
--------

The repository includes two CPU workflows:

- `CPU/VBM value comparison
  <https://github.com/ShotaMiyazaki94/microjax/tree/main/example/cpu/compare-binary-vbml>`_
- `CPU forward Jacobian
  <https://github.com/ShotaMiyazaki94/microjax/tree/main/example/cpu/jacobian-binary>`_
