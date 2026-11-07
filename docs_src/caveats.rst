Numerical accuracy
==================

Finite-source calculations use float64. Both backends combine a multipole
approximation away from caustics with image-plane integration nearby. The
measurements below compare magnification and derivatives against independent
reference calculations.

CPU magnification
-----------------

The critical limb-darkened benchmark uses ``u1=0.5`` and fixed reference values
from VBMicrolensing. Its binary cases exercise Cartesian,
polar, and mixed image geometries. Triple cases cover central and third-body
caustics, comparable masses, wide separations, and mass ratios down to
``q=q3=1e-6``.

.. list-table:: Maximum relative reference difference
   :header-rows: 1
   :widths: 45 20 35

   * - Lens model
     - Positions
     - Maximum difference
   * - Binary
     - 48
     - :math:`7.15\times10^{-5}`
   * - Triple
     - 6
     - :math:`7.22\times10^{-6}`

These are measured maxima over the listed cases. The frozen inputs and
references are in ``tools/cpu_benchmark_cases.json`` and
``tools/cpu_triple_cases.json``; :doc:`benchmarks` gives the reproduction
commands and broader binary coverage.

Derivative checks
-----------------

Independent finite differences check the binary position, source-radius,
separation, and mass-ratio derivatives. Triple checks cover all ten event
parameters and additional radius, mass-ratio, and third-lens-position
derivatives near central and third-body caustics. The regression tolerances
are ``5e-4`` relative for the binary reference Jacobian and ``1e-3`` for the
triple checks, with absolute tolerances for small entries.

The fixtures record each reference method, its finite-difference step sizes,
and convergence information. Near caustics, image topology and integration
subdivisions change. A derivative can therefore be more sensitive than the
magnification to numerical resolution. For a fit concentrated in such a
region, compare selected values and derivatives with an independent calculation
at the relevant parameters.

Resolution and diagnostics
--------------------------

The CPU integrator uses a 112-point source-limb trace and bounded quadrature.
For the accelerator integrator, ``n_limb`` controls source-boundary sampling;
``source_tile_size`` and ``radial_chunk_size`` control batching. Increasing
batch sizes changes throughput, not accuracy. See :doc:`performance`.

The solvers use fixed work limits rather than a requested relative-error
tolerance. CPU ``return_info=True`` exposes the structural status described
in :doc:`cpu_backend`; accelerator functions return ``NaN`` for detected
geometry, capacity, or non-finite failures.

For reproducing a calculation, record the microJAX version or commit and the
backend configuration. The :doc:`citing` page identifies the implementation
associated with the methods paper.
