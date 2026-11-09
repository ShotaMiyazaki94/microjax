Benchmarks
==========

How fast is a finite-source calculation, and how accurately does it reproduce
an independent reference? Both answers depend on the lens configuration and
where the source lies. The maps below show this variation for binary lenses,
comparing the CPU integrator with VBMicrolensing for a limb-darkened source
with ``u1=0.5``.

.. figure:: _static/cpu_parameter_maps_ld.png
   :alt: Binary-lens speed ratios and relative errors across mass ratio, source radius, and magnification, with uncertainty intervals and sample counts
   :width: 100%
   :target: _static/cpu_parameter_maps_ld.png

   CPU speed and accuracy across the sampled binary configurations.
   Click the figure to read the individual cells.

Reading the maps
----------------

The upper row shows the median runtime ratio, VBMicrolensing/microJAX.
A value of 2 means microJAX took half as long. The lower row shows the
95th-percentile relative magnification error in parts per million:
100 ppm corresponds to 0.01%. Reading the two rows together shows the
accuracy reached at a given speed.

Each cell gives an estimate, a 68.27% bootstrap interval, and the number of
measured configurations out of those available. The interval describes
uncertainty in the median or percentile, rather than the spread of individual
measurements. It is estimated from 4,000 resamples of whole configurations
within mass-ratio/source-radius groups. Regions with too few samples retain
their counts but omit the interval.

The source positions lie near selected caustics, with one position per
configuration. The grid spans ``q=1e-6..1``, ``s=0.5..1.5``, and
``rho=3e-5..1``. It includes both planetary and comparable-mass lenses.
The run contains 12,244 reference comparisons and 12,236 paired timings.
Paired timings include positions where both solvers agree with the saved
reference to within a relative difference of ``1e-3``.

This figure records the binary CPU implementation at
`998c548 <https://github.com/ShotaMiyazaki94/microjax/commit/998c548>`_
on 6 November 2026, using JAX/JAXLIB 0.8.0 and VBMicrolensing 5.3.5 on a
macOS 26.5 arm64 CPU. Timings compare single-position, full finite-source
calculations after compilation, with multipole screening disabled.
They describe that workload, rather than the throughput of an entire
light curve with the usual multipole screening enabled.

Running a comparison
--------------------

For a quick check on your own machine, run the bundled binary and triple
cases from a source checkout:

.. code-block:: sh

   python tools/cpu_benchmark.py --output results/cpu-benchmark.json

This checks magnification against saved references and measures warm
runtimes for magnification and forward Jacobians. Use ``--lens binary`` or
``--lens triple`` to run one model. The cases include central and third-body
caustics, comparable masses, and small mass ratios; their inputs and reference
methods are recorded in
`cpu_benchmark_cases.json <https://github.com/ShotaMiyazaki94/microjax/blob/develop/tools/cpu_benchmark_cases.json>`_
and
`cpu_triple_cases.json <https://github.com/ShotaMiyazaki94/microjax/blob/develop/tools/cpu_triple_cases.json>`_.

A recorded run at
`e1d8f59 <https://github.com/ShotaMiyazaki94/microjax/commit/e1d8f59d8317369f60fbebbdacea545955d767d3>`_
found maximum relative differences of :math:`7.15\times10^{-5}` over
48 binary positions and :math:`7.22\times10^{-6}` over six triple positions.
These small cases complement the broader binary map above. The generated
JSON report contains the settings, accuracy, and timings for your own run.

Recreating the maps
~~~~~~~~~~~~~~~~~~~

The regional benchmark needs a dense reference CSV and its same-stem JSON
metadata, which are not bundled with the package. If you have those inputs:

.. code-block:: sh

   python -m pip install VBMicrolensing matplotlib
   python tools/cpu_benchmark_grid.py \
       --reference-data /path/to/xdense_cpu.csv \
       --output-dir results/cpu/new-run/data --time-budget 300
   python tools/plot_cpu_benchmark.py \
       --data-dir results/cpu/new-run/data \
       --output-dir results/cpu/new-run/figures

The 300-second measurement budget includes selection and compilation.
Plotting requires a completed run. Alongside the figures,
``regional_statistics.csv`` contains the estimates, bootstrap standard
errors, 68.27% and 95% intervals, and sample counts for each region.
