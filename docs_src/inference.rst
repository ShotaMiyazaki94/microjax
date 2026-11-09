HMC injection–recovery
======================

This example adds noise to the :doc:`binary event <usage>` and fits its
seven event parameters with NumPyro NUTS on CPU. Source and blend fluxes
are integrated out of the likelihood. A light-curve comparison shows the
fit to the data; a corner plot shows parameter uncertainties and correlations.

Recorded run
------------

The figures below use the run from 9 November 2026 at
`4e9ffaa <https://github.com/ShotaMiyazaki94/microjax/commit/4e9ffaa>`_:
500 warm-up steps and 1,000 retained draws, with the settings described below.
NUTS took **179 seconds**, including compilation and warm-up; the local fit
took another 9.45 seconds. This used six CPU workers, float64, JAX 0.8.0,
and NumPyro 0.19.0. Curve evaluation and plotting are outside those timings.

The photometry is fitted closely, but **48 of the 1,000 retained transitions
are divergent**. The posterior summaries here are provisional:
`divergences can bias HMC estimates <https://mc-stan.org/learn-stan/diagnostics-warnings.html#divergent-transitions-after-warmup>`_.
No draw reached the tree-depth limit. Effective sample sizes are 81–614;
the smallest are for :math:`s`, :math:`\rho`, and :math:`q`.

.. figure:: _static/hmc_lightcurve.png
   :alt: Dense injected and posterior light curves with data, and residuals relative to the injection
   :width: 100%
   :target: _static/hmc_lightcurve.png

   Full event and a closer view of the peak. Grey points are the 180
   observations; orange is the injected curve. Blue shows the median and
   central 68% model interval from 64 posterior draws, including conditional
   source/blend flux uncertainty. Models are evaluated at 5,339 times, with
   spacing at most 0.0025 days around the peak.

The lower panels use the **injected curve as the common reference**. Grey
points show :math:`(F_{\rm data}-F_{\rm inj})/\sigma`; the blue line and band
show the corresponding posterior-model differences. At every plotting time,
:math:`\sigma(t)=0.01F_{\rm inj}(t)`, the same noise prescription used for
the observations. Zero therefore represents the injection.

.. figure:: _static/hmc_corner.png
   :alt: Seven-parameter corner plot with logarithmic source radius and marked divergent transitions
   :width: 100%
   :target: _static/hmc_corner.png

   Diagonal panels show marginal distributions; lower panels show pairwise
   correlations. Blue contours enclose approximately 68% and 95% of the
   smoothed distributions in the displayed coordinates. Orange marks the
   injection. Magenta crosses mark retained states associated with divergent
   transitions; all draws remain in the plotted distributions.

The :math:`q`–:math:`s` correlation is visible, and the source radius spans
several orders of magnitude, so :math:`\rho` is shown logarithmically.
With a 0.335-day cadence, the short caustic feature has only a few observations;
the radius is weakly constrained. Of the 48 divergent transitions, 35 are
associated with :math:`10^{-3}\leq\rho<10^{-2}`. Their cause has not been
isolated from this run alone.

.. csv-table:: Posterior summaries from this run
   :header: "Parameter", "Injected", "Median", "Central 95% interval", "ESS"
   :widths: 18 14 18 36 14

   ":math:`t_0` [day]", "0", "−0.0113", "[−0.0233, 0.0013]", "577"
   ":math:`t_E` [day]", "10", "9.814", "[9.536, 10.069]", "614"
   ":math:`u_0`", "0.1", "0.10191", "[0.09816, 0.10662]", "567"
   ":math:`q`", "0.001", "9.57×10⁻⁴", "[6.23×10⁻⁴, 1.17×10⁻³]", "94"
   ":math:`s`", "1", "0.99675", "[0.97914, 1.00657]", "81"
   ":math:`\alpha` [deg]", "50", "51.04", "[49.59, 52.67]", "263"
   ":math:`\rho`", "0.005", "3.46×10⁻⁴", "[1.27×10⁻⁵, 8.14×10⁻³]", "87"

All seven injected values lie in the reported 95% intervals. This agreement
for one noisy realization does not resolve the divergent transitions. ESS
accounts for correlation between draws; with one chain there is no
between-chain convergence comparison.

Running the example
-------------------

From a :doc:`source checkout <getting_started>`:

.. code-block:: sh

   python -m pip install -e . 'numpyro>=0.19' scipy matplotlib corner
   python example/hmc/injection_recovery.py

The script first fits from a rough initial solution and rescales the sampling
coordinates using the local covariance. It then runs **one chain with
500 warm-up steps and 1,000 retained draws in total**. Each model evaluation
distributes observation times across six CPU devices with JAX ``pmap``;
the worker count does not multiply the number of draws. This follows the
time-parallel calculation in the
`microlux example <https://github.com/CoastEgo/microlux/blob/master/example/KB0371.py>`_.

Launch the script in a fresh Python process so the CPU devices can be
configured before importing microJAX. Compilation and the local fit happen
before the NUTS progress bar starts. Results go to ``results/hmc/binary/``;
each run replaces that directory. Settings are constants at the top of
``example/hmc/injection_recovery.py``.

Data and priors
---------------

The data contain 180 uniformly spaced observations from day −30 to day 30,
about one every 0.335 days:

.. code-block:: python

   times = jnp.linspace(-30.0, 30.0, 180)

The flux model is

.. math::

   F(t) = F_s A(t; t_0,t_E,u_0,q,s,\alpha,\rho) + F_b.

Independent Gaussian noise has a standard deviation of 1% of the injected
flux at each time. Those error bars stay fixed in the fit. Linear limb
darkening is fixed at ``u1=0.5``. Injection and fitting use the same microJAX
model.

.. list-table:: Injected values and priors
   :header-rows: 1

   * - Parameter
     - Injected value
     - Prior
   * - :math:`t_0`
     - 0 days
     - Uniform(−30, 30) days
   * - :math:`t_E`
     - 10 days
     - LogUniform(0.1, 300) days
   * - :math:`u_0`
     - 0.1
     - Uniform(−1, 1)
   * - :math:`q`
     - 0.001
     - LogUniform(10⁻⁶, 1)
   * - :math:`s`
     - 1.0
     - LogUniform(0.2, 5)
   * - :math:`\alpha`
     - 50°
     - Uniform(−π, π) radians
   * - :math:`\rho`
     - 0.005
     - LogUniform(10⁻⁵, 0.1)
   * - :math:`F_s`
     - 1.0
     - :math:`\mathcal{N}(0,10^2)`
   * - :math:`F_b`
     - 0.2
     - :math:`\mathcal{N}(0,10^2)`

Uniform and LogUniform list bounds in physical units; LogUniform is uniform
in the logarithm between those bounds. The two flux priors are independent,
untruncated Gaussians with mean zero and standard deviation 10 in the chosen
flux units. Their width is fixed independently of the injection and fit.
These are broad priors; their ranges, widths, and shapes still encode assumptions.

``PRIOR_BOUNDS`` and ``LOG_INDICES`` define the seven event priors;
``FLUX_PRIOR_SCALE`` sets the flux-prior standard deviation. ``TRUTH`` and
``FLUX_TRUTH`` are used only for data generation and recovery comparisons.
``INITIAL`` starts the local fit and does not define a prior.

Marginalizing the fluxes
------------------------

For each trial event, let :math:`M=[A,1]` and
:math:`V=\mathrm{diag}(\sigma^2)`. With
:math:`\beta=(F_s,F_b)^\mathsf{T}\sim\mathcal{N}(0,\Lambda)` and
:math:`\Lambda=10^2 I`, integrating over both fluxes gives

.. math::

   p(f\mid\theta) = \mathcal{N}\!\left(f;\,0,\,V+M\Lambda M^\mathsf{T}\right).

The script uses :func:`microjax.likelihood.nll_ulens` to evaluate this
Gaussian marginal likelihood analytically through a two-by-two system,
including its determinant term. No numerical integration is needed.

For plotting, the conditional flux posterior is also Gaussian, with
precision :math:`B=\Lambda^{-1}+M^\mathsf{T}V^{-1}M` and mean
:math:`B^{-1}M^\mathsf{T}V^{-1}f`. Drawing from it includes source and blend
flux uncertainty in the model-curve interval.

The NumPyro model therefore samples only the seven event parameters:

.. literalinclude:: ../example/hmc/injection_recovery.py
   :language: python
   :pyobject: model

Initialization and sampling
---------------------------

L-BFGS-B finds a local posterior mode from ``INITIAL`` in unbounded logit
coordinates. Both the fit and NUTS use the marginal likelihood above and
the event priors, including the coordinate-transform Jacobian.

At the fitted point, the script estimates a covariance from the locally
fitted flux residuals and the transformed prior's curvature. The residual
Jacobian uses ``jax.jacfwd``, following the approach in
`numpyro-inferutils <https://github.com/kemasuda/numpyro-inferutils>`_.
This supplies the initial scales for whitened coordinates :math:`w`:

.. math::

   z = z_{\mathrm{MAP}} + Lw, \qquad LL^\mathsf{T} = C_{\mathrm{MAP}}.

NUTS starts near :math:`w=0` with an identity mass matrix. Warm-up adjusts
both the dense mass matrix and step size, then fixes them for the retained
draws. The coordinate transform stays fixed. The MAP covariance sets the
coordinates; the posterior still uses the broad priors and full marginal
likelihood.

The sampler uses ``forward_mode_differentiation=True``, which works through
the CPU integrator's adaptive loops, ``target_accept_prob=0.9``, and
``max_tree_depth=8``. This example explores the mode near the initial
solution; it does not search for every binary-lens degeneracy.

Saved output
------------

``run.json`` records settings, priors, fit status, seeds, and software versions.
``summary.json`` contains the parameter intervals and sampler diagnostics.
``samples.npz`` stores the seven physical parameters with shape ``(1, 1000)``;
angles remain in radians. These files are saved before the model curves
are evaluated.

``curves.npz`` stores a separate dense time grid, the injected curve, 64
posterior flux curves, and the selected sample indices. The fluxes are
conditioned on the 180 observations. Extra plotting times add resolution
to the figure and do not enter the likelihood.

To redraw the two figures from these saved arrays:

.. code-block:: sh

   python example/hmc/plot.py

This does not evaluate microJAX or run NUTS.
