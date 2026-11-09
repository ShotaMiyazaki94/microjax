HMC injection–recovery
======================

This example adds noise to the :doc:`binary event <usage>` and fits its
seven event parameters with NumPyro NUTS on CPU. Source and blend fluxes
are integrated out of the likelihood. A light-curve comparison shows the
fit to the data; a corner plot shows parameter uncertainties and correlations.

Running the example
-------------------

From a :doc:`source checkout <getting_started>`:

.. code-block:: sh

   python -m pip install -e . 'numpyro>=0.19' scipy matplotlib corner
   python example/hmc/injection_recovery.py

The script first fits from a rough initial solution and rescales the sampling
coordinates using the local covariance. It then runs **one chain with
150 warm-up steps and 1,000 retained draws in total**. Each model evaluation
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

Reading the results
-------------------

The run saves two figures:

* ``lightcurve.png`` shows the full event and a closer view of the peak,
  with residuals below. Blue is the median model flux and its central 68%
  interval from 64 posterior draws. For each event draw, source and blend
  fluxes are drawn from their conditional Gaussian posterior, so the band
  includes their uncertainty. Orange is the injected
  curve; data error bars show measurement noise.
* ``corner.png`` shows the seven event parameters. The diagonal gives each
  marginal distribution; the lower triangle shows pairwise correlations.
  Blue contours enclose approximately 68% and 95% of the smoothed pairwise
  distributions. Orange lines mark the injection. Angles are shown in
  degrees, and :math:`q` and :math:`\rho` are multiplied by 1,000 for
  readable axes. The plot uses `corner.py <https://corner.readthedocs.io/>`_.

Read the figures together: a close fit to the photometry can coexist with
correlated or weakly constrained parameters. At this uniform cadence, the
short caustic feature has only a few measurements. Compare the injection
with the posterior intervals, rather than expecting every posterior median
to equal the injected value.

``summary.json`` contains medians, central 68% and 95% intervals, and effective
sample sizes, which account for correlation between draws. It also records
divergences, the mean acceptance probability, and how often a draw used the
full 255-step NUTS budget. With one chain, no between-chain convergence
statistic is reported.

``run.json`` records the settings, priors, fit status, seeds, and software
versions. ``initialization.npz`` retains the fitted point and coordinate
transform. ``samples.npz`` contains the seven physical parameters, each
with shape ``(1, 1000)`` and angles in radians. The sampler state and
per-draw diagnostics are in ``sampler.npz``. These files are saved before
the model curves are computed.

To redraw the two figures from saved arrays:

.. code-block:: sh

   python example/hmc/plot.py

This does not evaluate microJAX or run NUTS. The result figures for the updated
uniform-cadence, flux-marginalized example will be added after the local
calculation completes.
