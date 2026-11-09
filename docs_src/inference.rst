HMC injection–recovery
======================

This example adds measurement noise to the :doc:`binary event <usage>` and
fits the resulting photometry. It samples all seven event parameters and
the source and blend fluxes, then compares their posterior distributions
with the injected values.

The sampler is NumPyro's
`NUTS <https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.hmc.NUTS>`_,
an adaptive form of Hamiltonian Monte Carlo. The example uses forward-mode
differentiation through microJAX, including the CPU integrator's adaptive
loops.

Running the example
-------------------

From a :doc:`source checkout <getting_started>`:

.. code-block:: sh

   python -m pip install -e . 'numpyro>=0.19' scipy matplotlib
   python example/hmc/injection_recovery.py

The script first fits the data from a rough event solution and estimates a
local covariance. It then runs **one chain with 500 retained draws in total**,
after 150 warm-up steps. Warm-up draws are not saved. Model compilation and
the fit happen before the NUTS progress bar starts.

This example has one CPU execution path and no command-line options.
The run settings are constants at the top of the script. Each light-curve
evaluation distributes observation times across six logical CPU devices
using ``jax.pmap``. The times are interleaved to share the short, expensive
feature across devices, then returned to their original order. This follows
the time-parallel calculation in the
`microlux example <https://github.com/CoastEgo/microlux/blob/master/example/KB0371.py>`_.
The number of CPU workers does not change the sample count. ``run.json``
records both counts. Launch the script in a fresh Python process so the
CPU devices can be configured before importing microJAX.

Results go to ``results/hmc/binary/`` in the checkout; each run replaces
the contents of that directory. The code is in
``example/hmc/injection_recovery.py`` and ``example/hmc/plot.py``.

The event and its priors
------------------------

The data contain 80 times between days −30 and 30 to sample the baseline,
40 around the broad peak between days −5 and 5, and 60 around the short
feature between days −0.9 and −0.4. Each interval is sampled evenly.
The flux model is

.. math::

   F(t) = F_s A(t; t_0,t_E,u_0,q,s,\alpha,\rho) + F_b.

Independent Gaussian noise has a standard deviation of 1% of the injected
flux at each time. Those error bars are held fixed in the fit. Linear limb
darkening is fixed at ``u1=0.5``.

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
     - Uniform(0, 10)
   * - :math:`F_b`
     - 0.2
     - Uniform(−5, 5)

Both arguments are bounds in physical units; LogUniform is uniform in the
logarithm between those bounds. These priors are independent of the injected
values and the initial guess. In particular, the mass-ratio prior includes
both planetary and comparable-mass binaries. ``PRIOR_BOUNDS`` and
``LOG_INDICES`` specify the priors in the script.

The likelihood is written directly in NumPyro:

.. literalinclude:: ../example/hmc/injection_recovery.py
   :language: python
   :pyobject: model

NUTS uses ``forward_mode_differentiation=True``, a dense mass matrix,
``target_accept_prob=0.9``, and ``max_tree_depth=8``. No custom gradient
wrapper is needed.

Initial position and mass matrix
--------------------------------

``INITIAL`` is a rough event solution used only to start a local fit.
L-BFGS-B finds a posterior mode in unbounded logit coordinates,
including the coordinate-transform Jacobian. The prior remains the broad
distribution above throughout optimization and sampling.

At the fitted point, the script estimates a covariance from the Gaussian
likelihood's Fisher information and the transformed prior's curvature.
The residual Jacobian uses ``jax.jacfwd``; this follows the approach in
`numpyro-inferutils <https://github.com/kemasuda/numpyro-inferutils>`_
and works with the CPU integrator. A small eigenvalue floor stabilizes the
matrix inversion.

The sampler uses whitened coordinates :math:`w`, defined by

.. math::

   z = z_{\mathrm{MAP}} + Lw, \qquad LL^\mathsf{T} = C_{\mathrm{MAP}},

where :math:`z` are the unbounded logit coordinates and :math:`C_{\mathrm{MAP}}`
is the estimated covariance. This makes the local covariance approximately
the identity, so NumPyro's covariance regularization acts on comparable scales.
NUTS starts near :math:`w=0` with an identity mass matrix. Warm-up adjusts
both the dense mass matrix and step size; both stay fixed during the 500
retained draws. The coordinate transform stays fixed throughout the run.

The original broad priors are transformed exactly, with their Jacobian
included by ``TransformedDistribution``. The MAP covariance sets the sampling
coordinates; the posterior still uses the original broad priors and full
nonlinear likelihood.

``run.json`` records the fit status and parameters. ``initialization.npz``
saves ``map_z``, ``cholesky``, and ``initial_whitened`` so the coordinate
transform and starting point can be reconstructed. A fit that reaches its
iteration limit can still supply an initial point, and its status is reported.

This example explores the mode near the initial solution; it does not search
for every binary-lens degeneracy. Injection and fitting use the same microJAX
model. ``TRUTH`` is used for mock data and recovery comparisons, not to set
the priors or initialize the sampler.

Reading the results
-------------------

The run saves three figures:

* ``lightcurve.png`` overlays the data, injected curve, and a 68% interval
  for the model flux from up to 64 posterior draws. This band describes uncertainty
  in the model curve; the data error bars show measurement noise.
* ``recovery.png`` shows the nine marginal posterior distributions and
  their injected values.
* ``traces.png`` shows the retained draws and the injected parameter values.

``summary.json`` contains posterior medians, central 68% and 95% intervals,
and effective sample sizes, which account for correlation between draws.
It also records divergences, the mean acceptance probability, and how often
a draw used the full 255-step NUTS budget. With one chain, no between-chain
convergence statistic is reported. ``run.json`` records the settings, priors,
seeds, and software versions.

The arrays are retained in ``data.npz``, ``samples.npz``, ``sampler.npz``, and
``curves.npz``. Posterior parameters in ``samples.npz`` have shape
``(1, 500)``, with angles in radians. ``sampler.npz`` stores the whitened
draws, diagnostics, and the final ``inverse_mass_matrix`` and ``step_size``
in whitened coordinates. Samples are saved before curve
evaluation and plotting, so those later steps cannot discard a completed
sampling run.

To redraw the figures from saved arrays:

.. code-block:: sh

   python example/hmc/plot.py

This plotting command does not evaluate microJAX or run the sampler.
After the first run, ``run.json``, ``summary.json``, and the three PNGs are
enough to review the recovery. The example is currently prepared for that
run; no recovery result is claimed here yet.
