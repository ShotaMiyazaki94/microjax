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

   python -m pip install -e . 'numpyro>=0.19' matplotlib
   python example/hmc/injection_recovery.py

This runs two sequential CPU chains, with 300 warm-up steps and 500 retained
draws per chain. The first model evaluation and the start of NUTS include
compilation. The progress bar then follows warm-up and sampling.

``--warmup``, ``--samples``, and ``--chains`` control the run length.
``--seed`` changes the sampler's random seed; the mock photometry keeps its
fixed noise seed. With a GPU-enabled JAX installation, use ``--backend gpu``.

Results go to ``results/hmc/binary/``. Choose a different ``--output-dir``
for each run so that earlier results are preserved. The two scripts are
available to download:
:download:`injection_recovery.py <../example/hmc/injection_recovery.py>` and
:download:`plot.py <../example/hmc/plot.py>`.

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
     - Normal(0.1, 1.0)
   * - :math:`t_E`
     - 10 days
     - LogNormal(ln 10.5, 0.3)
   * - :math:`u_0`
     - 0.1
     - Normal(0.11, 0.05)
   * - :math:`q`
     - 0.001
     - LogNormal(ln 0.0012, 0.8)
   * - :math:`s`
     - 1.0
     - LogNormal(ln 1.02, 0.15)
   * - :math:`\alpha`
     - 50°
     - Normal(49° in radians, 0.3 radians)
   * - :math:`\rho`
     - 0.005
     - LogNormal(ln 0.006, 0.7)
   * - :math:`F_s`
     - 1.0
     - LogNormal(ln 1.05, 0.3)
   * - :math:`F_b`
     - 0.2
     - Normal(0.15, 0.5)

The second argument is a standard deviation; LogNormal uses natural-log
space. The script samples standardised coordinates, then transforms them
to these physical parameters. This keeps their very different numerical
scales out of the initial sampling geometry.

Each chain starts at a different point near the approximate event solution
in ``PRIOR_CENTER``. This example explores the nearby posterior mode.
Injection and recovery use the same microJAX model, making this an
end-to-end check of the inference workflow.

The likelihood is written directly in NumPyro:

.. literalinclude:: ../example/hmc/injection_recovery.py
   :language: python
   :pyobject: model

NUTS uses ``forward_mode_differentiation=True``, a dense mass matrix,
``target_accept_prob=0.9``, and ``max_tree_depth=8``. No custom gradient
wrapper is needed.

Reading the results
-------------------

The run saves three figures:

* ``lightcurve.png`` overlays the data, injected curve, and a 68% interval
  for the model flux from up to 64 posterior draws. This band describes uncertainty
  in the model curve; the data error bars show measurement noise.
* ``recovery.png`` shows the nine marginal posterior distributions and
  their injected values.
* ``traces.png`` shows the retained draws from each chain.

``summary.json`` contains posterior medians, central 68% and 95% intervals,
effective sample sizes, and split :math:`\hat R`. It also records divergences,
acceptance probabilities, and how often a draw used the full 255-step NUTS
budget. Compare the chains and these diagnostics before interpreting the
recovery plot. ``run.json`` records the settings, priors, seeds, and software
versions.

The arrays are retained in ``data.npz``, ``samples.npz``, ``sampler.npz``, and
``curves.npz``. Posterior parameters in ``samples.npz`` have shape
``(chain, draw)``, with angles in radians. Samples are saved before curve
evaluation and plotting, so those later steps cannot discard a completed
sampling run.

To redraw the figures from saved arrays:

.. code-block:: sh

   python example/hmc/plot.py results/hmc/binary

This plotting command does not evaluate microJAX or run the sampler.
After the first run, ``run.json``, ``summary.json``, and the three PNGs are
enough to review the recovery. The example is currently prepared for that
run; no recovery result is claimed here yet.
