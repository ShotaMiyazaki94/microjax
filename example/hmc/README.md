# HMC injection–recovery

Generate mock photometry of a binary lens, then infer its seven event parameters
and source/blend fluxes with NumPyro NUTS on CPU.

From the repository root:

```bash
python -m pip install -e . 'numpyro>=0.19' scipy matplotlib
python example/hmc/injection_recovery.py
```

The script first fits the data from a rough initial solution and estimates a
local covariance for NUTS. It then runs **one chain with 150 warm-up steps
and 500 retained draws in total**. Each light-curve evaluation distributes
the observation times across six CPU devices with JAX `pmap`. CPU workers
do not multiply the number of samples. The settings are constants at the
top of the script; there are no command-line options.

Results go to `results/hmc/binary/` in the checkout. Each run replaces the
contents of this directory. Compilation and the local fit happen before the
sampling progress bar advances.

The script saves data, posterior samples, sampler diagnostics, model curves,
and three figures. To redraw the figures without running JAX or sampling again:

```bash
python example/hmc/plot.py
```

After the run, share `run.json`, `summary.json`, `lightcurve.png`, `recovery.png`,
and `traces.png`. The `.npz` files retain the arrays for further analysis.

The [tutorial](../../docs_src/inference.rst) describes the priors and outputs.
`TRUTH` defines the injected event; `INITIAL` is a rough solution used only
to start optimization. Broad uniform and log-uniform priors are specified
separately in `PRIOR_BOUNDS`. Neither the initial guess nor the fitted
covariance narrows those priors. The chain explores the mode found by the
local fit. Data and fitting use the same model.

`run.json` records the fit status and parameters. `initialization.npz` retains
the fitted point, initial position, and covariance passed to NumPyro's
`inverse_mass_matrix`. This matrix stays fixed; warm-up tunes the step size.
The full likelihood and broad priors are used throughout sampling.

This example is prepared for a user-run calculation; recovery results have
not yet been generated or assessed.
