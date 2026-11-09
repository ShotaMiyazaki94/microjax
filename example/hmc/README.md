# Fitting a simulated event with HMC

Generate noisy photometry of a binary lens and fit its seven event parameters.
We use Hamiltonian Monte Carlo (HMC) to estimate the uncertainties and compare
the inferred parameters with the values used to generate the data. NumPyro's
No-U-Turn Sampler (NUTS) runs on CPU. Source and blend fluxes are integrated
out analytically with broad Gaussian priors.

From the repository root:

```bash
python -m pip install -e . 'numpyro>=0.19' scipy matplotlib corner
python example/hmc/injection_recovery.py
```

The data contain 360 uniformly spaced times from day −30 to day 30,
about one every 0.167 days (4 hours).
Gaussian noise has a standard deviation of 3% of the injected flux.
The script fits from a rough initial guess, then rescales the sampling
coordinates using the estimated parameter widths and correlations.
It runs **one chain with 500 warm-up steps and 1,000 retained draws in total**.
Warm-up tunes the sampler's step size and relative parameter scales.
NUTS uses `max_tree_depth=10`.
Six CPU devices share the observation times in each model evaluation.

`TRUTH` and `FLUX_TRUTH` define the injection; `INITIAL` starts the local fit.
The event priors in `PRIOR_BOUNDS` use coarse ranges for this weeks-long event:
`t0` from −1 to 1 day, `tE` from 1 to 100 days, `u0` from 0 to 1,
`q` from 10⁻⁴ to 1, `s` from 0.5 to 2, and `rho` from 10⁻⁴ to 0.05.
All trajectory angles are included. Source and blend
fluxes each have a zero-mean Gaussian prior with standard deviation 10
(`FLUX_PRIOR_SCALE`), allowing both positive and negative values.
The [tutorial](../../docs_src/inference.rst) explains these choices and the
flux marginalization.

Results go to `results/hmc/binary/`; each run replaces that directory.
The script saves data, samples, diagnostics, model curves, and two figures:

- `lightcurve.png`: data and densely evaluated model curves, with both data
  and posterior residuals measured relative to the injection.
- `corner.png`: the seven event parameters plus source and blend fluxes,
  with logarithmic source radius.

For each of the 1,000 event samples, source and blend fluxes are drawn jointly
from their conditional Gaussian posterior. The corner plot includes all nine
variables; the model interval uses 64 of the same joint samples.
The tutorial shows this 360-point, 3%-noise run: NUTS took 190 seconds
including compilation and warm-up. The tutorial explains the sampling
checks and how to read the plots.
To redraw the figures from saved arrays without sampling again:

```bash
python example/hmc/plot.py
```

`run.json` records the settings and `summary.json` the parameter summaries
and sampling checks. The `.npz` files contain the samples, initialization,
and model curves.
