# HMC injection–recovery

Generate mock photometry of a binary lens and infer its seven event parameters
with NumPyro NUTS on CPU. Source and blend fluxes are marginalized over broad
Gaussian priors, using analytic integration.

From the repository root:

```bash
python -m pip install -e . 'numpyro>=0.19' scipy matplotlib corner
python example/hmc/injection_recovery.py
```

The data contain 180 uniformly spaced times from day −30 to day 30.
The script fits from a rough initial solution, whitens the sampling coordinates
using its covariance, then runs **one chain with 150 warm-up steps and
1,000 retained draws in total**. Warm-up adjusts the mass matrix and step size.
Six CPU devices share the observation times in each model evaluation.

`TRUTH` and `FLUX_TRUTH` define the injection; `INITIAL` starts the local fit.
The event priors are specified separately in `PRIOR_BOUNDS`. Source and blend
fluxes each have a zero-mean Gaussian prior with standard deviation 10
(`FLUX_PRIOR_SCALE`), without truncation. These priors are independent of the
injection; their ranges, widths, and shapes still encode assumptions.
The [tutorial](../../docs_src/inference.rst) explains these choices and the
flux marginalization.

Results go to `results/hmc/binary/`; each run replaces that directory.
The script saves data, samples, diagnostics, model curves, and two figures:

- `lightcurve.png`: data, posterior model interval, injection, and residuals.
- `corner.png`: the seven event parameters and their correlations.

The model interval includes conditional source/blend flux uncertainty.
To redraw the figures from saved arrays without sampling again:

```bash
python example/hmc/plot.py
```

After the run, share `run.json`, `summary.json`, and the two PNGs. The `.npz`
files retain the samples, initialization, and sampler diagnostics.
