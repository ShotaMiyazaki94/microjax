# Triple magnification benchmark

Compare microJAX on GPU with VBMicrolensing on a 1000-point triple-lens
trajectory. Run from the repository root:

```bash
python -m pip install VBMicrolensing==5.5 matplotlib
python example/gpu/compare-triple-vbml/code/compare_triple_profiles.py
```

The event has `q=0.1`, `s=1.1`, `q3=0.01`, `rho=0.01`, `u0=0.1`,
`tE=10` days, and `alpha=50°`. The third lens is at `0.3+1.2j` relative
to the midpoint of the first two. The script translates the positions sent
to VBMicrolensing to match microJAX's binary-centre-of-mass source frame.

The calculation covers −5 to 10 days, with `n_limb=128`, for uniform and
linear limb-darkened sources (`u1=0.5`). Use `--quick` for a 24-point run,
`--u1` to change limb darkening, or `--no-plot` to save only the numerical report.

## Measurements

The generated JSON report records the event parameters, software versions,
reference settings, relative magnification errors, and synchronized warm
timings. Compilation is excluded from the timed repetitions. Use `--repeats`
and `--accuracy` to change the measurement settings.

## Outputs

Files are written to `outputs/` and stay local; generated outputs are excluded
from Git:

- `benchmark_triple_profiles.json`: versions, event parameters, timings, and errors;
- `compare_triple_uniform_limb_dark.png`: light curves, residuals, and caustics;
- `compare_triple_profiles_<profile>_max_residual_icrs.{png,json}`:
  image boundaries and integration geometry at the largest residual.

The boundary illustration uses 80 limb samples by default; change it with
`--diagnostic-n-limb`. For parameter derivatives, see the
[triple Jacobian example](../jacobian-triple/README.md).

VBMicrolensing is an optional LGPL-3.0 dependency. Cite Bozza et al.,
*A&A* **694**, A219 (2025), when using its multiple-lens solver.
