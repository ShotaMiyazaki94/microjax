# Binary magnification benchmark

Compare microJAX on GPU with VBMicrolensing on a 1000-point binary-lens
trajectory. Run from the repository root:

```bash
python -m pip install VBMicrolensing==5.5 matplotlib
python example/gpu/compare-binary-vbml/code/compare_binary_profiles.py
```

The event has `s=0.9`, `q=0.01`, `rho=0.005`, `u0=0`, `tE=30` days,
and `alpha=40°`. Observation times span −15 to 15 days. The script evaluates
uniform and linear limb-darkened sources (`u1=0.5`) with `n_limb=64`.
Use `--s`, `--q`, `--rho`, `--u1`, and `--points` to change the event.

## Measurements

The generated JSON report records the
parameters, reference tolerances, warm timings, and relative magnification
errors for each profile. JAX compilation is excluded from the median of three
synchronized calls; change the count with `--repeats`.

VBMicrolensing uses `Tol=RelTol=1e-4` for timing and `1e-6` for residuals.
The report also compares those two reference resolutions. For its version 5.5
API, the script sets `solver.a1` and calls `BinaryMag2` for finite-source
magnification.

## Outputs

Files are written to `outputs/` and stay local; generated outputs are excluded
from Git:

- `benchmark_binary_profiles.json`: settings, timings, and error statistics;
- `compare_binary_uniform_limb_dark.png`: light curves, residuals, and caustics;
- `compare_binary_profiles_<profile>_max_residual_icrs.{png,json}`:
  image boundaries and integration geometry at the largest residual.

For parameter derivatives, see the [binary Jacobian example](../jacobian-binary/README.md).
