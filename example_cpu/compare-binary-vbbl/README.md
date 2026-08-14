# CPU Binary-Lens Benchmarks

CPU counterpart of `example/compare-binary-vbbl`. One script evaluates
uniform and linearly limb-darkened sources on exactly the same 1,000-point
trajectory and compares both profiles with VBMicrolensing (VBML).
The substantive solver change is the magnification call. The default CPU
scheduler traces 64 source-limb support points exactly once, selects a fixed
high-order Cartesian or polar route from that image state, and performs no
coarse/fine convergence test, retry, or rescue. The support count locates image topology;
it is not the integration quadrature order.

```python
mag_binary(..., backend="cpu", return_info=True)
```

`return_info=True` selects the best-effort CPU magnification field so the
original plotting and timing flow remains unchanged. Explicit exhaustion and
error-bound statistics can be checked with separate parameter sweeps; those
local benchmark datasets are not part of this example.

```bash
python example_cpu/compare-binary-vbbl/code/compare_binary_profiles.py
```

The Python sources live in `code/`; generated files are written to `outputs/`:

- `compare_binary_uniform_limb_dark.png`: shared magnification panel followed
  by full-width Uniform and LD residual panels.
- `compare_binary_profiles_uniform_max_residual_icrs.png`: two-panel Uniform
  diagnostic showing the image profile and final one-dimensional integral.
- `compare_binary_profiles_limb_dark_max_residual_icrs.png`: corresponding LD
  diagnostic.
- `benchmark_binary_profiles.json`: shared configuration, timings, status
  counts, and accuracy statistics.

The script evaluates VBML twice for each profile. `Tol=RelTol=1e-4` is used
for the reported VBML speed and CPU/VBML ratio; `Tol=RelTol=1e-6` is used as
the residual reference (and for the VBML curve in the comparison figure).
The `1e-4` timing is a median over `--repeats`; the `1e-6` reference is a
single run. The residual panels show both references: solid for `1e-6` and
dotted for the VBML reference spread `VBML(1e-4) - VBML(1e-6)`. The solid
curve is `microJAX - VBML(1e-6)`. Both timings and residual statistics are
retained in `benchmark_binary_profiles.json`.

Runtime, tier counts, and accuracy vary with the command-line configuration.
The console prints point-source, hexadecapole, VBML, and one-shot CPU ICRS
timings in the same detailed format as the original examples. Exact inputs and
results from the latest run are stored in `benchmark_binary_profiles.json`.
Any structurally invalid samples remain explicitly marked. Full one-shot
results do not carry a per-point error estimate.

The combined comparison prints `rho` and the number of structurally valid
results whose measured reference residual exceeds the configured validation
threshold for each profile. The production CPU multipole gate is fixed
internally; this threshold is an external validation statistic, not a per-point
guarantee.
It writes all three figures, then exits nonzero if either profile exceeds the
configured validation criterion.
