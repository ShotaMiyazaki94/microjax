# CPU Binary-Lens Benchmarks

CPU counterpart of `example/gpu/compare-binary-vbml`. One script evaluates
uniform and linearly limb-darkened sources on the same trajectory (500 points
by default) and compares both profiles with VBMicrolensing (VBML).
The default CPU scheduler uses one shared 112-point source-limb trace for
both profiles, then selects Cartesian or theta-first integration with a
bounded work budget. The support count locates image topology; it is not the
integration quadrature order.

```python
mag_binary(..., backend="cpu", return_info=True)
```

`return_info=True` exposes the best-effort CPU magnification and diagnostic
statuses. The comparison reports reference differences for this trajectory;
it does not establish an error bound over other lens configurations.

```bash
python example/cpu/compare-binary-vbml/code/compare_binary_profiles.py
```

The Python sources live in `code/`. Running the example generates local files
under `outputs/`; regenerate them when measuring a different implementation or
environment. Existing generated files may belong to an earlier solver version.

- `compare_binary_uniform_limb_dark.png`: shared magnification panel followed
  by full-width Uniform and LD residual panels.
- `compare_binary_profiles_uniform_max_residual_icrs.png`: a legacy illustrative
  reconstruction of the image profile and one-dimensional integration geometry
  at the largest status-zero Uniform residual.
- `compare_binary_profiles_limb_dark_max_residual_icrs.png`: corresponding LD
  diagnostic.
- `benchmark_binary_profiles.json`: shared configuration, timings, status
  counts, and accuracy statistics.

The boundary reconstructions use older support and quadrature helpers. They
illustrate the geometry at a sample selected from the measured light curve;
they do not reproduce the current shared solver's support or GL5/GK11
allocation. The main light curve and its residuals use the public CPU API.

The script evaluates VBML twice for each profile. `Tol=RelTol=1e-4` is used
for the reported VBML speed and CPU/VBML ratio; `Tol=RelTol=1e-6` is used as
the residual reference (and for the VBML curve in the comparison figure).
The `1e-4` timing is a median over `--repeats`; the `1e-6` reference is a
single run. The residual panels show both references: solid for `1e-6` and
dotted for the VBML reference spread `VBML(1e-4) - VBML(1e-6)`. The solid
curve is `microJAX - VBML(1e-6)`. Both timings and residual statistics are
retained in `benchmark_binary_profiles.json`.

The comparison sets the linear coefficient with `solver.a1` and evaluates
both profiles through `BinaryMag2`. The output records the installed
VBMicrolensing version.

Runtime, route counts, and accuracy vary with the command-line configuration.
The console prints point-source, hexadecapole, VBML, and one-shot CPU ICRS
timings. Each local run stores its inputs and results in
`benchmark_binary_profiles.json`. Nonzero diagnostic statuses are explicitly
marked. Full one-shot
results do not carry a per-point error estimate.

The combined comparison prints `rho` and the number of structurally valid
results whose measured reference residual exceeds the configured validation
threshold for each profile. The production CPU multipole gate is fixed
internally; this threshold is an external validation statistic, not a per-point
guarantee.
It writes all three figures and prints a warning if either profile exceeds
the configured validation criterion. Use `tools/cpu_accuracy.py` for the
separate CPU value-regression gate.
