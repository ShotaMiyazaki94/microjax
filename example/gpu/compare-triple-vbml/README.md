# Triple-Lens Benchmark Against VBMicrolensing

This example compares the finite-source triple-lens magnification from
`microjax.inverse_ray.mag_triple` with `VBMicrolensing.MultiMag2` (uniform
source) or `VBMicrolensing.MultiMagDark` (linear limb darkening).

The lens has mass ratios `1:q:q3 = 1:0.1:0.01`. The first two lenses have
separation `s = 1.1`; the third is at `0.3 + 1.2i` in their midpoint frame.
Both solvers receive source coordinates relative to the centre of mass of the
first two lenses. Both comparison scripts explicitly translate the lens
positions passed to VBMicrolensing so that its coordinate system matches
microJAX.

## Install

From the repository environment:

```bash
python -m pip install VBMicrolensing==5.5 matplotlib
```

VBMicrolensing is an optional benchmark dependency and is not imported by the
microJAX package itself.

## Run

Uniform source:

```bash
python example/compare-triple-vbml/compare_triple_uniform.py
```

Linear limb darkening with `u1 = 0.5`:

```bash
python example/compare-triple-vbml/compare_triple_limb_dark.py
```

For a short CPU smoke run:

```bash
python example/compare-triple-vbml/compare_triple_uniform.py --quick
```

The scripts separate JAX compilation from repeated execution and report the
median execution time of each solver. Each profile writes:

- `compare_triple_<profile>.csv`: both light curves and pointwise relative
  differences;
- `compare_triple_<profile>.json`: versions, parameters, timings, and summary
  error statistics;
- `compare_triple_<profile>.png`: light curves, residuals, caustics, lens
  positions, and the source trajectory;
- `compare_triple_<profile>_max_residual_icrs.{png,json}`: image-plane
  boundary roots, radial integration cells, and metadata at the largest
  pointwise discrepancy. The diagnostic defaults to 80 source-limb samples
  independently of the 500-sample comparison solver, avoiding excessive
  triple-root tracking memory while retaining an interpretable geometry plot.

A representative 1000-point CUDA run prints:

```text
number of data points: 1000
computation time: 0.514 sec (0.514 ms per point), median of 3, with VBMicrolensing.MultiMag2
computation time: 3.929 sec (3.929 ms per point), median of 3, with microJAX mag_triple, n_limb=500
microJAX JIT warm-up time: 9.038 sec
relative difference: median=1.128e-06, p95=5.524e-06, max=3.506e-04 at t=7.94294
```

The `compare_triple_limb_dark.py` run measured a median relative
difference of `2.551e-05`, a 95th percentile of `4.912e-05`, and a maximum
of `2.474e-04`. Timings are hardware-dependent. The error statistics above
describe the checked-in trajectory and numerical settings, not a general
accuracy bound.

`VBMicrolensing` is LGPL-3.0 licensed and is only an external runtime
dependency of this example. Scientific use of its multiple-lens solver should
cite Bozza et al., *A&A* **694**, A219 (2025), in addition to the references
requested by the upstream project.
