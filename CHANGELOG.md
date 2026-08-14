# Changelog

This file records user-visible changes to microJAX. The project follows
[Semantic Versioning](https://semver.org/) while it remains in the `0.x`
development series.

## Unreleased — target: 0.2.0

The `0.2` line is a redesigned release, not a patch update to `0.1.1`.

### Added

- Current `microjax.inverse_ray.mag_binary` and `mag_triple` light-curve APIs,
  with `BinaryMagConfig` and `TripleMagConfig` for source-boundary sampling.
- Full finite-source calculations for uniform and linearly limb-darkened
  binary and triple lenses.
- Direct calculation of the angles where image-plane circles cross a lensed
  source boundary, without using a uniform angular grid.
- Automatic coordinate recentering for small isolated images that would
  otherwise occupy a very narrow angular range.
- Automatic selection between a fast finite-source approximation and the full
  image-boundary calculation at each source position.
- GPU-oriented processing of complete source trajectories.
- A legacy coverage-oriented binary-lens CPU scheduler, now selected explicitly
  with `backend="cpu-adaptive"`,
  using root-free Bernstein Cartesian ICRS together with an independent polar
  radial-moment chart for uniform and linearly limb-darkened sources.
  The uniform Bernstein work list uses exact static mask lookup compaction,
  which retains the same selected intervals and fallback decisions while
  reducing the measured Cartesian schedule cost by about 1.4--1.5x.
  Small uniform sources receive source-radius-dependent Bernstein depths up to
  20 (22 for the high-accuracy path), and the multipole/primary shortcut uses
  a `0.25 * rtol` agreement gate. An eight-radius, 8,000-point VBBL sweep fails
  closed if any certified value exceeds the requested tolerance. Uniform
  Bernstein crossings use an implicit-root custom JVP. A 52-sample Cartesian
  scout supports two-view fast acceptance and a topology-only five-view,
  three-independent-axis consensus; rejected points rebuild the polar support
  at 64 samples. Periodic root assignment and one-sample branch turns expose
  extrema missed by a plain sampled sign change. Exact radial tangencies then
  make narrow fold support independent of source-limb sample density. Linear limb
  darkening conditionally refines support to 96 samples before its compact
  polar cross-certificate. The backend provides explicit status diagnostics
  and calibrated `rtol=1e-3` and `1e-4` modes. Its dynamic scheduler supports
  forward-mode AD; reverse-mode AD through data-dependent loops is not
  supported.
- A fixed-route CPU scheduler selected by default with `backend="cpu"`.
  `backend="cpu-one-shot"` remains an equivalent compatibility alias. It
  performs one source-limb trace, chooses one Cartesian or polar quadrature
  from the resulting image state, and reports a failed certificate without
  any retry, rescue chart, order escalation, or retracing.  Uniform Cartesian
  strips use the root-free Bernstein lookup without companion-root repair.
- User-configurable source tiling and radial-region scheduling through
  `BinaryMagConfig` and `TripleMagConfig`.
- Parallax and binary orbital-motion trajectory utilities.
- Forward-mode binary- and triple-lens Jacobian examples for uniform and
  linearly limb-darkened sources, including compile-plus-first and compiled
  GPU timing figures.
- Combined uniform/limb-darkened GPU comparisons against VBMicrolensing, with
  benchmark JSON, residual figures, and maximum-residual image-boundary
  diagnostics.

### Changed

- The recommended finite-source imports now come from `microjax.inverse_ray`.
- The paper-era two-dimensional image-plane grid has been replaced by an
  integration over the regions enclosed by calculated image boundaries.
- A returned finite value is a numerical estimate without a guaranteed error
  bound. If the image boundary or integration region cannot be constructed,
  the public functions return `NaN`.
- Public binary- and triple-lens configuration contains `n_limb` plus advanced
  static scheduler controls; numerical accuracy settings remain internal.
- The public binary accelerator default now uses the A100 dense-benchmark fast
  route: 64 source-limb samples, 512 source points per outer tile, a 40-cell
  local radial kernel, one fixed 19-point radial rule, and no retry or
  mass-ratio chart branch. The matching GPU benchmark defaults to the fast
  chart and 128 configurations per launch.
- The public triple accelerator default now uses 128 source-limb samples, a
  100-point outer tile, and an eight-cell radial chunk. On the checked
  1,000-point VBMicrolensing trajectory this kept uniform and limb-darkened
  maximum relative errors below `1e-3`; wider binary-style source or radial
  batches reduced triple forward-Jacobian throughput.
- Accelerator examples now read their limb counts from the public
  configuration defaults. Their source files and generated products use a
  consistent `code/` and `outputs/` layout, with separate `uniform/` and
  `limb_dark/` Jacobian outputs.
- Active binary-lens and FSPL numerical comparisons now use VBMicrolensing
  consistently, including the explicit limb-darkened APIs. VBBinaryLensing is
  retained only for historical fixture provenance and orbital-motion
  compatibility tests.
- The finite-source point-lens FFTLog implementation is now imported from
  `microjax.fspl`; the former `microjax.fastlens` package name has been
  removed.

### Fixed

- Reduced isolated `NaN` results near binary-lens source boundaries by making
  the angular boundary calculation more robust.
- Fixed low-mass-ratio binary-lens boundary failures caused by asymmetric
  validation of near-unit reciprocal root pairs and roundoff-only radial
  turning points in compact planetary image charts.
- Corrected the planetary-caustic multipole guard from `-1/s` to `a-1/s` in
  the symmetric internal lens frame, preventing finite sources on wide
  planetary caustics from being accepted by the fast approximation.
- Fixed correlated Cartesian certificates for highly magnified, nearly
  annular small-q images. The CPU scheduler now measures tangential versus
  radial motion in its existing limb trace and sends only ill-conditioned
  images directly to the polar chart without another lens-equation solve.
- Removed forward-mode `NaN` values from linear limb-darkening quadrature by
  evaluating a benign square-root radicand on inactive image intervals before
  masking them out.

### Compatibility

- Treat `0.2.0` as potentially API-breaking relative to `0.1.1`.
- Only names imported from `microjax.inverse_ray` are part of the recommended
  finite-source API. Other modules under that package may change during the
  `0.x` series.
- Reproduction of the published methods-paper implementation should use the
  `v0.1.1` Git tag or the `microjaxx==0.1.1` distribution.

### Removed

- The legacy `microjax.caustics` finite-source solver and the experimental
  `microjax.contour` backend. Use `microjax.inverse_ray` for binary- and
  triple-lens finite-source calculations. The paper-era implementation remains
  available from the `v0.1.1` tag and distribution.
- Reverse-mode comparison code and reverse-Jacobian artifacts from the GPU
  Jacobian examples. The examples now benchmark only the supported production
  forward-mode path; their `ad_benchmark.png` figures compare magnification
  with the forward Jacobian, not forward mode with reverse mode.

## 0.1.1 — paper version

- Archived implementation associated with Miyazaki & Kawahara (2025), ApJ,
  994, 144.
- Software archive: <https://doi.org/10.5281/zenodo.17247892>.
