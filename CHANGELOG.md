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
- User-configurable source tiling and radial-region scheduling through
  `BinaryMagConfig` and `TripleMagConfig`.
- Parallax and binary orbital-motion trajectory utilities.
- A triple-lens Jacobian example and binary-lens VBBinaryLensing comparison
  records.

### Changed

- The recommended finite-source imports now come from `microjax.inverse_ray`.
- The paper-era two-dimensional image-plane grid has been replaced by an
  integration over the regions enclosed by calculated image boundaries.
- A returned finite value is a numerical estimate without a guaranteed error
  bound. If the image boundary or integration region cannot be constructed,
  the public functions return `NaN`.
- Public binary- and triple-lens configuration contains `n_limb` plus advanced
  static scheduler controls; numerical accuracy settings remain internal.
- A100 measurements set the binary radial scheduler default to 64 regions;
  the triple-lens default remains 8 because it was faster for the measured
  trajectory with 888 full solves out of 1000 positions.
- The finite-source point-lens FFTLog implementation is now imported from
  `microjax.fspl`; the former `microjax.fastlens` package name has been
  removed.

### Fixed

- Reduced isolated `NaN` results near binary-lens source boundaries by making
  the angular boundary calculation more robust.

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

## 0.1.1 — paper version

- Archived implementation associated with Miyazaki & Kawahara (2025), ApJ,
  994, 144.
- Software archive: <https://doi.org/10.5281/zenodo.17247892>.
