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
- Parallax and binary orbital-motion trajectory utilities.
- Binary and triple Jacobian examples and VBBinaryLensing comparison records.
- A detailed Japanese implementation report covering the complete algorithm,
  numerical integration, differentiation, and failure behavior.

### Changed

- The recommended finite-source imports now come from `microjax.inverse_ray`.
- The paper-era two-dimensional image-plane grid has been replaced by an
  integration over the regions enclosed by calculated image boundaries.
- A returned finite value is a numerical estimate without a guaranteed error
  bound. If the image boundary or integration region cannot be constructed,
  the public functions return `NaN`.
- GPU batching, integration safety margins, and internal error diagnostics are
  now selected by the solver instead of being user settings.
- Public configuration now contains only ``n_limb``. The former
  ``relative_tolerance``, ``angular_atol``, ``margin_r``, and
  ``parallel_regions`` settings are internal implementation details.

### Compatibility

- Treat `0.2.0` as potentially API-breaking relative to `0.1.1`.
- Only names imported from `microjax.inverse_ray` are part of the recommended
  finite-source API. Other modules under that package may change during the
  `0.x` series.
- Reproduction of the published methods-paper implementation should use the
  `v0.1.1` Git tag or the `microjaxx==0.1.1` distribution.

## 0.1.1 — paper version

- Archived implementation associated with Miyazaki & Kawahara (2025), ApJ,
  994, 144.
- Software archive: <https://doi.org/10.5281/zenodo.17247892>.
