# Changelog

## Unreleased — 0.2.0

This release introduces a new API for binary- and triple-lens modeling.

- Import `mag_binary` and `mag_triple` directly from `microjax` to calculate
  magnification at source positions. Use `make_mag_binary` or `make_mag_triple`
  to choose `backend="cpu"` or `backend="gpu"` once; CPU is the default.
- For event modeling, `make_binary_lightcurve` and `make_triple_lightcurve`
  return `lightcurve(parameters, times, u1=...)`. Binary parameters are
  `[t0, tE, u0, q, s, alpha, rho]`; triples append `[q3, r3, psi]`.
  These fixed-length arrays support JIT compilation and forward Jacobians.
- CPU binary and triple calculations share an adaptive image integrator.
  Uniform and linearly limb-darkened sources use the same calculation, with
  `u1=0` selecting a uniform brightness profile. `return_info=True` adds
  status flags without discarding the calculated magnification.
- GPU calculations integrate image boundaries in batches. `BinaryMagConfig`
  and `TripleMagConfig` control boundary sampling and batching.
- Added CPU and GPU examples with light curves, image geometry, and parameter
  derivatives, plus CPU accuracy and speed maps against VBMicrolensing.
- Removed the old dense-grid, retry, and contour implementations and their
  separate APIs. Numerical internals now live under `microjax._magnification`.
  CPU and GPU implementations have sibling `cpu/` and `gpu/` packages, with
  common geometry and multipole selection alongside the API.
- Single-lens modeling has moved to
  [singlens](https://github.com/ShotaMiyazaki94/singlens).

For the published methods-paper implementation, use the `v0.1.1` Git tag or
`microjaxx==0.1.1`. The API above replaces the previous `inverse_ray` imports
and per-call backend arguments.

## 0.1.1 — paper version

- Archived implementation associated with Miyazaki & Kawahara (2025), ApJ,
  994, 144.
- Software archive: <https://doi.org/10.5281/zenodo.17247892>.
