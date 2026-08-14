# FSPL FFT vs VBBL comparison (JAX)

Minimal example to benchmark the JAX FFT‑based finite‐source point‑lens (FSPL)
implementation (`microjax.fspl.fspl_*`) against **VBBinaryLensing** (VBBL).
It produces magnification curves and relative residuals along a Paczynski track
for a grid of source sizes.

## Requirements

- `microjax` (this repo, editable install recommended)
- `VBBinaryLensing` (`pip install VBBinaryLensing`)
- `matplotlib`
- `jax`, `jaxlib` (CPU is fine)

## How to run

You can run it from anywhere (no need to `cd`):

```bash
python example/gpu/fspl_fft/compare_fspl_vbbl.py
```

This generates `fspl_vs_vbbl.png` alongside the script (in
`example/gpu/fspl_fft`).
If VBBL is not installed, the script will print a short message and exit cleanly. JAX is
forced to CPU for consistent timing output; per-ρ timings and speed ratios vs.
VBBL are printed in milliseconds.

## What it does

- Compares **uniform disk** and **linear limb‑darkening** (`a1=0.2,0.5,0.8`).
- ρ grid: `1e-3, 1e-2, 1e-1, 1.0, 3.0` (default; editable at top of script)
- Time grid: `t/tE` in `[-3, 3]` with 1000 points, u(t)=sqrt(u0^2 + (t/tE)^2), default `u0=0.0` (set >0 to avoid u=0 if desired)
- Plots A(t) (top, log y) and relative residuals vs VBBL (bottom, log y); 1% line shown.
- Logs per-ρ runtimes (ms) for VBBL and FSPL and their ratios.

## Sample output

![fspl_vs_vbbl](fspl_vs_vbbl.png)
