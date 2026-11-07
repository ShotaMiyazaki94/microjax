# CPU light curves and Jacobians

Run from the repository root:

```bash
python -m pip install -e . matplotlib
python example/cpu/lightcurve.py
python example/cpu/lightcurve.py --lens triple
```

[`lightcurve.py`](lightcurve.py) uses the dedicated `backend="cpu"` integration
for both lens models, with float64 and the CPU device selected explicitly.
It obtains the light curve and its Jacobian together using `jax.jacfwd`.
The first evaluation includes JAX compilation. Linear limb darkening is fixed
at `u1=0.5` in these examples; the CPU backend also accepts dynamic `u1`, and
`u1=0` uses the same implementation.

Both examples match the parameters, sampling and plot layout of their
corresponding GPU limb-darkened examples. The upper panel shows magnification
with a caustic, critical-curve and source-trajectory inset; the lower panels
show forward derivatives. The figures here are calculated by the CPU backend.

## Binary lens

The default reproduces the
[GPU binary example](../gpu/jacobian-binary/outputs/limb_dark/binary_limb_dark_jacobian.png):
500 times from −5 to 10 days, `t0=0`, `tE=10`, `u0=0.1`, `q=0.001`, `s=1`,
`alpha=50°`, and `rho=0.005`. The seven Jacobian columns follow
`t0, tE, u0, q, s, alpha, rho`.

![CPU binary-lens magnification, lens geometry, and seven forward-Jacobian panels](lightcurve.png)

## Triple lens

`--lens triple` reproduces the
[GPU triple example](../gpu/jacobian-triple/outputs/limb_dark/triple_limb_dark_jacobian.png):
1000 times from −5 to 10 days, `t0=0`, `tE=10`, `u0=0.1`, `q=0.1`, `s=1.1`,
`alpha=50°`, `rho=0.01`, and `q3=0.01`. The third lens has midpoint-frame
position `0.3+1.2j`, encoded as `r3=abs(0.3+1.2j)` and
`psi=angle(0.3+1.2j)`. The ten Jacobian columns follow
`t0, tE, u0, q, s, alpha, rho, q3, r3, psi`.

![CPU triple-lens magnification, lens geometry, and ten forward-Jacobian panels](triple_jacobian.png)

Edit the parameters and time grid in `main()` to explore another event.
The finite-source CPU models require positive mass ratios and source radius,
with distinct lens positions.

## Outputs

Each run writes `lightcurve.png`, `lightcurve.csv` (time and magnification),
and `jacobian_forward.npy`:

| Model | Runtime output directory | Jacobian shape |
| --- | --- | --- |
| Binary | `results/cpu/example/` | 500 × 7 |
| Triple | `results/cpu/example/triple/` | 1000 × 10 |

Runtime outputs stay local and are excluded from Git. Only these two
representative README previews are versioned; refresh them after running:

```bash
cp results/cpu/example/lightcurve.png example/cpu/lightcurve.png
cp results/cpu/example/triple/lightcurve.png example/cpu/triple_jacobian.png
```

These are usage examples. Accuracy and speed measurements live in
[`tools/cpu_benchmark.py`](../../tools/cpu_benchmark.py); regional binary maps
and uncertainty plots are described in [CONTRIBUTING.md](../../CONTRIBUTING.md).
