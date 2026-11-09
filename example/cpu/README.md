# CPU light curves and Jacobians

These examples show how the lens geometry produces a light curve, and how
that curve responds to each event parameter. Run them from the repository
root after installing microJAX and Matplotlib:

```bash
python example/cpu/lightcurve.py
python example/cpu/lightcurve.py --lens triple
```

Install with `python -m pip install -e . matplotlib`.
[`lightcurve.py`](lightcurve.py) creates a `make_binary_lightcurve` or
`make_triple_lightcurve` function with `backend="cpu"`, then passes an event
parameter dictionary. It uses the CPU device and float64, and computes
magnification and all parameter derivatives in one `jax.jacfwd` call.
Linear limb darkening is fixed at `u1=0.5`.

## Binary lens

The default evaluates 500 times from −5 to 10 days. The event has `t0=0`,
`tE=10`, `u0=0.1`, `q=0.001`, `s=1`, `alpha=50°`, and `rho=0.005`.
The seven Jacobian columns follow `t0, tE, u0, q, s, alpha, rho`.

![Binary light curve, caustics, source trajectory, and seven parameter derivatives](lightcurve.png)

The upper panel shows magnification with the caustics, critical curves, and
source trajectory in the inset. The lower panels show the seven derivatives.
The [GPU binary example](../gpu/jacobian-binary/README.md) uses the same event.

## Triple lens

`--lens triple` evaluates 1000 times over the same interval. The event has
`t0=0`, `tE=10`, `u0=0.1`, `q=0.1`, `s=1.1`, `alpha=50°`, `rho=0.01`,
and `q3=0.01`. The third lens is at `0.3+1.2j` relative to the midpoint of
the first two lenses: `r3=abs(0.3+1.2j)`, `psi=angle(0.3+1.2j)`.
The ten Jacobian columns follow `t0, tE, u0, q, s, alpha, rho, q3, r3, psi`.

![Triple light curve, caustics, source trajectory, and ten parameter derivatives](triple_jacobian.png)

The [GPU triple example](../gpu/jacobian-triple/README.md) uses the same event.
Edit `params` and `times` in `main()` to explore another trajectory.

## Outputs

Each run writes `lightcurve.png`, `lightcurve.csv` (time and magnification),
and `jacobian_forward.npy`:

| Model | Output directory | Jacobian shape |
| --- | --- | --- |
| Binary | `results/cpu/example/` | 500 × 7 |
| Triple | `results/cpu/example/triple/` | 1000 × 10 |

Load the saved arrays in a notebook to explore the results without
repeating the calculation.

For accuracy and timing comparisons, see the
[benchmarks](../../docs_src/benchmarks.rst).
