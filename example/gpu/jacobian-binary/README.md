# Binary-lens Jacobian on GPU

This example calculates a binary-lens light curve and shows how it changes
with each event parameter. Run it from the repository root with microJAX,
Matplotlib, and GPU-enabled JAX installed:

```bash
python example/gpu/jacobian-binary/code/grads_limb_dark_binary.py
```

The calculation uses `jax.jacfwd` with limb darkening fixed at `u1=0.5`.

The event has `t0=0`, `tE=10` days, `u0=0.1`, `q=0.001`, `s=1`,
`alpha=50°`, and `rho=0.005`. There are 500 observations from −5 to 10 days.
Jacobian columns follow `t0, tE, u0, q, s, alpha, rho`.

![Binary-lens light curve, image geometry, and seven forward derivatives](outputs/limb_dark/binary_limb_dark_jacobian.png)

The [CPU example](../../cpu/README.md#binary-lens) evaluates the same event
with `make_binary_lightcurve(backend="cpu")` and produces the same
panel layout.

## Changing the example

Use `--quick` for a smaller run, `--u1` to change limb darkening, or
`--n-points` to change the time sampling. Run the script with `--help` for
all options.

The light curve, Jacobian array, figures, and timing report are saved under
`outputs/limb_dark/`. Set `--output-dir PATH` to use another directory.
The timing report separates the first call, which includes compilation,
from subsequent calls.

For magnification accuracy, see the
[VBMicrolensing comparison](../compare-binary-vbml/README.md).
