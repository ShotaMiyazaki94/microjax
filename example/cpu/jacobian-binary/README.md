CPU Binary-lens Jacobian Example
================================

This is the CPU-backend counterpart of
[`example/gpu/jacobian-binary`](../../gpu/jacobian-binary). It differentiates
every light-curve sample with respect to `t0, tE, u0, q, s, alpha, rho`.
Its trajectory and point count can differ from the GPU example; compare timings
only after matching the configuration and measurement procedure. The finite-source
call uses
`mag_binary(..., backend="cpu", return_info=True)`.

The CPU ICRS scheduler has data-dependent sequential loops, so this example
uses forward-mode AD (`jax.jacfwd`) exclusively. Reverse-mode AD is not part of
the CPU API. `n_limb` is deliberately absent: it does not control CPU accuracy.
The comparison threshold is configured only for external validation; it does
not alter the fixed production CPU route.

Run
---

The uniform-source example uses 500 points by default:

```console
python example/cpu/jacobian-binary/code/grads_uniform_binary.py
```

The linearly limb-darkened counterpart fixes `u1=0.5` by default:

```console
python example/cpu/jacobian-binary/code/grads_limb_dark_binary.py
```

For a 24-point smoke test, pass `--quick`. The first invocation measures JIT
compilation plus execution; the reported steady-state value is the median of
three subsequent synchronised calls. Change the sample count and repeats with
`--n-points` and `--repeats`.

Both scripts require finite magnifications and forward derivatives. Nonzero
CPU statuses are printed as diagnostic warnings. No numerical differentiation
is performed; finite derivatives alone do not validate their accuracy.

Outputs
-------

Python sources are kept in `code/`. Running the examples generates local products
under `outputs/uniform/` and `outputs/limb_dark/` respectively. Regenerate them
to measure the implementation in your environment; existing generated files
may belong to an earlier solver version.

- `magnification.csv`: time and magnification.
- `jacobian_forward.npy`: `n_time x 7` forward-mode Jacobian.
- `benchmark.json`: device, configuration, CPU routes/statuses, timings, and
  forward/value runtime ratio.
- `binary_jacobian.png` (or `binary_limb_dark_jacobian.png`): magnification,
  caustic geometry, and all seven sensitivities.
- `ad_benchmark.png`: compile-plus-first and steady-state CPU timings.

All files can be redirected with `--output-dir`; use `--no-plot` for numerical
products only.
