CPU Binary-lens Jacobian Example
================================

This is the CPU-backend counterpart of
[`example/gpu/jacobian-binary`](../../gpu/jacobian-binary). It preserves the
same trajectories and differentiates every light-curve sample with respect to
`t0, tE, u0, q, s, alpha, rho`. The only model-level change is the finite-source
call: this version uses
`mag_binary(..., backend="cpu", return_info=True)`.

The CPU ICRS scheduler has data-dependent sequential loops, so this example
uses forward-mode AD (`jax.jacfwd`) exclusively. Reverse-mode AD is not part of
the CPU API. `n_limb` is deliberately absent: it does not control CPU accuracy.
The comparison threshold is configured only for external validation; it does
not alter the fixed production CPU route.

Run
---

The full uniform-source example uses the 1,000-point comparison trajectory:

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

Both scripts require finite magnifications, finite forward derivatives, and a
zero CPU status for every sample. No numerical differentiation is performed.

Measured result
---------------

The bundled 1,000-point products were generated on the Apple CPU reported as
`TFRT_CPU_0` by JAX 0.8.0. Timings are medians of three compiled calls:

| source | magnification | forward Jacobian | Jacobian / value |
| --- | ---: | ---: | ---: |
| uniform | 0.213 s | 0.433 s | 2.03x |
| linear LD (`u1=0.5`) | 0.0617 s | 0.126 s | 2.05x |

All 1,000 samples in both runs had status zero. These timings are specific to
this machine, trajectory, JAX version, and tolerance. Compile-plus-first-call
times were 6.26 s and 15.00 s for uniform value/Jacobian, versus 2.27 s and
5.55 s for linear LD.

Outputs
-------

Python sources are kept in `code/`. Generated products are written under
`outputs/uniform/` and `outputs/limb_dark/` respectively.

- `magnification.csv`: time and magnification.
- `jacobian_forward.npy`: `n_time x 7` forward-mode Jacobian.
- `benchmark.json`: device, configuration, CPU tiers/statuses, timings, and
  forward/value runtime ratio.
- `binary_jacobian.png` (or `binary_limb_dark_jacobian.png`): magnification,
  caustic geometry, and all seven sensitivities.
- `ad_benchmark.png`: compile-plus-first and steady-state CPU timings.

All files can be redirected with `--output-dir`; use `--no-plot` for numerical
products only.
