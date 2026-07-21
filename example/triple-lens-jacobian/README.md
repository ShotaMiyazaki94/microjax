Triple-lens Jacobian Example
============================

This directory is the triple-lens counterpart of
[`example/binary-lens-jacobian`](../binary-lens-jacobian). The
[`grads_uniform_paper.py`](grads_uniform_paper.py) benchmark uses the current
retry-free [`microjax.inverse_ray.lightcurve.mag_triple`](../../src/microjax/inverse_ray/lightcurve.py)
implementation to compute uniform-source magnification and its forward
Jacobian with respect to `t0, tE, u0, q, s, alpha, rho, q3, r3, psi`.

Like the binary example, it reports JIT warm-up separately from the median of
synchronised compiled executions. Forward mode is the intended production AD
path: there are ten scalar inputs and hundreds of light-curve outputs. Reverse
mode is deliberately not included. The triple boundary calculation currently
uses the global polar chart and the retry-free G15/K31 fixed-1 radial rule.

Run
---

The default uses 500 trajectory points and `Nlimb=500`. A CUDA-enabled JAX
installation is strongly recommended:

```console
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  python example/triple-lens-jacobian/grads_uniform_paper.py
```

For a smaller GPU smoke run:

```console
python example/triple-lens-jacobian/grads_uniform_paper.py --quick
```

Use `--repeats` to change the number of compiled timing runs,
`--output-dir` to redirect all products, and `--no-plot` when only numerical
products and timings are needed. Numerical controls are exposed as
`--n-points`, `--n-limb`, `--margin-r`, `--angular-atol`, and
`--relative-tolerance`; source batching remains an internal scheduler detail.

Outputs
-------

- `magnification.csv`: time and magnification columns.
- `jacobian_forward.npy`: forward-mode Jacobian (`n_time x 10`).
- `benchmark.json`: configuration, device, warm-up, and compiled timings.
- `triple_jacobian.png`: magnification, source/lens geometry, and ten
  sensitivity panels.

The older `jacobian_full.npy` and `full_jac.png` files in this directory are
retained paper-era products; the current script does not read or overwrite
them.
