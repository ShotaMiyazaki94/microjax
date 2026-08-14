Triple-lens Jacobian Example
============================

This directory is the triple-lens counterpart of
[`example/jacobian-binary`](../jacobian-binary). The
[`grads_uniform_triple.py`](grads_uniform_triple.py) benchmark uses the current
[`microjax.inverse_ray.mag_triple`](../../src/microjax/inverse_ray/lightcurve.py)
function to compute uniform-source magnification and its forward Jacobian with
respect to `t0, tE, u0, q, s, alpha, rho, q3, r3, psi`. Here “Jacobian” means
the derivative of every light-curve point with respect to each of these ten
parameters.

Like the binary example, it reports JIT warm-up separately from the median of
synchronised compiled executions. Forward mode is the intended production AD
path: there are ten scalar inputs and hundreds of light-curve outputs. Reverse
mode is deliberately not included. The calculation automatically changes its
coordinate origin for small isolated images when that improves their angular
resolution; no user setting is required.

Run
---

The default uses 500 trajectory points and `n_limb=500`. A CUDA-enabled JAX
installation is strongly recommended:

```console
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  python example/jacobian-triple/grads_uniform_triple.py
```

For a smaller GPU smoke run:

```console
python example/jacobian-triple/grads_uniform_triple.py --quick
```

Use `--repeats` to change the number of compiled timing runs,
`--output-dir` to redirect all products, and `--no-plot` when only numerical
products and timings are needed. The workload controls exposed here are
`--n-points` and `--n-limb`. Other numerical and GPU execution choices are
selected automatically by microJAX.

Outputs
-------

- `magnification.csv`: time and magnification columns.
- `jacobian_forward.npy`: forward-mode Jacobian (`n_time x 10`).
- `benchmark.json`: configuration, device, warm-up, and compiled timings.
- `triple_jacobian.png`: magnification, source/lens geometry, and ten
  sensitivity panels.
