Triple‑lens Jacobian Example
===========================

This example computes a triple‑lens magnification time series and its full Jacobian
with respect to model parameters. The computation is heavy and relies on repeated
inverse‑ray integrations and automatic differentiation.

Practical note:
- CPU: Runs but can be impractically slow for the default settings.
- GPU: Strongly recommended. Install JAX with CUDA/ROCm and, if needed, set
  `JAX_PLATFORMS=cuda` before running.

Run:
```
python grads_uniform_paper.py
```

Outputs:
- `time_mag.csv`: time and magnification
- `jacobian.npy`: Jacobian array
- `full_jacobian_plot.png`: figure with magnification and sensitivities

