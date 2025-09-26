Triple-lens Jacobian Example
===========================

This directory reproduces the triple-lens Jacobian diagnostic figure from the
paper using [`grads_uniform_paper.py`](grads_uniform_paper.py) (now identical to
the previously separate `full_jac_triple.py`). By default the script reads the
cached data products bundled here and regenerates the plot.

Run::

    python grads_uniform_paper.py

This renders the summary figure to [`full_jac.png`](full_jac.png).

Regenerating the data
---------------------
The heavy numerical workload (inverse-ray integrations + automatic
differentation) is guarded behind an `if (0):` block near the top of the
script. Flip that sentinel to `if True:` to recompute the cached arrays. A
CUDA-enabled JAX install is strongly recommended; CPU-only execution can take
orders of magnitude longer.

Outputs
-------
- [`magnification.csv`](magnification.csv): cached magnification time series.
- [`jacobian_full.npy`](jacobian_full.npy): cached parameter Jacobian (`n_params × n_time`).
- [`full_jac.png`](full_jac.png): magnification and sensitivity panels reproduced from the paper.
