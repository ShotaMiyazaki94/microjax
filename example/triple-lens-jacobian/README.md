Triple-lens Jacobian Example
===========================

Reproduces the triple-lens Jacobian diagnostic figure from the paper. By
default the script loads the pre-generated magnification samples and Jacobian
stored in `time_mag.csv` and `jacobian.npy`, then redraws the summary plot.

Run::

    python grads_uniform_paper.py

This produces `full_jacobian_plot.png`.

Regenerating the data
---------------------
Set the guarded block in `grads_uniform_paper.py` from `if(0):` to
`if True:` (or simply edit the script) to recompute the light curve and
Jacobian with microJAX. That path performs heavy inverse-ray integrations and
automatic differentiation at every time stamp. GPU acceleration is strongly
recommended when regenerating the arrays; install the CUDA-enabled JAX build
and export `JAX_PLATFORMS=cuda` if necessary. Expect the CPU version to be
orders of magnitude slower.

Outputs
-------
- `time_mag.csv`: cached time and magnification samples.
- `jacobian.npy`: cached Jacobian tensor (`n_time × n_params`).
- `full_jacobian_plot.png`: magnification and parameter sensitivities.
