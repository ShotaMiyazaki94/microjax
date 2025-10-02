Binary-Lens Benchmarks
======================

Side-by-side comparisons between microJAX and `VBBinaryLensing` for uniform
and limb-darkened binary microlensing light curves.

Contents
--------
- `compare_binary_uniform.py`: tracks a uniform source and measures relative
  accuracy and runtime against `VBBinaryLensing.BinaryMag2`.
- `compare_binary_limb_dark.py`: same setup but with a linear limb-darkening
  coefficient `u1 = 0.7`.

How to run
----------
1. Install the optional dependency::

       pip install VBBinaryLensing

2. Execute either script with `python`. The programs JIT-compile the
   microJAX solvers, evaluate the light curve, and export a comparison plot
   (`compare_binary_*.png`).

Both scripts assume double-precision JAX. GPU acceleration is helpful but not
required.

Below is a representative runtime snapshot collected on an NVIDIA A100 GPU::

  $ python example/compare-vbbl/compare_binary_uniform.py
  point-source (microJAX):        0.001 s  (0.001 ms per point)
  hexadecapole (microJAX):        0.102 s  (0.102 ms per point)
  VBBinaryLensing:                1.065 s  (1.065 ms per point)
  microJAX mag_binary (config):   0.245 s  (0.245 ms per point)
  output -> example/compare-vbbl/compare_binary_uniform.png

  $ python example/compare-vbbl/compare_binary_limb_dark.py
  point-source (microJAX):        0.001 s  (0.001 ms per point)
  hexadecapole (microJAX):        0.102 s  (0.102 ms per point)
  VBBinaryLensing:                3.715 s  (3.715 ms per point)
  microJAX mag_binary (config):   0.344 s  (0.344 ms per point)
  output -> example/compare-vbbl/compare_binary_limb_dark.png
