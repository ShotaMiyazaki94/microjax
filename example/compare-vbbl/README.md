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

Below is an example of the execution time on an NVIDIA A100 GPU:

```text
python example/compare-vbbl/compare_binary_uniform.py
computation time: 0.001 sec (0.001 ms per points) for point-source in microjax
computation time: 0.102 sec (0.102 ms per points) for hexadecapole in microjax
computation time: 1.065 sec (1.065 ms per points) with VBBinaryLensing
computation time: 0.245 sec (0.245 ms per points) with microjax mag_binary, 200 chunk_size, 1000 max_full, 500 rbin, 500 thbin
output: example/compare-vbbl/compare_binary_uniform.png

python example/compare-vbbl/compare_binary_limb_dark.py
computation time: 0.001 sec (0.001 ms per points) for point-source in microjax
computation time: 0.102 sec (0.102 ms per points) for hexadecapole in microjax
computation time: 3.715 sec (3.715 ms per points) with VBBinaryLensing
computation time: 0.344 sec (0.344 ms per points) with microjax mag_binary, 200 chunk_size, 1000 max_full, 500 rbin, 500 thbin
output: example/compare-vbbl/compare_binary_limb_dark.png
```
