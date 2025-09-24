<p align="center">
  <img src="logo/microjax.png" width="50%">
</p>

**microJAX is a GPU-accelerated, differentiable microlensing modeling library written in JAX.**

# microJAX

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/built_with-JAX-blue)](https://github.com/google/jax)
[![PyPI](https://img.shields.io/pypi/v/microjaxx.svg)](https://pypi.org/project/microjaxx/)
![Status](https://img.shields.io/badge/status-alpha-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**microJAX** is a **fully‑differentiable**, **GPU‑accelerated** software for modelling gravitational microlensing light curves produced by **binary**, and **triple** lens systems, using the **image-centered ray shooting (ICRS)** method [(e.g., Bennett 2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1408B/abstract). Written entirely in [JAX](https://github.com/google/jax), it delivers millisecond‑level evaluations of extended-source magnifications *and* exact gradients for every model parameter through the use of [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html), enabling gradient‑based Bayesian inference workflows such as Hamiltonian Monte Carlo (HMC) and variational inference.

This software is under active development and not yet feature complete.

---

## ✨ Key Features

| Category                | Description                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Lens Systems**        | Supports point-source and finite-source magnification calculations for binary and triple lens systems                    |
| **Extended Sources**    | Models uniform and limb-darkened source profiles                                                     |
| **Computational Core**  | Implements the Image-Centered Ray Shooting (ICRS) algorithm in JAX, fully optimized for GPU acceleration                 |
| **Root-Finding Engine** | Uses a differentiable Ehrlich-Aberth method for complex polynomial roots with [implicit gradients](http://implicit-layers-tutorial.org/implicit_functions/) for stable optimization |
| **Bayesian Inference**  | Provides a ready-to-use likelihood function compatible with NumPyro's HMC and variational inference frameworks           |

## 📦 Installation

From PyPI (recommended):

```bash
pip install microjaxx
```

Notes:

- PyPI package name: `microjaxx`
- Python import name: `microjax`

```python
import microjax
```

Development install (from source):

```bash
# clone the repository
git clone https://github.com/ShotaMiyazaki94/microjax.git
cd microjax
pip install -e ".[dev]"
```

GPU support: JAX/JAXLIB with CUDA/ROCm depends on your environment. Please follow the official JAX installation guide to install the appropriate `jaxlib` for your accelerator:

- JAX installation (CPU/GPU): https://jax.readthedocs.io/en/latest/installation.html

---

## 🚀 Quickstart

Compute an extended-source binary-lens magnification light curve using the image-centered ray shooting (ICRS) method:

Note: `mag_binary` also works on CPU but is very slow.

```python
import jax
import jax.numpy as jnp
from microjax.inverse_ray.lightcurve import mag_binary
jax.config.update("jax_enable_x64", True)

# Binary-lens parameters
s, q = 1.0, 0.01            # separation and mass ratio (m2/m1)
rho = 0.02                  # source radius (Einstein units)
tE, u0 = 30.0, 0.0          # Einstein time [days], impact parameter
alpha = jnp.deg2rad(10.0)  # trajectory angle
t0 = 0.0

N_points = 1000
t = t0 + jnp.linspace(-2*tE, 2*tE, N_points)
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 =  u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

# Extended-source magnification (binary lens)
mu = mag_binary(w_points, rho, s=s, q=q)
```

For point-source magnification, use:

Note: `mag_point_source` runs on CPU (and GPU), so it works without a GPU.

```python
from microjax.point_source import mag_point_source
mu_point = mag_point_source(w_points, nlenses=2, s=s, q=q)
```

---

## Example output

| Visualization of the ICRS method (binary-lens) | Triple-lens magnification and its gradients | Compare with VBBL (binary uniform source) |
| --- | --- | --- |
| ![ICRS](example/visualize-icrs/visualize_example.png) | ![Triple-lens](example/triple-lens-jacobian/full_jacobian_plot.png) | ![Compare VBBL](example/compare-vbbl/compare_binary_uniform.png) |

Refer to the [example](example/) directory for code that creates these plots.

Note: Finite-source calculation with microJAX is extremely slow without a GPU. So, the latter two examples are significantly slower on the CPU.
---

## ⚠️ Known Limitations

- Triple-lens hexadecapole/ghost-image test is not yet implemented: triple-lens calculations fall back to full contour integration everywhere, which can be substantially slower.
- Finite-source magnification trades memory/runtime for accuracy through resolution parameters; tune these settings to match your GPU's available memory and throughput.
- GPU tests are opt-in; run them explicitly with `pytest -m gpu`. If JAX cannot see a CUDA GPU, those tests are skipped.
- For numerical stability and agreement across libraries, enable 64-bit precision in JAX (`jax_enable_x64=True`).

## 📚 References
* [Miyazaki & Kawahara (in prep.)](): `microjax` paper (expected within 2025!)
* [Bennett (2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1408B/abstract): Image-centred ray shooting (ICRS) method   
* [Cassan (2017)](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true): Hexadecapole approximations
* [Sugiyama (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...937...63S/abstract): Fast FFT-based magnification evaluation with a single-lens extended source model

## 🤝 Contributing

Pull requests are welcome!  Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding style, test suite, and CI guidelines.  Bug reports can be filed via GitHub Issues.

### Running Tests

CPU-only tests:

```
pytest -q
```

GPU-only tests are opt-in and skipped by default. To run them on a CUDA-capable machine:

```
# optionally: export JAX_PLATFORMS=cuda
pytest -m gpu -q
```

These tests require JAX to detect a CUDA device. If not available, they are skipped. No additional environment flag is required beyond the `-m gpu` marker.

## 📜 License

This project is licensed under the [MIT License](LICENSE).  If you use `microJAX` in academic work, please cite the upcoming Miyazaki et al. (2025) methods paper.

---
