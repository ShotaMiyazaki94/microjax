<p align="center">
  <img src="logo/microjax.png" width="50%">
</p>

**microJAX is a GPU-accelerated, differentiable microlensing modeling library written in JAX.**

# microjax

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

Import name remains:

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

GPU support: JAX/JAXLIB with CUDA/ROCm depends on your environment. Please follow the official JAX installation guide to install the appropriate `jaxlib` for your accelerator.

---

## Example output

| Visualization of the ICRS method (binary-lens) | Triple-lens magnification and its gradients |
| --------------------------------------- | --------------------------------------------- |
| ![ICRS](example/visualize-icrs/visualize_example.png) | ![Triple-lens](example/triple-lens-jacobian/full_jacobian_plot.png) |

Refer to the [example](example/) directory for code that creates these plots.

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

GPU-only (A100) tests are opt-in and skipped by default. To run them on an A100 machine:

```
export MICROJAX_GPU_TESTS=1
pytest -m gpu -q
```

These tests require JAX to detect an NVIDIA A100 (CUDA) device. If not available or the env var is not set, they are skipped.

## 📜 License

This project is licensed under the [MIT License](LICENSE).  If you use `microJAX` in academic work, please cite the upcoming Miyazaki et al. (2025) methods paper.

---
