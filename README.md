<p align="center">
  <img width = "500" src="logo/microjax.png"/>
  <br>
</p>

# microjax

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-%F0%9F%A6%81-lightgrey)](https://github.com/google/jax)
![Status](https://img.shields.io/badge/status-alpha-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**microJAX** is a **fully‑differentiable**, **GPU‑accelerated** software for modelling gravitational microlensing light curves produced by **binary**, and **triple** lens systems, using the **image-centered ray shooting (ICRS)** method [(e.g., Bennett 2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1408B/abstract). Written entirely in [JAX](https://github.com/google/jax), it delivers millisecond‑level evaluations of extended-source magnifications *and* exact gradients for every model parameter through the use of [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html), enabling gradient‑based Bayesian inference workflows such as Hamiltonian Monte Carlo (HMC) and variational inference. 


---

## ✨ Key Features
| Category | Highlights |
|----------|------------|
| **Lens types** | Point‑source and finite‑source magnifications for binary and triple lenses |
| **Extended sources** | Uniform & limb‑darkened source profiles |
| **Core engine** | Modified Image‑Centred Ray Shooting (ICRS) rewritten in JAX |
| **Root solver** | Differentiable Ehrlich-Aberth polynomial solver with [implicit gradients](http://implicit-layers-tutorial.org/implicit_functions/) |
| **Inference ready** | Drop‑in likelihood for NumPyro HMC & VI pipelines |


## 📦 Installation

```bash
# clone the repository
git clone https://github.com/ShotaMiyazaki94/microjax.git
cd microjax

# editable install with all extras (GPU/TPU support depends on your JAX wheel)
pip install -e .[dev]
```

> **Note** : microJAX is in active development and not yet on PyPI.  API changes may still
> occur before the first stable (v1.0) release.

---

## 📚 References
* [Miyazaki & Kawahara (in prep.)](): `microjax` paper (expected within 2025!)
* [Bennett (2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1408B/abstract): Image-centred ray shooting (ICRS) method   
* [Cassan (2017)](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true): Quadrupole & hexadecapole approximations
* [Sugiyama (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...937...63S/abstract): Fast FFT-based magnification evaluation with a single-lens extended source model

- `microJAX` paper coming soon (assumed within 2025)!
- [FFT based evaluation of microlensing magnification with extended source](https://ui.adsabs.harvard.edu/abs/2022ApJ...937...63S/abstract)
- [Fast computation of quadrupole and hexadecapole approximations in microlensing with a single point-source evaluation](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true)

## 🤝 Contributing

Pull requests are welcome!  Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding style, test suite, and CI guidelines.  Bug reports can be filed via GitHub Issues.

## 📜 License

This project is licensed under the [MIT License](LICENSE).  If you use `microJAX` in academic work, please cite the upcoming Miyazaki et al. (2025) methods paper.

---