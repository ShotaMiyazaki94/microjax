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

## Features
- Fast (miliseconds) and accurate computation of binary and triple lens microlensing light curves for extended uniform and limb-darkened sources.
- Automatic differentiation enables the use of gradient-based inference methods such as Hamiltonian Monte Carlo when fitting multiple lens microlensing light curves.
- A differentiable JAX version of a complex polynomial root solver which uses the Aberth-Ehrlich method to obtain all roots of a complex polynomial at once using an implicit deflation strategy. The gradient of the solutions with respect to the polynomial coefficients is obtained through [implicit differentiation](http://implicit-layers-tutorial.org/implicit_functions/).
- Hexadecapole approximation from [Cassan 2017](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true) is used to substantially speed up the computation of the magnification everywhere except near the caustics.

## 📚 References
* [Bennett 2010](https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1408B/abstract): Image-centred ray shooting (ICRS) method   
* [Cassan 2017](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true): Quadrupole & hexadecapole approximations 

- `microJAX` paper coming soon (assumed within 2025)!
- [FFT based evaluation of microlensing magnification with extended source](https://ui.adsabs.harvard.edu/abs/2022ApJ...937...63S/abstract)
- [Fast computation of quadrupole and hexadecapole approximations in microlensing with a single point-source evaluation](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true)

## 🤝 Contributing

Pull requests are welcome!  Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding style, test suite, and CI guidelines.  Bug reports can be filed via GitHub Issues.

## 📜 License

This project is licensed under the [MIT License](LICENSE).  If you use `microJAX` in academic work, please cite the upcoming Miyazaki et al. (2025) methods paper.

---