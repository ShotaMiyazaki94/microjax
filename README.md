<p align="center">
  <img width = "500" src="logo/microjax.png"/>
  <br>
</p>

# microjax

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-%F0%9F%A6%81-lightgrey)](https://github.com/google/jax)
![Status](https://img.shields.io/badge/status-alpha-orange)
![License](https://img.shields.io/badge/license-MIT-green)

`microjax` is a code for computing microlensing light curves of single, binary, and triple lens systems using the image-centered inverse-ray shooting (ICIRS) method [(Bennett 2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1408B/abstract). 

It is built using the [JAX](https://github.com/google/jax) library which enables the computation of *exact* gradients of the code outputs with respect to all input parameters through the use of [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html). 

## Installation
`microJAX` is still being actively developed and is not yet released on PyPI. To install the development version, clone this repository, and run
```python
pip install -e .
```
## Features
- Fast (miliseconds) and accurate computation of binary and triple lens microlensing light curves for extended uniform and limb-darkened sources.
- Automatic differentiation enables the use of gradient-based inference methods such as Hamiltonian Monte Carlo when fitting multiple lens microlensing light curves.
- A differentiable JAX version of a complex polynomial root solver which uses the Aberth-Ehrlich method to obtain all roots of a complex polynomial at once using an implicit deflation strategy. The gradient of the solutions with respect to the polynomial coefficients is obtained through [implicit differentiation](http://implicit-layers-tutorial.org/implicit_functions/).
- Hexadecapole approximation from [Cassan 2017](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true) is used to substantially speed up the computation of the magnification everywhere except near the caustics.

## References
- `microJAX` paper coming soon (assumed within 2025)!
- [FFT based evaluation of microlensing magnification with extended source](https://ui.adsabs.harvard.edu/abs/2022ApJ...937...63S/abstract)
- [VBBINARYLENSING: a public package for microlensing light-curve computation](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract)
- [Fast computation of quadrupole and hexadecapole approximations in microlensing with a single point-source evaluation](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true)