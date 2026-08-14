<p align="center">
  <img src="logo/microjax.png" width="50%" alt="microJAX logo">
</p>

# microJAX

**Differentiable microlensing models for CPUs and GPUs in JAX.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/built%20with-JAX-blue)](https://github.com/jax-ml/jax)
[![PyPI](https://img.shields.io/pypi/v/microjaxx.svg)](https://pypi.org/project/microjaxx/)
[![DOI](https://zenodo.org/badge/774485090.svg)](https://doi.org/10.5281/zenodo.17247892)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#accuracy-and-limitations)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

microJAX models point and finite sources in single-, binary-, and triple-lens
microlensing systems. It provides an accelerator-oriented boundary integrator,
a separate binary-lens CPU backend, forward-mode automatic differentiation,
caustic calculations, and trajectory utilities.

The PyPI distribution is named `microjaxx`; the Python package is imported as
`microjax`.

## Installation

```bash
python -m pip install microjaxx
```

Install the current source tree for development:

```bash
git clone https://github.com/ShotaMiyazaki94/microjax.git
cd microjax
python -m pip install -e ".[dev]"
```

Install a JAX build appropriate for your CPU, CUDA, or ROCm platform using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The archived methods-paper implementation remains available as
`microjaxx==0.1.1`; the current `0.2` line is a substantial redesign. See the
[changelog](CHANGELOG.md) before comparing results across these versions.

## Quick start

Enable double precision before creating arrays or compiling functions:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from microjax.inverse_ray import mag_binary

w = jnp.asarray([0.10 + 0.20j, 0.60 - 0.20j])
rho, s, q = 0.01, 1.0, 0.3

# Accelerator-oriented backend (default)
mu_accelerator = mag_binary(w, rho, s=s, q=q)

# Binary-lens CPU backend with diagnostics
cpu = mag_binary(
    w,
    rho,
    s=s,
    q=q,
    backend="cpu",
    return_info=True,
)
mu_cpu = cpu.magnification
valid_cpu = cpu.status == 0
```

The first call includes JAX compilation. The CPU backend has a distinct
execution and diagnostic contract; read the CPU guide before using it in an
inference pipeline.

## Documentation

- [Hosted documentation](https://shotamiyazaki94.github.io/microjax/)
- [Getting started](docs/getting_started.html)
- [Usage and public workflows](docs/usage.html)
- [CPU binary-lens backend](docs/cpu_backend.html)
- [Accelerator performance tuning](docs/performance.html)
- [API reference](docs/modules.html)
- [Troubleshooting](docs/troubleshooting.html)
- [Citation and reproducibility](docs/citing.html)

Runnable workflows and their recorded outputs are grouped under
[`example/cpu/`](example/cpu/) and [`example/gpu/`](example/gpu/).

## Accuracy and limitations

microJAX is research software. Returned finite-source values are numerical
estimates without guaranteed error bounds, and difficult configurations may
return `NaN` or a non-zero CPU status. Validate magnifications and derivatives
over the parameter region used in an analysis. The documentation describes
backend-specific failure handling and reproducibility requirements.

## Citation

If you use microJAX, cite Miyazaki & Kawahara (2025), ApJ, 994, 144
([doi:10.3847/1538-4357/ae1005](https://doi.org/10.3847/1538-4357/ae1005))
and the archived software
([doi:10.5281/zenodo.17247892](https://doi.org/10.5281/zenodo.17247892)).
See the [citation guide](docs/citing.html) for BibTeX and version-reporting
requirements.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for test profiles and contribution
guidelines. Bug reports should include a minimal reproducer, the microJAX and
JAX/JAXLIB versions, the execution platform, and whether x64 mode is enabled.

## License

microJAX is distributed under the [MIT License](LICENSE). Third-party code and
attribution are listed in [third_party/README.md](third_party/README.md).
