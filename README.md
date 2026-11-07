# microJAX

<img src="logo/microjax.png" alt="microJAX logo" width="280">

Differentiable binary- and triple-lens microlensing in JAX: finite sources,
limb darkening, light-curve Jacobians, and caustics on CPUs and GPUs.

```bash
python -m pip install microjaxx
```

The package is installed as `microjaxx` and imported as `microjax`.
Enable float64 before creating JAX arrays. For GPU installation, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Single-lens modeling is available in [singlens](https://github.com/ShotaMiyazaki94/singlens).

## Quickstart

Calculate a limb-darkened binary light curve:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from microjax.inverse_ray import mag_binary

times = jnp.linspace(-5., 10., 101)
t0, t_e, u0 = 0., 10., 0.1
alpha = jnp.deg2rad(50.)
w = ((times - t0) / t_e + 1j * u0) * jnp.exp(1j * alpha)

lightcurve = jax.jit(lambda rho: mag_binary(
    w, rho, s=1., q=0.001, u1=0.5, backend="cpu",
))
A = lightcurve(0.005)
```

`w` contains complex source positions in Einstein-radius units; `rho` is
the source radius. The output has one magnification per position.
Use `mag_triple` with `q3`, `r3`, and `psi` for a triple lens.
Both functions accept `backend="cpu"` or `backend="accelerator"` (the default).

Differentiate the source radius with JAX:

```python
dA_drho = jax.jacfwd(lightcurve)(0.005)
```

The [CPU examples](example/cpu/) plot complete light curves and their seven-
or ten-parameter Jacobians. [GPU examples](example/gpu/jacobian-binary/) use
the same events with the accelerator backend.

## Documentation

- [Installation](docs_src/getting_started.rst) and [light curves](docs_src/usage.rst)
- [Automatic differentiation](docs_src/autodiff.rst)
- [Examples](docs_src/examples/index.rst)
- [CPU backend](docs_src/cpu_backend.rst)
- [Numerical accuracy](docs_src/caveats.rst) and [benchmarks](docs_src/benchmarks.rst)
- [API reference](https://shotamiyazaki94.github.io/microjax/modules.html)

Browse the [documentation website](https://shotamiyazaki94.github.io/microjax/)
or the source guides above for the current branch. For source installation,
tests, and documentation builds, see [CONTRIBUTING](CONTRIBUTING.md).

## Citation and license

If you use microJAX, cite Miyazaki & Kawahara (2025), ApJ, 994, 144
([paper](https://doi.org/10.3847/1538-4357/ae1005)) and the
[software archive](https://doi.org/10.5281/zenodo.17247892).
The paper describes `v0.1.1`; the current solver is the `0.2` series.
See the [citation guide](docs_src/citing.rst) for BibTeX.

microJAX is released under the [MIT License](LICENSE).
[Third-party attribution](third_party/README.md) covers code adapted from
`caustics`.
