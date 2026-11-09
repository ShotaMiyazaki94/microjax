# Examples

Compute binary and triple light curves, differentiate event parameters, and
compare magnifications with VBMicrolensing. Run from the repository root:

```bash
python -m pip install -e . matplotlib
```

| Example | Run | Result |
| --- | --- | --- |
| [Binary lens on CPU](cpu/README.md#binary-lens) | `python example/cpu/lightcurve.py` | Light curve and seven-parameter Jacobian |
| [Triple lens on CPU](cpu/README.md#triple-lens) | `python example/cpu/lightcurve.py --lens triple` | Light curve and ten-parameter Jacobian |
| [Binary lens on GPU](gpu/jacobian-binary/README.md) | `python example/gpu/jacobian-binary/code/jacobian.py` | Light curve, Jacobian, and timings |
| [Triple lens on GPU](gpu/jacobian-triple/README.md) | `python example/gpu/jacobian-triple/code/jacobian.py` | Light curve, Jacobian, and timings |
| [HMC injection–recovery](hmc/README.md) | `python example/hmc/injection_recovery.py` | Posterior samples, recovery plots, and sampler diagnostics; see the example for dependencies |
| Magnification benchmarks | [Binary](gpu/compare-binary-vbml/README.md) · [Triple](gpu/compare-triple-vbml/README.md) | VBMicrolensing comparison and image geometry |

The light-curve examples use linear limb darkening with `u1=0.5`. CPU and
GPU examples share their event parameters and observation times. See each
README for figures and output paths, or browse the
[illustrated gallery](../docs_src/examples/index.rst).

GPU runs use a CUDA-enabled JAX installation. The VBMicrolensing comparisons
also need `python -m pip install VBMicrolensing`.
