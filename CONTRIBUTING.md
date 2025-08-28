# How to contribute to microJAX

## Branch model and pull-request (PR)
We adopt a simple git-flow model, consisting of master, develop, contributor-defined, and release branches. PR by contributors via a folked repository is always welcome.

- master: stable branch. Once a set of functions is implemented in the **develop** branch, the maintainer (@ShotaMiyazaki94) will merge them to the master branch through the release process (release branch).

- develop: default branch for developers. PR by contributors will be merged to this branch, after the review process.

- contributor-defined branches: The contirbutors make their own branch originated from the develop branch and after developing, we send a PR to the develop branch. The branch name is like the followings: feature/hminus_opacity, bugfix/hminus, etc.

- release: we make the release branch prior to a new version release. The contributors are expected to review the release branch.  

## Issues and Discussion

## Tests

### CPU vs GPU tests

The test suite includes both CPU-only tests and GPU-only tests for inverse_ray functionality. GPU tests are designed for NVIDIA A100 and are skipped by default.

- CPU tests: simply run `pytest -q`.
- GPU tests (A100 only): set an env var and select the marker:

```
export MICROJAX_GPU_TESTS=1
pytest -m gpu -q
```

The GPU tests will run only if:
- `MICROJAX_GPU_TESTS=1` is set, and
- JAX detects a CUDA device whose `device_kind` contains `A100`.

### A100 CI runner setup (self-hosted)

If you want to run GPU tests in CI, set up a self-hosted GitHub Actions runner on an A100 machine and label it, e.g., `self-hosted, gpu, a100`. An example workflow is provided in `.github/workflows/gpu-tests.yml`.

Checklist for the runner machine:
- Install NVIDIA drivers and CUDA compatible with your JAX build.
- Install JAX with CUDA support and verify `python -c "import jax; print(jax.devices('cuda'))"` lists an A100 device.
- Optional dependencies (if used in tests): `VBBinaryLensing`, `VBMicrolensing`, `astropy`.
- Ensure environment variables are set for CI job:
  - `MICROJAX_GPU_TESTS=1`
  - `JAX_PLATFORMS=cuda`

With the runner online, you can trigger the provided workflow manually or on a schedule. The workflow runs only tests marked `gpu`.
