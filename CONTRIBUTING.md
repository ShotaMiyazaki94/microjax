# Contributing to microJAX

Bug reports, documentation improvements, and pull requests are welcome.

Before opening an issue, please check whether the problem is already reported.
Include a minimal reproducing example, the microJAX and JAX/JAXLIB versions,
the execution platform, and whether 64-bit mode is enabled.

For a code contribution:

1. Create a branch from the current default branch.
2. Keep the change focused and add or update tests when behavior changes.
3. Run the relevant tests locally.
4. Open a pull request describing the motivation, user-visible effect, and
   validation performed.

## Tests

For a compact CPU development loop, run:

```bash
pytest -c pytest-cpu-fast.ini -q
```

Run the default test suite with:

```bash
pytest -q
```

The default suite excludes long-running numerical regression tests. Run those
checks explicitly when changing the finite-source integration algorithm:

```bash
pytest -m slow -q
```

GPU-specific tests require a CUDA-capable JAX installation:

```bash
pytest -c pytest-gpu.ini -q
```

Numerical changes should include an independent comparison or regression case
appropriate to the affected calculation. Performance measurements should
report the device, precision, JAX/JAXLIB versions, input size, and whether
compilation time is included.
