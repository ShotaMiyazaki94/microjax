# Contributing to microJAX

Include a minimal reproducing example, the microJAX and JAX/JAXLIB versions,
the execution platform, and whether 64-bit mode is enabled when reporting a bug.

For code changes, branch from `develop`, keep the implementation focused, run
checks relevant to the change, and describe the behavior and validation in the
pull request. Add a regression test only when it protects a meaningful contract
or reproduces a concrete failure.

## CPU development

The CPU backend has one implementation and one focused test file:

```bash
pytest -q tests/unittests/inverse_ray/test_cpu.py
```

Use the critical benchmark when changing CPU numerical behavior or performance:

```bash
python tools/cpu_benchmark.py --output results/cpu-benchmark.json
```

It compares linear limb-darkened values (``u1=0.5``) with fixed references and
times a forward Jacobian, checking only its finiteness. It reports synchronized
warm runtime with compilation excluded. Independent numerical AD checks and
the uniform coefficient limit remain in the focused CPU tests.
Compare before and after on the same machine and inputs. Frozen references
have numerical limitations; their small set is a regression screen rather
than a universal accuracy claim. Extend it only for a concrete uncovered risk.

Keep measurements and generated figures under `results/`, and any retired local
experiments under `archive/`. Both directories are excluded from version control.
Commit source, focused tests and their small required fixtures; do not commit
measurement outputs, generated HTML, caches or machine-specific files. Historical
CPU implementations are available in Git history.

The optional regional benchmark requires a separately supplied dense LD reference
CSV and its same-stem JSON metadata; these inputs are not distributed. It also
requires VBMicrolensing, and plotting requires Matplotlib:

```bash
python tools/cpu_benchmark_grid.py --reference-data /path/to/xdense_cpu.csv --output-dir results/cpu/new-run/data --time-budget 300
python tools/plot_cpu_benchmark.py --data-dir results/cpu/new-run/data --output-dir results/cpu/new-run/figures
```

## Other checks

Run the regular package suite when changing shared components:

```bash
pytest -q
```

GPU-specific tests require a CUDA-capable JAX installation:

```bash
pytest -c pytest-gpu.ini -q
```

The regular suite excludes tests marked `slow`. Run an individual slow test
only when its numerical case is relevant to the change. Broad grids and
repeated full-suite runs are not part of the routine CPU development loop.
