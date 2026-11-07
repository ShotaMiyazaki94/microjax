# Contributing to microJAX

## Source installation

```bash
git clone --branch develop https://github.com/ShotaMiyazaki94/microjax.git
cd microjax
python -m pip install -e '.[dev]'
```

Create a branch from `develop` for each change. Install Matplotlib to run the
figure examples:

```bash
python -m pip install matplotlib
python example/cpu/lightcurve.py
```

## Tests and benchmarks

Run the tests relevant to the change. The shared binary/triple CPU integrator
has seven focused checks:

```bash
pytest -q tests/unittests/inverse_ray/test_cpu.py tests/unittests/inverse_ray/test_cpu_triple.py
```

For changes to CPU numerical behavior or performance, run the small
limb-darkened benchmark on the same machine before and after the change:

```bash
python tools/cpu_benchmark.py --output results/cpu-benchmark.json
```

It compares magnifications with bundled references and times forward
Jacobians. Independent derivative comparisons live in the focused tests.
The optional regional maps are described in
[Benchmarks](docs_src/benchmarks.rst).

Run the regular package suite when changing shared components:

```bash
pytest -q
```

The regular suite excludes `gpu` and `slow` tests. On a CUDA machine, run
`pytest -c pytest-gpu.ini -q`. Run an individual slow test when its numerical
case is relevant. Add a regression test for a concrete failure or public
behavior; routine changes do not need another broad parameter sweep.

## Documentation and examples

Edit the guides in `docs_src/` and build the documentation locally:

```bash
python -m sphinx -b html -W docs_src docs
```

Open `docs/index.html` to review it. `docs_src/` is the source; `docs/` is
the generated site and is excluded from Git. Keep examples short,
show the resulting figure, and give a command that runs from the repository
root. Documentation builds should read existing figures rather than run
long calculations.

The Documentation workflow builds pull requests and publishes `develop` through
GitHub Pages. For the initial migration, install the workflow on `develop` and
confirm its build succeeds, then set **Settings → Pages → Source → GitHub Actions**.
Deployment stays disabled while Pages still uses the old branch source.
Generated HTML is uploaded as a Pages artifact instead of committed to Git.

For a longer experiment, use a standalone script or a manually launched
batch. Save its configuration, numerical outputs, and log under `results/`;
use a notebook to load those outputs and make plots. There is no need to keep
a notebook session open while the calculation runs.

## Commits and pull requests

Commit source, documentation, focused tests, and small required fixtures.
Keep runtime measurements in `results/`, generated HTML in `docs/`, and retired
local experiments in `archive/`; these directories are ignored. Selected
example PNGs are versioned beside their scripts so the README figures can
be viewed without running a calculation.

Describe the resulting behavior and the checks performed in the pull
request. For bug reports, include a small reproducer, the microJAX and JAX
versions, device, and precision setting.
