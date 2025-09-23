Development
===========

This page collects notes for contributors and power users who want to run the
full test suite, regenerate documentation, or profile kernels.

Test suite
----------

Install development dependencies first::

   python -m pip install -e ".[dev]"

CPU default suite::

   pytest -q

GPU-specific tests (opt-in)::

   pytest -m gpu -q

The GPU group remains skipped when no CUDA platform is visible; set
``JAX_PLATFORMS=cuda`` to force GPU execution when both CPU and CUDA libraries
are installed.  For deterministic runs, export ``XLA_FLAGS=--xla_gpu_deterministic_ops``.

Documentation
-------------

The Sphinx sources reside in ``docs_src``.  Rebuild the HTML site via::

   cd docs_src
   make html

Artifacts appear under ``docs/``.  If you automate documentation deployment,
consider using ``sphinx-build -b html docs_src docs`` in CI for explicit output
paths.

Code style & tooling
--------------------

- Run ``ruff check --fix`` followed by ``ruff format`` to keep Python files tidy.
- Static type hints are gradually being added; run ``pyright`` when modifying
  typed modules.
- Enable 64-bit mode in tests to mirror production usage::

     python - <<'PY'
     from jax import config
     config.update("jax_enable_x64", True)
     PY

Profiling tips
--------------

- Wrap expensive kernels with ``jax.profiler.trace_function`` or use
  ``jax.profiler.start_trace`` to generate traces viewable in TensorBoard.
- When experimenting with quadrature settings, benchmark on representative
  trajectories—``chunk_size`` and ``MAX_FULL_CALLS`` interact strongly.

Questions & support
-------------------

File an issue on GitHub or start a discussion thread.  Please include your JAX
versions, accelerator information, and a minimal reproducer.
