Development
===========

This page collects notes for contributors and power users who want to run the
full test suite or rebuild the documentation.

Test suite
----------

Install development dependencies first::

   pip install -e ".[dev]"

Run the default (CPU) test selection::

   pytest -q

GPU-specific tests are opt-in. On a CUDA-capable machine run::

   pytest -m gpu -q

The GPU tests are skipped automatically when JAX cannot see a CUDA device. No
additional environment variable is required, though ``JAX_PLATFORMS=cuda`` can
be helpful if you have both CPU and GPU runtimes installed.

Documentation
-------------

The Sphinx sources live in ``docs_src`` and the rendered HTML is committed in
``docs`` for convenience.

Rebuild the docs locally::

   cd docs_src
   make html

The generated site appears in ``docs/``. You may publish it on GitHub Pages by
pushing the rendered files or by wiring up a documentation workflow.

Coding style
------------

- ``ruff`` handles linting and formatting; run ``ruff check --fix`` before
  committing changes.
- Enable JAX 64-bit mode during tests for reproducibility::

     python - <<'PY'
     from jax import config
     config.update("jax_enable_x64", True)
     PY

Where to ask questions
----------------------

Open a GitHub issue or start a discussion if you encounter problems or want to
propose improvements.
