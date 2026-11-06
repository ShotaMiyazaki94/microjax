# Examples

Examples are grouped by the backend they are intended to exercise:

- [`gpu/`](gpu/) contains the accelerator-oriented binary and triple lens
  workflows.
- Current CPU checks and benchmarks live in [`tools/`](../tools/) and
  [`tests/unittests/inverse_ray/test_cpu.py`](../tests/unittests/inverse_ray/test_cpu.py).
  Previous CPU examples remain available in Git history.

Run commands from the repository root unless an example README says otherwise.
Committed numerical products record the hardware, JAX version, trajectory, and
solver configuration used for that run; they are not general performance or
accuracy guarantees.
