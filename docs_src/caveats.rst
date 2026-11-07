Solver Caveats
==============

This page collects microJAX-specific constraints that can materially affect a
scientific result. General JAX installation and device troubleshooting belong
in the `JAX documentation <https://docs.jax.dev/>`_.

Backend scope is explicit
-------------------------

``mag_binary`` and ``mag_triple`` use the accelerator-oriented backend unless
``backend="cpu"`` is requested. The dedicated CPU integrator supports both
lens models and is selected explicitly, independently of the JAX device.
``BinaryMagConfig`` and ``TripleMagConfig`` configure the accelerator path and
do not tune the CPU path. Running the default implementation on a CPU device
does not select the dedicated CPU algorithm.

The CPU finite-source contract requires positive mass ratios and source radius,
with distinct lens positions. An exact zero third mass or coincident lenses
changes the lens model's degree and is outside the triple contract; use the
binary API for a two-lens model.

Finite values are not accuracy certificates
--------------------------------------------

The accelerator solver and production CPU solver use bounded work. They do not
increase the integration order or retry indefinitely until a requested error
tolerance is met. A finite magnification therefore has no guaranteed relative
error bound. Validate values and derivatives over the parameter region used in
an analysis against an independent implementation.

Failure diagnostics depend on the backend
-----------------------------------------

The accelerator finite-source API returns ``NaN`` when it detects invalid
image geometry, exhausted fixed capacity, or a non-finite calculation.
The CPU backend preserves the best-effort magnification even when ``status``
is nonzero. ``return_info=True`` returns only ``magnification`` and ``status``.
Status flags structural concerns and does not establish an accuracy or
quadrature-convergence guarantee. A CPU calculation can still produce a
non-finite value.

Record the complete lens and source configuration for problematic samples.
For CPU-specific interpretation of values and diagnostics, see
:doc:`cpu_backend`.

The numerical graph is piecewise
--------------------------------

Caustic crossings change image topology, and the solver switches between a
multipole approximation and a full finite-source calculation. CPU full solves
also select among fixed integration charts. Values and forward derivatives can
therefore be non-smooth near route and topology boundaries even when they are
finite. Automatic differentiation does not certify numerical accuracy.

Use x64 mode and forward differentiation
----------------------------------------

Enable ``jax_enable_x64`` before constructing arrays or compiling functions.
The polynomial roots and image-boundary calculations are numerically sensitive
in single precision. Forward mode (``jax.jvp`` or ``jax.jacfwd``) is the
supported light-curve differentiation route; reverse mode through the CPU
scheduler's data-dependent loops is not part of its API.

Configuration changes compile separately
----------------------------------------

Accelerator scheduler settings and array shapes participate in JAX's static
compilation. Changing them creates a separate executable, and the first call
includes compilation. Warm up the exact backend, source profile, configuration,
and array shape before measuring runtime. See :doc:`performance` for the
measured accelerator scope.

The paper version is a different solver line
--------------------------------------------

The methods paper corresponds to ``v0.1.1``. The ``0.2`` series is a substantial
redesign rather than a patch-level continuation. Record the exact release or
Git commit and do not combine results from the two lines without explicit
cross-validation. See :doc:`citing` for the complete reproducibility checklist.
