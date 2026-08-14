Solver Caveats
==============

This page collects microJAX-specific constraints that can materially affect a
scientific result. General JAX installation and device troubleshooting belong
in the `JAX documentation <https://docs.jax.dev/>`_.

Backend scope is explicit
-------------------------

``mag_binary`` uses the accelerator-oriented backend unless ``backend="cpu"``
is requested. The CPU backend is a separate binary-lens implementation, not an
automatic fallback selected from the available device. ``BinaryMagConfig``
configures the accelerator path and does not tune the CPU path. Finite-source
triple-lens calculations use the accelerator-oriented implementation.

Finite values are not accuracy certificates
--------------------------------------------

The accelerator solver and production CPU solver use bounded work. They do not
increase the integration order or retry indefinitely until a requested error
tolerance is met. A finite magnification therefore has no guaranteed relative
error bound. Validate values and derivatives over the parameter region used in
an analysis against an independent implementation.

Rejected configurations return NaN
----------------------------------

The ordinary finite-source API returns ``NaN`` when it detects invalid image
geometry, exhausted fixed capacity, or a non-finite calculation. Do not replace
such samples silently or treat them as zero likelihood without recording the
complete lens and source configuration. For CPU-specific solver investigation,
advanced diagnostic output is described in :doc:`cpu_backend`.

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
