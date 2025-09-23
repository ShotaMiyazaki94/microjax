Concepts & Architecture
=======================

This page sketches the internal layout of ``microJAX`` so you can orient
yourself before extending the library or wiring it into larger projects.

Core packages
-------------

``microjax.point_source``
    Implements the complex lens equation, Jacobians, and point-source
    magnification for up to three lenses.  Provides the polynomial root solver
    interface used by higher-level routines.

``microjax.inverse_ray``
    Houses the finite-source machinery: adaptive light-curve drivers
    (:mod:`microjax.inverse_ray.lightcurve`), contour integrators, limb
    darkening profiles, and the heuristics that switch between multipole and
    full inverse-ray solves.

``microjax.multipole``
    Supplies the hexadecapole approximation used as the "fast path" for finite
    sources, exposing both magnifications and error estimates for adaptive
    switching.

``microjax.trajectory``
    Contains helpers to construct source trajectories, including annual
    parallax and custom time sampling.

``microjax.likelihood``
    Offers analytic marginalisation routines for flux models, ready to drop into
    NumPyro or black-box optimisers.

Execution model
---------------

1. Build complex source coordinates ``w = x + i y``.
2. Call a magnification routine (:func:`microjax.point_source.mag_point_source`,
   :func:`microjax.inverse_ray.lightcurve.mag_binary`, etc.).
3. Compose the result with photometric likelihoods or inference code.

All heavy kernels are JIT-compatible and designed for batch execution.  GPU
workloads benefit from passing large, contiguous chunks of trajectories so that
inverse-ray chunks stay resident on device.

Accuracy strategy
-----------------

- Start with the multipole approximation to establish a baseline.
- Use heuristics (caustic proximity, planetary tests, error estimates from the
  multipole evaluation) to decide when a contour integral is required.
- Re-use previously solved image positions to warm-start root finders and keep
  JAX graphs compact.

Key data conventions
--------------------

- Complex numbers encode two-dimensional coordinates throughout the public API.
- Array broadcasting follows NumPy semantics; batches live on the leading axes.
- 64-bit mode is recommended for numerical stability (``jax_enable_x64=True``).

Further reading
---------------

See :doc:`usage` for concrete code snippets, and browse the auto-generated API
reference for callable signatures.
