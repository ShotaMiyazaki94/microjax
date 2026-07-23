"""Shared fixed-shape result types and integration conventions."""

from typing import NamedTuple

import jax.numpy as jnp

Array = jnp.ndarray

# Eight cells retain A100 throughput while reducing the direct outer-vmap peak.
SEQUENTIAL_RADIAL_CHUNK_SIZE = 8


class BoundaryMagnificationResult(NamedTuple):
    """Magnification plus the numerical error estimate and status bit mask."""

    magnification: Array
    estimated_error: Array
    status: Array


def integration_dtypes(w_center: complex) -> tuple[jnp.dtype, jnp.dtype]:
    """Return the real and complex dtypes used by boundary integration."""

    real_dtype = jnp.asarray(w_center).real.dtype
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    return real_dtype, complex_dtype


def unwrap_boundary_result(
    result: BoundaryMagnificationResult, return_info: bool, fatal_statuses: int
) -> Array | BoundaryMagnificationResult:
    """Return diagnostics on request; otherwise map fatal status to ``nan``."""

    if return_info:
        return result
    fatal = (result.status & fatal_statuses) != 0
    return jnp.where(fatal, jnp.nan, result.magnification)
