"""Value and structural status returned by the CPU solver."""

from typing import NamedTuple
import jax


class CpuMagnificationResult(NamedTuple):
    """Magnification and structural status, with one entry per source position."""

    magnification: jax.Array
    status: jax.Array


INVALID_ROOTS = 1 << 20
NONFINITE = 1 << 21
