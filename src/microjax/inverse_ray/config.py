"""Source-boundary sampling configuration for finite-source calculations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _BoundaryMagConfig:
    """Number of points used to trace the circular source boundary."""

    n_limb: int = 500


@dataclass(frozen=True)
class BinaryMagConfig(_BoundaryMagConfig):
    """Source-boundary sampling configuration for ``mag_binary``."""


@dataclass(frozen=True)
class TripleMagConfig(_BoundaryMagConfig):
    """Source-boundary sampling configuration for ``mag_triple``."""


DEFAULT_BINARY_CONFIG = BinaryMagConfig()
DEFAULT_TRIPLE_CONFIG = TripleMagConfig()

__all__ = [
    "BinaryMagConfig",
    "DEFAULT_BINARY_CONFIG",
    "DEFAULT_TRIPLE_CONFIG",
    "TripleMagConfig",
]
