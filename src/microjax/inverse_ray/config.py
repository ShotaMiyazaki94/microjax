"""Static configuration for finite-source calculations."""

from dataclasses import dataclass
from operator import index


def _validate_positive_integer(name: str, value: int) -> None:
    """Validate a user-provided static shape without coercing its type."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        integer_value = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if integer_value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class _BoundaryMagConfig:
    """Shared topology-sampling and accelerator-scheduling settings.

    ``source_tile_size`` controls the outer batch of source positions.
    ``radial_chunk_size`` controls the inner batch of radial regions; a value
    of 64 evaluates the complete fixed-capacity region buffer together.

    These values alter Python control flow or static JAX shapes, so changing
    them produces a separately compiled executable.
    """

    n_limb: int = 500
    source_tile_size: int = 100
    radial_chunk_size: int = 8

    def __post_init__(self) -> None:
        """Reject scheduler values that cannot define valid static shapes."""

        _validate_positive_integer("source_tile_size", self.source_tile_size)
        _validate_positive_integer("radial_chunk_size", self.radial_chunk_size)


@dataclass(frozen=True)
class BinaryMagConfig(_BoundaryMagConfig):
    """Static configuration for ``mag_binary``."""

    radial_chunk_size: int = 64


@dataclass(frozen=True)
class TripleMagConfig(_BoundaryMagConfig):
    """Static configuration for ``mag_triple``."""


DEFAULT_BINARY_CONFIG = BinaryMagConfig()
DEFAULT_TRIPLE_CONFIG = TripleMagConfig()

__all__ = [
    "BinaryMagConfig",
    "DEFAULT_BINARY_CONFIG",
    "DEFAULT_TRIPLE_CONFIG",
    "TripleMagConfig",
]
