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
    """Source-boundary sampling and batching for the GPU integrator.

    ``source_tile_size`` batches source positions; ``radial_chunk_size``
    batches radial image regions within each position. Changing these
    settings produces a new JAX compilation.
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
    """Sampling and batching settings for GPU binary magnification.

    ``n_limb`` samples the source boundary and must be at least 64.
    The two batch sizes control memory use and throughput.
    """

    n_limb: int = 64
    source_tile_size: int = 512
    radial_chunk_size: int = 40

    def __post_init__(self) -> None:
        """Retain the validated 64-limb minimum of the binary GPU path."""

        super().__post_init__()
        if self.n_limb < 64:
            raise ValueError("binary n_limb must be at least 64")


@dataclass(frozen=True)
class TripleMagConfig(_BoundaryMagConfig):
    """Sampling and batching settings for GPU triple magnification.

    ``n_limb`` samples the source boundary. The two batch sizes control
    memory use and throughput, independently of the sampling resolution.
    """

    n_limb: int = 128


DEFAULT_BINARY_CONFIG = BinaryMagConfig()
DEFAULT_TRIPLE_CONFIG = TripleMagConfig()

__all__ = [
    "BinaryMagConfig",
    "DEFAULT_BINARY_CONFIG",
    "DEFAULT_TRIPLE_CONFIG",
    "TripleMagConfig",
]
