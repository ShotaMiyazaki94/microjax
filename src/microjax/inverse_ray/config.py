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
    """Static configuration for the accelerator ``mag_binary`` fast path.

    The defaults expose at least 512 independent source points to the GPU and
    retain 64 source-limb samples. The 40-lane radial buffer covers the audited
    local-chart topology in one parallel launch without paying for the complete
    64-slot general-purpose buffer. Every fast radial cell uses one externally
    audited 19-point rule across the full binary mass-ratio range.
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
    """Static configuration for the accelerator ``mag_triple`` path.

    The A100 triple-lens audit selected 128 source-limb samples. The smaller
    100-point outer tile and eight-cell radial chunk retain substantially more
    throughput for the ten-direction forward Jacobian than the wider binary
    scheduler while keeping the audited comparison track below ``1e-3``
    relative error.
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
