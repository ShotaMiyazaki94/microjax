"""Immutable static configuration for inverse-ray solvers.

Physical and differentiable quantities stay as explicit array arguments.
These objects contain only choices that alter Python control flow or static
XLA shapes and therefore belong outside the differentiated parameter tree.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _BoundaryMagConfig:
    """Shared static settings for one-pass boundary light-curve solvers."""

    n_limb: int = 500
    margin_r: float = 0.5
    angular_atol: float = 1e-5
    relative_tolerance: float = 1e-4
    parallel_regions: bool = False


@dataclass(frozen=True)
class BinaryMagConfig(_BoundaryMagConfig):
    """Static scheduler settings for :func:`inverse_ray.mag_binary`."""


@dataclass(frozen=True)
class TripleMagConfig(_BoundaryMagConfig):
    """Static scheduler settings for :func:`inverse_ray.mag_triple`."""


DEFAULT_BINARY_CONFIG = BinaryMagConfig()
DEFAULT_TRIPLE_CONFIG = TripleMagConfig()

__all__ = [
    "BinaryMagConfig",
    "DEFAULT_BINARY_CONFIG",
    "DEFAULT_TRIPLE_CONFIG",
    "TripleMagConfig",
]
