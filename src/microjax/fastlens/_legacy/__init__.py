# Legacy (CPU) implementations.
from .fftlog import fftlog, hankel
from .mag_fft import (
    magnification,
    magnification_disk,
    magnification_limb,
    magnification_log,
    A_point,
)

__all__ = [
    "fftlog",
    "hankel",
    "magnification",
    "magnification_disk",
    "magnification_limb",
    "magnification_log",
    "A_point",
]
