"""Public finite-source magnification API.

``mag_binary`` and ``mag_triple`` calculate magnification along a complex
source trajectory. ``BinaryMagConfig`` and ``TripleMagConfig`` control source
boundary sampling and the static accelerator scheduling used by the full
finite-source calculation.

Use ``u1=0`` for a uniform source or a positive ``u1`` for linear limb
darkening. Enable JAX 64-bit mode for production calculations.
"""

from .config import BinaryMagConfig, TripleMagConfig
from .lightcurve import mag_binary, mag_triple

__all__ = ["BinaryMagConfig", "TripleMagConfig", "mag_binary", "mag_triple"]
