# -*- coding: utf-8 -*-
from ._magnification.api import (
    mag_binary,
    mag_triple,
    make_mag_binary,
    make_mag_triple,
    make_binary_lightcurve,
    make_triple_lightcurve,
)
from ._magnification.gpu.config import BinaryMagConfig, TripleMagConfig
from ._magnification.cpu.result import CpuMagnificationResult

__all__ = [
    "mag_binary",
    "mag_triple",
    "make_mag_binary",
    "make_mag_triple",
    "make_binary_lightcurve",
    "make_triple_lightcurve",
    "BinaryMagConfig",
    "TripleMagConfig",
    "CpuMagnificationResult",
]

__version__ = "0.2.0"
__uri__ = ""
__author__ = "Shota Miyazaki and collaborators"
__email__ = "miyazaki@ir.isas.jaxa.jp"
__license__ = "MIT"
__description__ = "Auto-differentiable Microlensing Modeling Framework with JAX."
