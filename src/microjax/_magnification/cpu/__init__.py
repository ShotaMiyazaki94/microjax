"""Binary/triple CPU integration, isolated from GPU numerical kernels."""

from .lightcurve import mag_binary_cpu_lightcurve
from .result import CpuMagnificationResult

__all__ = ["CpuMagnificationResult", "mag_binary_cpu_lightcurve"]
