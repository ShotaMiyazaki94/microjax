"""Binary/triple CPU integration, isolated from accelerator numerical kernels."""

from .lightcurve import mag_binary_cpu_lightcurve
from .result import CpuMagnificationResult

__all__ = ["CpuMagnificationResult", "mag_binary_cpu_lightcurve"]
