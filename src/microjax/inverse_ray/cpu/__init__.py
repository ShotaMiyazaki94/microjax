"""CPU-oriented inverse-ray kernels used by ``mag_binary(backend="cpu")``.

The package remains isolated from the accelerator boundary integrator, while
the parent module exposes the validated hybrid scheduler as the public API.
"""

from .coefficients import binary_quintic_coefficients
from .limb_dark import mag_limb_dark_cpu, mag_limb_dark_cpu_fixed
from .lightcurve import (
    mag_binary_cpu_lightcurve,
    mag_binary_cpu_one_shot_lightcurve,
)
from .one_shot import (
    ONE_SHOT_INVALID_ROOTS,
    ONE_SHOT_NONFINITE,
    ONE_SHOT_UNRESOLVED_GEOMETRY,
    mag_limb_dark_cpu_one_shot,
    mag_uniform_cpu_one_shot,
)
from .uniform import (
    CPU_TIER_EXHAUSTED,
    CpuMagnificationResult,
    mag_uniform_cpu,
    mag_uniform_cpu_fixed,
)

__all__ = [
    "CpuMagnificationResult",
    "CPU_TIER_EXHAUSTED",
    "ONE_SHOT_INVALID_ROOTS",
    "ONE_SHOT_NONFINITE",
    "ONE_SHOT_UNRESOLVED_GEOMETRY",
    "binary_quintic_coefficients",
    "mag_uniform_cpu",
    "mag_uniform_cpu_fixed",
    "mag_limb_dark_cpu",
    "mag_limb_dark_cpu_fixed",
    "mag_binary_cpu_lightcurve",
    "mag_binary_cpu_one_shot_lightcurve",
    "mag_limb_dark_cpu_one_shot",
    "mag_uniform_cpu_one_shot",
]
