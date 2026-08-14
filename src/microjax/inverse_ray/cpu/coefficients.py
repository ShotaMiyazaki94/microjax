"""Compatibility exports for the shared stable binary-lens coefficients.

The coefficient factorization follows Equation 6 of Wang, Wang & Dong
(2025, ApJS 276, 40), expressed in microJAX's public binary centre-of-mass
frame.  In particular, terms whose true scale is the secondary mass fraction
remain explicitly proportional to that fraction instead of being formed by
subtracting order-unity quantities.
"""

from ..geometry.coefficients import BinaryQuintic, binary_quintic_coefficients


__all__ = ["BinaryQuintic", "binary_quintic_coefficients"]
