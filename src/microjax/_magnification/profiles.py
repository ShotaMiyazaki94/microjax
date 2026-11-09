"""Axisymmetric source-brightness profiles for boundary integration.

Currently implements a linear limb-darkening law with a custom JVP so that
gradients remain informative near the limb.
"""

import jax.numpy as jnp
from typing import Union

Array = jnp.ndarray


def linear_limb_intensity(d: Union[float, Array], u1: float = 0.0) -> Union[float, Array]:
    """Safe normalized linear limb intensity for exact inside intervals.

    Unlike :func:`Is_limb_1st`, this function needs no surrogate edge JVP: the
    boundary-root integrator differentiates the moving interval endpoints and
    its sine-squared coordinate has zero Jacobian at the limb.
    """

    d = jnp.asarray(d)
    strictly_inside = d < 1.0
    # ``sqrt(max(1-d**2, 0))`` has the correct value outside the disk but its
    # reverse rule forms ``0 * inf`` at the clipped square root. Boundary-root
    # quadrature can land a few ulps outside the limb, so evaluate a benign
    # radicand on the inactive branch before masking ``mu`` to zero. At exactly
    # the limb the intensity value is retained while the distance derivative
    # is zero; motion of the integration endpoint supplies the boundary term.
    safe_radicand = jnp.where(strictly_inside, 1.0 - d**2, 1.0)
    mu = jnp.where(strictly_inside, jnp.sqrt(safe_radicand), 0.0)
    normalization = 3.0 / (jnp.pi * (3.0 - u1))
    intensity = normalization * (1.0 - u1 * (1.0 - mu))
    return jnp.where(d <= 1.0, intensity, 0.0)


# @partial(jit, static_argnames=("u1"))
