"""Limb-darkening profiles and their JAX-friendly derivatives.

Currently implements a linear limb-darkening law with a custom JVP so that
gradients remain informative near the limb.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax import custom_jvp
from typing import Union

Array = jnp.ndarray

#@partial(jit, static_argnames=("u1"))
@custom_jvp
def Is_limb_1st(d: Union[float, Array], u1: float = 0.0) -> Union[float, Array]:
    """Linear limb-darkening intensity profile, normalized.

    Implements: ``I(d) = I0 * [1 - u1 * (1 - mu)]`` with ``mu = sqrt(1-d^2)``
    and ``I0 = 3/(pi*(3-u1))`` so that the disk-integrated flux equals 1.

    Parameters
    - d: float/array – Radial distance normalized by ``rho`` (0 at center, 1 at limb).
    - u1: float – Linear limb-darkening coefficient in [0, 1].

    Returns
    - I: float/array – Normalized intensity; zeroed for ``d >= 1``.
    """
    mu = jnp.sqrt(1.0 - d**2)
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    I  = I0 * (1.0 - u1 * (1.0 - mu))
    return jnp.where(d < 1.0, I, 0.0)

@Is_limb_1st.defjvp
def Is_limb_1st_jvp(primals, tangents):
    d, u1 = primals
    d_dot, u1_dot = tangents

    fac = 100.0
    mu = jnp.sqrt(jnp.maximum(1e-3, 1.0 - d**2))
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    dI0_du1 = -3.0 / jnp.pi / (3.0 - u1)**2
    prof = 1.0 - u1 * (1.0 - mu)
    dprof_du1 = - (1.0 - mu)
    dprof_dmu = u1
    dmu_dd = -d / mu
    z = 1.0 - d
    sigmoid = jax.nn.sigmoid(fac * z)
    dsig_dz = fac * sigmoid * (1.0 - sigmoid)
    dz_dd = -1.0

    primal_out = I0 * prof * sigmoid
    tangent_out = u1_dot * (dI0_du1 * prof * sigmoid + I0 * dprof_du1 * sigmoid) \
      + d_dot * (I0 * dprof_dmu * dmu_dd * sigmoid + I0 * prof * dsig_dz * dz_dd)
    return primal_out, tangent_out
