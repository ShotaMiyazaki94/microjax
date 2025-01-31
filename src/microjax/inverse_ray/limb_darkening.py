import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax import custom_jvp

#@partial(jit, static_argnames=("u1"))
@custom_jvp
def Is_limb_1st(d, u1=0.0):
    """
    Calculate the normalized limb-darkened intensity using a linear limb-darkening law.
    - The equation implements the linear limb-darkening law:
      I(r) = I0 * (1 - u1 * (1 - sqrt(1 - r))),
      where I0 is a normalization constant ensuring that the total flux is conserved.
    - `d` is normalized by rho.
    - `u1` should be in the range [0, 1].
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