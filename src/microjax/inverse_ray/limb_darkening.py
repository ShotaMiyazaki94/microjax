import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnames=("u1"))
def Is_limb_1st(d, u1=0.0):
    """
    Calculate the normalized limb-darkened intensity using a linear limb-darkening law.
    - The equation implements the linear limb-darkening law:
      I(r) = I0 * (1 - u1 * (1 - sqrt(1 - r))),
      where I0 is a normalization constant ensuring that the total flux is conserved.
    - `u1` should be in the range [0, 1].
    """
    mu = jnp.sqrt(1.0 - d**2)
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    I  = I0 * (1.0 - u1 * (1.0 - mu))
    return jnp.where(d < 1.0, I, 0.0) 