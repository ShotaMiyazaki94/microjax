import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnames=("u1"))
def Is_limb_1st(d, u1=0.0):
    """
    Calculate the normalized limb-darkened intensity using a linear limb-darkening law.

    Parameters
    ----------
    r : array-like or float
        Radial distance from the center of the star's disk, 
        normalized such that r = 1 corresponds to the edge of the stellar disk.
        Should be in the range [0, 1].
    u1 : float, optional
        Linear limb-darkening coefficient. Defaults to 0.0, which corresponds 
        to a uniform disk with no limb darkening.

    Returns
    -------
    I : array-like or float
        Normalized intensity at the given radial distance(s), calculated as:
        I(r) = (3 / (π * (3 - u1))) * (1 - u1 * (1 - sqrt(1 - r))).

    Notes
    -----
    - The returned intensity is normalized such that the integral over the stellar disk is 1.
    - The equation implements the linear limb-darkening law:
      I(r) = I0 * (1 - u1 * (1 - sqrt(1 - r))),
      where I0 is a normalization constant ensuring that the total flux is conserved.
    - For physically meaningful results, `u1` should be in the range [0, 1], though 
      values outside this range can be used for testing or hypothetical scenarios.
    """
    mu = jnp.sqrt(1.0 - d**2)
    I0 = 3.0 / jnp.pi / (3.0 - u1)
    I  = I0 * (1.0 - u1 * (1.0 - mu))
    return jnp.where(d < 1.0, I, 0.0) 