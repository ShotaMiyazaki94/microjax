"""Boundary and membership helpers for inverse-ray integration.

Provides JAX-compatible utilities for classifying points relative to the source
disk and for computing smooth boundary weights used in finite-source angular
integration. Custom JVPs avoid zero gradients at discontinuities.
"""

import jax 
import jax.numpy as jnp
from jax import custom_jvp
from microjax.point_source import lens_eq
from typing import Union

Array = jnp.ndarray

@custom_jvp
def calc_facB(delta_B: Union[float, Array], delta_c: Union[float, Array]) -> Union[float, Array]:
    """Smooth boundary factor for partial angular cells.

    Transitions from a linear rule for small ``delta_B`` to an asymptotic form
    for larger values to better approximate the covered fraction within a cell.

    Parameters
    - delta_B: float/array – Estimated local angular gap size.
    - delta_c: float – Smoothing/transition threshold.

    Returns
    - facB: float/array – Weighting factor applied to boundary cells.
    """
    facB = jnp.where(
        delta_B > delta_c,
        (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B) * (0.5 + delta_B),
        (2.0 / 3.0) * delta_B + 0.5,
    )
    return facB

@calc_facB.defjvp
def calc_facB_jvp(primal, tangent):
    delta_B, delta_c = primal
    delta_B_dot, delta_c_dot = tangent
    primal_out = calc_facB(delta_B, delta_c)
    #facB = (2.0 / 3.0) * delta_B + 0.5 
    tangent_out = 2.0 / 3.0 * delta_B_dot # applying the 1.5-order rule
    return primal_out, tangent_out

@custom_jvp
def step_smooth(x: Union[float, Array], fac: float = 100.0) -> Union[float, Array]:
    """Heaviside-like step with a sigmoid derivative for JVPs.

    - Function value: hard step ``1[x>0]`` for exact classification.
    - Derivative (JVP): steep sigmoid to provide nonzero gradients.
    """
    return jnp.where(x > 0, 1.0, 0.0)

@step_smooth.defjvp
def step_smooth_jvp(primal, tangent):
    x, fac = primal
    x_dot, fac_dot = tangent
    primal_out = step_smooth(x)

    z = x * fac
    sigmoid = jax.nn.sigmoid(z)
    dsig_dz = sigmoid * (1.0 - sigmoid)
    dz_dx   = fac
    dz_dfac = x
    tangent_out = x_dot * dsig_dz * dz_dx + fac_dot * dsig_dz * dz_dfac
    return primal_out, tangent_out 

@custom_jvp 
def in_source(distances: Array, rho: float) -> Array:
    """Smoothed indicator for whether points lie inside a circular source.

    Returns 1.0 if ``distances < rho`` and 0.0 otherwise. The JVP uses a steep
    sigmoid to avoid zero gradients at the boundary.
    """
    return jnp.where(distances - rho < 0.0, 1.0, 0.0)

@in_source.defjvp
def in_source_jvp(primal, tangent):
    distances, rho = primal
    distances_dot, rho_dot = tangent
    primal_out = in_source(distances, rho)

    z = (rho - distances) / rho 
    factor = 100.0 
    sigmoid_input = factor * z
    sigmoid = jax.nn.sigmoid(sigmoid_input)
    sigmoid_derivative = sigmoid * (1.0 - sigmoid) * factor
    dz_distances = -1.0 / rho
    dz_rho = distances / rho**2
    tangent_out = sigmoid_derivative * (dz_distances * distances_dot + dz_rho * rho_dot)
    primal_out = sigmoid
    return primal_out, tangent_out

def distance_from_source_adaptive(
    r0: float,
    th_unit: Array,
    th_min: float,
    th_max: float,
    w_center_shifted: complex,
    shifted: float,
    nlenses: int = 2,
    **_params,
) -> Array:
    """Distance to source for adaptively sampled angles at radius ``r0``.

    ``th_unit`` in [0, 1] maps linearly to ``[th_min, th_max]`` to support
    adaptive angular refinement within a subinterval.
    """
    th_values = th_min + (th_max - th_min) * th_unit
    x_th = r0 * jnp.cos(th_values)
    y_th = r0 * jnp.sin(th_values)
    z_th = x_th + 1j * y_th
    image_mesh = lens_eq(z_th - shifted, nlenses=nlenses, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    return distances

def distance_from_source(
    r0: float,
    th_values: Array,
    w_center_shifted: complex,
    shifted: float,
    nlenses: int = 2,
    **_params,
) -> Array:
    """Distance to source center for a fixed radius and set of angles."""
    x_th = r0 * jnp.cos(th_values)
    y_th = r0 * jnp.sin(th_values)
    z_th = x_th + 1j * y_th
    image_mesh = lens_eq(z_th - shifted, nlenses=nlenses, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    return distances
