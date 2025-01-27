import jax 
import jax.numpy as jnp
from jax import custom_jvp
from microjax.point_source import lens_eq

@custom_jvp
def calc_facB(delta_B, delta_c):
    facB = jnp.where(delta_B > delta_c,
                     (2.0 / 3.0) * jnp.sqrt(1.0 + 0.5 / delta_B) * (0.5 + delta_B),
                     (2.0 / 3.0) * delta_B + 0.5)
    return facB

@calc_facB.defjvp
def calc_facB_jvp(primal, tangent):
    delta_B, delta_c = primal
    delta_B_dot, delta_c_dot = tangent
    primal_out = calc_facB(delta_B, delta_c)
    #facB = (2.0 / 3.0) * delta_B + 0.5 
    tangent_out = 2.0 / 3.0 * delta_B_dot
    return primal_out, tangent_out

@custom_jvp
def step_smooth(x, fac=100.0):
    """
    For function eval., step function.
    For derivative, sigmoid function.
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
def in_source(distances, rho):
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

def distance_from_source(r0, th_values, w_center_shifted, shifted, nlenses=2, **_params):
    x_th = r0 * jnp.cos(th_values)
    y_th = r0 * jnp.sin(th_values)
    z_th = x_th + 1j * y_th
    image_mesh = lens_eq(z_th - shifted, nlenses=nlenses, **_params)
    distances = jnp.abs(image_mesh - w_center_shifted)
    return distances