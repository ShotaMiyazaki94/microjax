import jax
import jax.numpy as jnp
from jax import jit
jax.config.update("jax_enable_x64", True)
from microjax.point_source import lens_eq
from microjax.point_source import critical_and_caustic_curves
from microjax.coeffs import _poly_coeffs_triple as coeff_tri_old
from microjax.coeffs import _poly_coeffs_binary
from microjax.coeffs_triple import _poly_coeffs_triple as coeff_tri_new

import time
q  = 0.1
s  = 1.1
q3 = 5e-3
r3 = 0.3 + 1.2j 
psi = jnp.arctan2(r3.imag, r3.real)
w_center = 0.1 + 0.1j
a = 0.5 * s
e1 = q / (1.0 + q + q3)
e2 = 1.0 / (1.0 + q + q3)
jax.numpy.set_printoptions(precision=4, suppress=True)

start = time.time()
binary = jit(_poly_coeffs_binary)
binary(w_center, a, e1)
end = time.time()
print("binary coeff: ", end-start, "sec")
start = time.time()
binary(w_center, a, e1)
end = time.time()
print("binary coeff, exec: ", end-start, "sec")

start = time.time()
new = jit(coeff_tri_new)
new(w_center, a, r3, e1, e2)
end = time.time()
print("new coeff_tri: ", end-start, "sec")
start = time.time()
new(w_center, a, r3, e1, e2)
end = time.time()
print("new coeff_tri, exec: ", end-start, "sec")

start = time.time()
old = jit(coeff_tri_old)
old(w_center, a, r3, e1, e2)
end = time.time()
print("old coeff_tri: ", end-start, "sec")
start = time.time()
old(w_center, a, r3, e1, e2)
end = time.time()
print("old coeff_tri, exec: ", end-start, "sec")