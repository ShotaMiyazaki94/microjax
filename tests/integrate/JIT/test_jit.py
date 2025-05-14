import jax
import jax.numpy as jnp
from jax import jit
jax.config.update("jax_enable_x64", True)
from microjax.point_source import lens_eq
from microjax.point_source import critical_and_caustic_curves
from microjax.coeffs import _poly_coeffs_triple as coeff_tri_old
from microjax.coeffs import _poly_coeffs_binary
from microjax.coeffs_triple import _poly_coeffs_triple as coeff_tri_new
from microjax.coeffs_triple import _poly_coeffs_triple_CM
from microjax.poly_solver import poly_roots
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
jax.numpy.set_printoptions(precision=3, suppress=True)

coeff_new = coeff_tri_new(w_center, a, r3, e1, e2)
coeff_old = coeff_tri_old(w_center, a, r3, e1, e2)
roots_new = poly_roots(coeff_new)
roots_old = poly_roots(coeff_old)
coeff_CM, shift = _poly_coeffs_triple_CM(w_center, a, r3, e1, e2)
#print("coeff_CM: ", coeff_CM)
roots_CM = poly_roots(coeff_CM, 0, -1)
#roots_CM = poly_roots(jnp.moveaxis(coeff_CM, 0, -1))
#print("roots_new: ", jnp.sort(roots_new))
#print("roots_old: ", jnp.sort(roots_old))
roots_CM_shift = roots_CM + shift
#print("shift: ", shift)
#print("roots_CM: ", jnp.sort(roots_CM_shift))
diff = jnp.abs(jnp.sort(roots_old) - jnp.sort(roots_CM_shift))
print("diff: ", jnp.all(diff < 1e-9))

#exit(0)

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
new = jit(_poly_coeffs_triple_CM)
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