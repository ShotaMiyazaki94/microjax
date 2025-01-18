import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib.pyplot as plt
from jax import jit, vmap, grad, jacfwd
from microjax.point_source import _images_point_source
import MulensModel as mm
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import time

s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.1  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
q3 = 1e-2
r3 = 0.3+1.2j 
psi = jnp.arctan2(r3.imag, r3.real)

alpha = jnp.deg2rad(40) # angle between lens axis and source trajectory
tE = 10 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.2 # impact parameter
t  =  t0 + jnp.linspace(-tE, tE, 1000)

# Position of the center of the source with respect to the center of mass.
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)

w = jnp.array(y1 + 1j * y2, dtype=complex)
a = 0.5*s
e1 = q/(1 + q + q3)  
e2 = 1/(1 + q + q3) 
r3 = r3*jnp.exp(1j*psi)
_params = {"a": a, "r3": r3, "e1": e1, "e2": e2}

start = time.time()
result = _images_point_source(w, a=a, e1=e1)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
end = time.time()
print(f"nlenses=2: {end - start:.3f} sec")

start = time.time()
result = _images_point_source(w, a=a, e1=e1, r3=r3, e2=e2, nlenses=3)
jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
end = time.time()
print(f"nlenses=3: {end - start:.3f} sec")
