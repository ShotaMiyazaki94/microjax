import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from microjax.caustics.extended_source import mag_extended_source
from microjax.point_source import lens_eq, _images_point_source, critical_and_caustic_curves
from microjax.image_area_all import image_area_all

w_center = jnp.complex128(0.425 + 0.0j)
#w_center = jnp.complex128(-0.1 - 0.1j)
q  = 0.5
s  = 1.0
a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"q": q, "s": s, "a": a, "e1": e1}
rho =  0.1
NBIN = 20

import time
start_time = time.time()
mag = mag_extended_source(w_center, rho, **_params)
jax.device_put(mag).block_until_ready()
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")
