import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import jax.numpy as jnp
from microjax.inverse_ray.lightcurve_uniform import mag_lightcurve
from microjax.point_source import mag_point_source, critical_and_caustic_curves
from microjax.point_source import _images_point_source
import jax
from jax import lax, jit
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from functools import partial

q = 0.5
s = 1.0
alpha = jnp.deg2rad(65) # angle between lens axis and source trajectory
tE = 20 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.1 # impact parameter
rho = 5e-3

t  =  jnp.linspace(-15, 12.5, 100)
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
x_cm = 0.5 * s * (1.0 - q) / (1.0 + q) # mid-point -> center-of-mass

a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"a": a, "e1": e1, "q": q, "s": s}

@jit
def mag_(w_points, rho, **_params):
    def body_fn(_, w):
        mag = mag_lightcurve(w, rho, nlenses=2, **_params)
        return 0, mag
    _, mags = lax.scan(body_fn, 0, w_points)
    return mags

mags_extended = mag_(w_points, rho, **_params)

print(mags_extended)
