from jax import config 
config.update('jax_enable_x64', True) 
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from microjax.inverse_ray.lightcurve import mag_binary
from microjax.likelihood import linear_chi2

t0 = 516.7182 - 0.5
tE = 25.52
u0 = -6.3e-02
q = 7.28e-05
s = 0.947
alpha =  jnp.pi + 2.340
rho = 1.6e-3

#t = t0 + jnp.linspace(-10, 10, 1000)
t  =  jnp.linspace(514, 515.2, 300)
tau = (t - t0)/tE 
um  = u0 
y1 = -um*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = um*jnp.cos(alpha) + tau*jnp.sin(alpha)
w_points = jnp.array(y1 + y2 * 1j, dtype=complex)

mags = mag_binary(w_points, rho, s=s, q=q, cubic=True, r_resolution=500, th_resolution=500)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,5)
import seaborn as sns
sns.set_theme(style='ticks', font_scale=1.4, font="serif") 
plt.plot(t, mags, "-", color='red', label="2L1S (microjax)")
plt.legend()
plt.savefig("example/wfirst_095/plot.pdf")
plt.close()