import numpy as np
import VBBinaryLensing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font="serif", style="ticks", font_scale=1.4)

import jax.numpy as jnp
import jax
from jax import jit, jacfwd
jax.config.update("jax_enable_x64", True)

from microjax.inverse_ray.lightcurve import mag_binary
from microjax.likelihood import nll_ulens
import corner

data = np.load("example/ob140124/flux.npy")
t_data, flux_data, fluxe_data = data[0] - 2450000, data[1], data[2]
t_data = jnp.array(t_data)
flux_data = jnp.array(flux_data)
fluxe_data = jnp.array(fluxe_data)
from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
data_input = (t_data, flux_data, fluxe_data, coords.ra.deg, coords.dec.deg)

param_keys = ["t0", "tE", "u0", "log_q", "log_s", "alpha_deg", "log_rho"]

param_adam = np.load("example/ob140124/adam_fwd_params.npz")
param_dict = {key: param_adam[key] for key in param_adam.files}
for key in param_adam.files:
    print(f"{key}:", param_adam[key])