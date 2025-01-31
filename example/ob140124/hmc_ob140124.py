import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MulensModel as mm
plt.rcParams["figure.figsize"] = (12,5)
import seaborn as sns
sns.set_theme(style='ticks', font_scale=1.2,)
from matplotlib import rc
#rc('text', usetex=True)

import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import config 
config.update('jax_enable_x64', True) 
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from microjax.inverse_ray.lightcurve import mag_lc

data = pd.read_csv("example/data/ogle-2014-blg-0124/phot.dat", delim_whitespace=True,header=None, names=["HJD", "mag", "mage", "seeing", "sky"])
data = data[(data.HJD>2.4563e+6)&(data.mage<0.4)]
#data = data[(data.HJD>2.4566e+6)&(data.mage<0.4)]
data["HJD"] -= 2450000
mag0 = 18.0
data["flux"] = 10**(-0.4*(data.mag - mag0))
data["fluxe"] = data.flux * 0.4 * np.log(10) * data.mage

# initial guess
_t_0, _u_0, _t_E = 6.83640951e+03, 2.24211333e-01, 1.33559958e+02 
_s, _q, _alpha = 9.16157288e-01, 5.87559438e-04, 1.00066409e+02
_rho, _pi_E_N, _pi_E_E = 2.44003713e-03, 1.82341182e-01,9.58542572e-02
_fs, _fb = 8.06074085e-01, 8.62216897e-01
theta_0 = jnp.array([_t_0, _u_0, _t_E, _s, _q, _alpha, _rho, _pi_E_N, _pi_E_E, _fs, _fb])

from astropy.coordinates import SkyCoord
coords = SkyCoord("18:02:29.21 −28:23:46.5", unit=("hourangle", "deg"))
if(0):
    #params_mm = {'t_0': _t_0, 'u_0': _u_0, 't_E': _t_E, 's': _s, 'q': _q, 'alpha': _alpha, 'rho': _rho}
    params_mm = {'t_0': _t_0, 'u_0': _u_0, 't_E': _t_E, 's': _s, 'q': _q, 'alpha': _alpha, 'rho': _rho, 'pi_E_N': _pi_E_N, 'pi_E_E': _pi_E_E}
    model = mm.Model(parameters=params_mm, coords=coords)
    model.set_default_magnification_method("VBBL")
    time_plot = jnp.linspace(6350,7000,2000)
    amp_mm = model.get_magnification(time_plot)
    plt.plot(time_plot, amp_mm*_fs + _fb, "-", lw=2, color="red", label="MulensModel")
    plt.errorbar(data.HJD, data.flux, yerr=data.fluxe, fmt='.', color='black', label="OGLE-I")
    plt.title("OGLE-2014-BLG-0124")
    plt.xlabel("HJD - 2,450,000")
    plt.ylabel("flux")
    plt.xlim(6650,7000)
    plt.legend()
    plt.savefig("example/ob140124/ob140124_data.pdf")
    plt.show()

