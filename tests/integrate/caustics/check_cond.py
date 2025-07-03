import numpy as np
from jax import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator

from microjax.point_source import critical_and_caustic_curves
from microjax.point_source import _images_point_source
from microjax.utils import *
from microjax.multipole import _mag_hexadecapole
from microjax.caustics.lightcurve import _caustics_proximity_test

import MulensModel as mm
def mag_vbb_binary(w0, rho, a, e1, u1=0., accuracy=5e-05):
    e2 = 1 - e1
    s = 2 * a
    bl = mm.BinaryLens(e2, e1, s)
    return bl.vbbl_magnification(w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)

rho = 0.02
a   = 0.8
e1  = 0.05
e2  = 1.0 - e1
q   = e1 / (1.0 - e1)
s   = 2 * a
critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, s=s, q=q, npts=100)

_x = jnp.linspace(-0.5, 1.3, int(3*200))
_y = jnp.linspace(-0.3, 0.3, 200)
xgrid, ygrid = jnp.meshgrid(_x, _y)
wgrid = xgrid + 1j*ygrid
x_cm = 0.5 * s * (1.0 - q) / (1.0 + q)

# Full calculation with VBB
mags_ref = np.zeros_like(wgrid).astype(float)

for i in range(wgrid.shape[0]):
    for j in range(wgrid.shape[1]):
        w_center = wgrid[i, j]
        mags_ref[i, j]  = mag_vbb_binary(w_center, rho, a, e1, u1=0.0)

# Evaluate hex approx. and the test
z, z_mask = _images_point_source(wgrid - x_cm, nlenses=2, a=a, e1=e1)
mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=2, a=a,e1=e1)
err_hex = jnp.abs(mu_multi - mags_ref) / mags_ref
test = _caustics_proximity_test(wgrid - x_cm, z, z_mask, rho, delta_mu_multi, 
                                nlenses=2,  a=a, e1=e1)

fig, ax = plt.subplots(figsize=(14, 10))

cmap1 = colors.ListedColormap(['grey', 'white'])
cmap2 = colors.ListedColormap(['white', 'red'])
cmap3 = colors.ListedColormap(['white', 'green'])

im = ax.pcolormesh(xgrid, ygrid, test, cmap=cmap1, alpha=1.0, zorder=-1)
im = ax.pcolormesh(xgrid, ygrid, err_hex > 1e-03, cmap=cmap2, alpha=0.7, zorder=-1)
#im = ax.pcolormesh(xgrid, ygrid, err_hex > 1e-03, cmap=cmap3, alpha=0.3, zorder=-1)

for cc in caustic_curves:
    ax.plot(cc.real, cc.imag, color="black", lw=0.7)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='grey', label=r'Tests evaluate to "False"', alpha=1.0),
    Patch(facecolor='red', label=r'$\epsilon_\mathrm{rel}>10^{-3}$', alpha=0.7),
    #Patch(facecolor='green', label=r'$\epsilon_\mathrm{rel}>10^{-3}$', alpha=0.7)
]
ax.legend(handles=legend_elements, fontsize=14)
c = plt.Circle((0, 0), radius=rho, fill=False, facecolor=None, zorder=-1)
ax.add_patch(c)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_aspect(1)
#ax.set(xlim=(-0.5, 1.), ylim=(-0.2, 0.2))
ax.set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$")
ax.set_rasterization_zorder(0)
fig.savefig("tests/integrate/caustics/check_cond.pdf",bbox_incehs="tight")
plt.show()