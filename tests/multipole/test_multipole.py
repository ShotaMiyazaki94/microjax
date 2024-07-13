import jax.numpy as jnp
from jax import jit, jacfwd 
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

from microjax.caustics.lightcurve import magnifications
from microjax.caustics.extended_source import mag_extended_source
from microjax.point_source import critical_and_caustic_curves, _images_point_source, mag_point_source
from microjax.multipole import _mag_hexadecapole

# Parameters
s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.1  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
a = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"a": a, "e1": e1, "s": s, "q": q}

critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)

alpha = jnp.deg2rad(65) # angle between lens axis and source trajectory
tE = 20 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.1 # impact parameter
rho = 5e-3

t  =  jnp.linspace(-15, 12.5, 1000)
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 

w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
x_cm = 0.5 * s * (1.0 - q) / (1.0 + q) # mid-point -> center-of-mass

z, z_mask = _images_point_source(w_points - x_cm, nlenses=2, **_params)
mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, u1= 0.5, nlenses=2, **_params)
mag_points = mag_point_source(w_points, nlenses=2, **_params) 

mag_full = lambda w: mag_extended_source(w, rho, nlenses=2, **_params)
mag_ext  = jax.jit(jax.vmap(mag_full))(w_points)

fig, ax = plt.subplots(
    1, 1,
    #figsize=(5, 5),
    #gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1, 1, 1], 'wspace':0.3},
    #sharex=True,
)
ax_in = inset_axes(ax,
    width="60%", height="60%", 
    bbox_transform=ax.transAxes,
    bbox_to_anchor=(-0.2, 0.3, 0.6, 0.6)
    #bbox_to_anchor=(-0.1, 0.05, .4, .4),
)
ax_in.set_aspect(1)
ax_in.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")
for cc in caustic_curves:
    ax_in.plot(cc.real, cc.imag, color='black', lw=0.7)
circles = [
    plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, zorder=-1) for xi, yi in zip(w_points.real, w_points.imag)
]
c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.05)
ax_in.add_collection(c)
ax_in.set_aspect(1)
ax_in.set(xlim=(-1., 1.2), ylim=(-0.8, 1.))

ax.plot(t, mag_ext)
ax.plot(t, mu_multi, "--")
ax.plot(t, mag_points, ":")
ax.set_yscale("log")
plt.show()