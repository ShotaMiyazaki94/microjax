import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import jax.numpy as jnp
from microjax.caustics.extended_source import mag_extended_source
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

t  =  jnp.linspace(-15, 12.5, 1000)
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
x_cm = 0.5 * s * (1.0 - q) / (1.0 + q) # mid-point -> center-of-mass

a  = 0.5 * s
e1 = q / (1.0 + q) 
_params = {"a": a, "e1": e1, "q": q, "s": s}

@partial(jit, static_argnames=("npts_limb", "npts_ld", "limb_darkening"))
def mag_binary(w_points, rho, s, q, u1=0., npts_limb=300, npts_ld=100, limb_darkening=False):
    def body_fn(_, w):
        mag = mag_extended_source(
            w,
            rho,
            nlenses=2,
            npts_limb=npts_limb,
            limb_darkening=limb_darkening,
            npts_ld=npts_ld,
            u1=u1,
            s=s,
            q=q,
        )
        return 0, mag

    _, mags = lax.scan(body_fn, 0, w_points)
    return mags

mags_ext = mag_binary(w_points, rho, s=s, q=q)
mags_poi = mag_point_source(w_points, s=s, q=q)
critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)

fig, ax = plt.subplots()
ax_in = inset_axes(ax,
    width="60%", height="60%", 
    bbox_transform=ax.transAxes,
    bbox_to_anchor=(-0.2, 0.3, 0.6, 0.6)
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

ax.plot(t, mags_ext)
ax.plot(t, mags_poi, ":")
ax.set_yscale("log")
plt.show()