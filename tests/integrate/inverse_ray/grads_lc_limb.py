import jax.numpy as jnp
from jax import jit, jacfwd 
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

from microjax.inverse_ray.lightcurve import mag_lc_uniform, mag_lc
from microjax.point_source import critical_and_caustic_curves

# Parameters
s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.5  # mass ratio: mass of the lens on the right divided by mass of the lens on the left

alpha = jnp.deg2rad(60) # angle between lens axis and source trajectory
tE = 10.0 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.0 # impact parameter
rho = 0.02

a  = 0.5 * s
e1 = q / (1.0 + q)

# Position of the center of the source with respect to the center of mass.
t  =  jnp.linspace(-22, 12, 500)

r_resolution  = 1000
th_resolution = 4000
cubic = True

@jit
def get_mag(params):
    s, q, rho, alpha, u0, t0, tE = params
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 

    _params = {"q": q, "s": s}
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    return w_points, mag_lc(w_points, rho, nlenses=2, u1=0.0, q=q, s=s, cubic=cubic,
                            r_resolution=r_resolution, th_resolution=th_resolution)

params = jnp.array([s, q, rho, alpha, u0, t0, tE])
w_points, A = get_mag(params)
print("mag finish")

# Evaluate the Jacobian at every point
mag_jac = jit(jacfwd(lambda params: get_mag(params)[1]))
jac_eval = mag_jac(params)
print("jac finish")

fig, ax = plt.subplots(
    8, 1,
    figsize=(14, 14),
    gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1, 1, 1], 'wspace':0.3},
    sharex=True,
)
# Inset axes for images
ax_in = inset_axes(ax[0],
    width="60%", # 
    height="60%", 
    bbox_transform=ax[0].transAxes,
    bbox_to_anchor=(-0.4, 0.05, .9, .9),
)
ax_in.set_aspect(1)
ax_in.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")

critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)
for cc in caustic_curves:
    ax_in.plot(cc.real, cc.imag, color='black', lw=0.7)

circles = [
    plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, zorder=-1) for xi,yi in zip(w_points.real, w_points.imag)
]
c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.05)

ax_in.add_collection(c)
ax_in.set_aspect(1)
ax_in.set(xlim=(-1., 1.2), ylim=(-0.8, 1.))

ax[0].plot(t, A, color='black', lw=2)

for i, _a in enumerate(ax[1:]):
    _a.plot(t, jac_eval[:, i], lw=2., color='C0')

labels = [
    r'$A(t)$',
    r'$\frac{\partial A}{\partial s}$', r'$\frac{\partial A}{\partial q}$', r'$\frac{\partial A}{\partial \rho}$',
     r'$\frac{\partial A}{\partial \alpha}$', r'$\frac{\partial A}{\partial u_0}$', r'$\frac{\partial A}{\partial t_0}$', 
     r'$\frac{\partial A}{\partial t_E}$'
]

labelx = -0.07  # axes coords
for i, _a in enumerate(ax):
    _a.set_ylabel(
        labels[i], 
        rotation=0, 
        verticalalignment='center',
        horizontalalignment='right',
        fontsize=20,
    )
    _a.yaxis.set_label_coords(labelx, 0.5)
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())

ax[-1].set_xlabel('$t$ [days]')
ax_in.set_rasterization_zorder(0)
if cubic:
    fig.savefig(f"tests/integrate/inverse_ray/figs/grads_lc_limb_r{r_resolution}_{th_resolution}_cub.pdf", bbox_inches="tight")
    print(f"tests/integrate/inverse_ray/figs/grads_lc_limb_r{r_resolution}_{th_resolution}_cub.pdf")
else:
    fig.savefig(f"tests/integrate/inverse_ray/figs/grads_lc_limb_r{r_resolution}_{th_resolution}_lin.pdf", bbox_inches="tight")
    print(f"tests/integrate/inverse_ray/figs/grads_lc_limb_r{r_resolution}_{th_resolution}_lin.pdf")
plt.close()