import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from microjax.point_source import critical_and_caustic_curves
import matplotlib as mpl
import seaborn as sns
sns.set_theme(font="serif", font_scale=1.0,style="ticks")

file = np.loadtxt("paper/time_mag.csv", delimiter=",")
t, A = file.T[0], file.T[1]
jac = np.load("paper/jacobian.npy").T

param_names = ['t0', 'tE', 'u0', 'q', 's', 'alpha', 'rho', 'q3', 'r3', 'psi']

n_params = jac.shape[0]

fig, ax = plt.subplots(
    8, 1,
    figsize=(14, 10),
    gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1, 1, 1], 'wspace':0.3},
    sharex=True,
)

fig, axes = plt.subplots(11, 1, figsize=(12, 8), sharex=True,
                         gridspec_kw={'height_ratios': [6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'wspace':0.3}) 
                         #gridspec_kw={'hspace': 0.1, 'height_ratios': [2] + [1]*n_params})

# 増光率
axes[0].plot(t, A, label='Magnification $A(t)$', color='black')
axes[0].set_ylabel('Magnification')
#axes[0].legend(loc='upper right')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
ax_in = inset_axes(axes[0],
    width="60%", # 
    height="60%", 
    bbox_transform=axes[0].transAxes,
    bbox_to_anchor=(-0.5, 0.05, .9, .9),
)
ax_in.set_aspect(1)
ax_in.set_aspect(1)
ax_in.set(xlabel="Re$(w)$", ylabel="Im$(w)$")

s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
q  = 0.1  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
q3 = 0.03
r3_complex = 0.3+1.2j 
psi = jnp.arctan2(r3_complex.imag, r3_complex.real)
alpha = jnp.deg2rad(50) # angle between lens axis and source trajectory
tE = 10 # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.1 # impact parameter
rho = 0.02
_params = {"q": q, "s": s, "q3": q3, "r3": jnp.abs(r3_complex), "psi": psi}

t  =  t0 + jnp.linspace(-0.5*tE, tE, 500)
tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
w_points = jnp.array(y1 + 1j * y2, dtype=complex)

critical_curves, caustic_curves = critical_and_caustic_curves(npts=1000, nlenses=3, **_params)
for cc in caustic_curves:
    ax_in.plot(cc.real, cc.imag, color='red', lw=0.7)
for cc in critical_curves:
    ax_in.plot(cc.real, cc.imag, color='green', lw=0.7)
ax_in.plot(-q*s, 0 ,"x",c="k", ms=2)
ax_in.plot((1.0-q)*s, 0 ,"x",c="k", ms=2)
ax_in.plot(r3_complex.real - (0.5*s - s/(1 + q)), r3_complex.imag ,"x",c="k", ms=2)

circles = [
    plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, zorder=-1) for xi,yi in zip(w_points.real, w_points.imag)
]
c = mpl.collections.PatchCollection(circles, 
                                    match_original=True, 
                                    alpha=0.05, 
                                    edgecolor="blue", 
                                    linewidth=0.5, 
                                    zorder=10)
ax_in.add_collection(c)
ax_in.set_aspect(1)
ax_in.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

# 各ヤコビアン
labels = [
    #r'$A(t)$',
    r'$\frac{\partial A}{\partial u_0}$', r'$\frac{\partial A}{\partial t_0}$',r'$\frac{\partial A}{\partial t_E}$',
    r'$\frac{\partial A}{\partial q}$', r'$\frac{\partial A}{\partial s}$', r'$\frac{\partial A}{\partial \alpha}$',
    r'$\frac{\partial A}{\partial \rho}$',
    r'$\frac{\partial A}{\partial q_3}$', r'$\frac{\partial A}{\partial r_3}$', r'$\frac{\partial A}{\partial \psi}$'
]
for i, l in enumerate(labels):
    axes[i+1].plot(t, jac[i])
    axes[i+1].set_ylabel(l)


#for i, pname in enumerate(param_names):
#    axes[i+1].set_ylabel(rf'$\partial A / \partial {pname}$')
#    axes[i + 1].plot(t, jac[i], label=rf'$\partial A / \partial {pname}$')
    #axes[i + 1].set_ylabel(pname)
    #axes[i + 1].legend(loc='upper right')
    #axes[i + 1].grid(True)

axes[-1].set_xlabel('Time [t]')

plt.tight_layout()
plt.savefig("paper/figure/full_jacobian_plot.pdf", dpi=300)
plt.show()
