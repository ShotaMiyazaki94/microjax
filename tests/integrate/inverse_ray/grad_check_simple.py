import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
import time
from microjax.inverse_ray.merge_area import calc_source_limb, calculate_overlap_and_range 
import jax.numpy as jnp
from jax import jit, vmap
from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import critical_and_caustic_curves
jax.config.update("jax_enable_x64", True)

from jax import jacfwd
def test_jacfwd():
    s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
    q  = 0.5  # mass ratio: mass of the lens on the right divided by mass of the lens on the left
    alpha = jnp.deg2rad(60) # angle between lens axis and source trajectory
    tE = 10.0 # einstein radius crossing time
    t0 = 0.0 # time of peak magnification
    u0 = 0.0 # impact parameter
    rho = 0.02

    params = jnp.array([s, q, rho, alpha, u0, t0, tE])

    def mag(t, params):
        s, q, rho, alpha, u0, t0, tE = params
        tau = (t - t0)/tE
        y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
        y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
        w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
        _params = {"q": q, "s": s}
        f      = lambda w: mag_uniform(w, rho, r_resolution=200, th_resolution=200, **_params)
        f_vmap =  vmap(f, in_axes=(0,))
        return w_points, f_vmap(w_points) 

    t  =  jnp.linspace(-22, 12, 1000)
    w_points, A  = mag(t, params)
    print("mag finish")
    mag_jac  = jacfwd(lambda params: mag(t, params)[1])
    jac_eval = mag_jac(params)
    print("jac finish")

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import AutoMinorLocator
    import matplotlib as mpl
    fig, ax = plt.subplots(
    8, 1,
    figsize=(10, 10),
    gridspec_kw={'height_ratios': [5, 1, 1, 1, 1, 1, 1, 1], 'wspace':0.2},
    sharex=True,
    )
    ax[0].plot(t, A, lw=2., color='C0')
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
        
    
    plt.show()

if __name__ == "__main__":
    test_jacfwd()    
