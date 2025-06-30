import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev 
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import critical_and_caustic_curves

# Parameters
s  = 1.1  
q  = 0.1  
q3 = 0.03
r3_complex = 0.3 + 1.2j 
psi = jnp.arctan2(r3_complex.imag, r3_complex.real)

alpha = jnp.deg2rad(50) # angle between lens axis and source trajectory
tE = 10  # einstein radius crossing time
t0 = 0.0 # time of peak magnification
u0 = 0.1 # impact parameter
t  =  t0 + jnp.linspace(-0.5*tE, tE, 500)
rho = 0.02

tau = (t - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
w_points = jnp.array(y1 + 1j * y2, dtype=complex)

r_resolution  = 500
th_resolution = 500
Nlimb = 500
MAX_FULL_CALLS = 500
cubic = True

def chunked_vmap(func, data, chunk_size):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        results.append(jax.vmap(func)(chunk))
    return jnp.concatenate(results)

@jit
def get_mag(params):
    t0, tE, u0, q, s, alpha, rho, q3, r3, psi = params
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    _params = {"q": q, "s": s, "q3": q3, "r3": r3, "psi": psi}
    def mag_mj(w):
        return mag_uniform(w, rho, nlenses=3, **_params, cubic=cubic, 
                           r_resolution=r_resolution, th_resolution=th_resolution)
    magnifications = chunked_vmap(mag_mj, w_points, chunk_size=50)
    return w_points, magnifications

if(0):
    import time
    params = jnp.array([t0, tE, u0, q, s, alpha, rho, q3, jnp.abs(r3_complex), psi])
    get_mag(params)
    print("start")
    start = time.time()
    w_points, A = get_mag(params)
    end = time.time()
    print("mag finish: %.3f sec"%(end - start))
    # Evaluate the Jacobian at every point
    mag_jac = jit(jacfwd(lambda params: get_mag(params)[1]))
    mag_jac(params)
    print("start")
    start = time.time()
    jac_eval = mag_jac(params)
    end = time.time()
    print("jac finish: %.3f sec"%(end - start))

    t_np = np.array([t, A]).T
    jac_np = np.array(jac_eval)
    np.savetxt("example/triple-lens-jacobian/time_mag.csv", t_np, delimiter=",")
    np.save("example/triple-lens-jacobian/jacobian.npy", jac_np)

# Plotting
file = np.loadtxt("example/triple-lens-jacobian/time_mag.csv", delimiter=",")
t, A = file.T[0], file.T[1]
jac = np.load("example/triple-lens-jacobian/jacobian.npy").T

param_names = ['t0', 'tE', 'u0', 'q', 's', 'alpha', 'rho', 'q3', 'r3', 'psi']

n_params = jac.shape[0]

fig, axes = plt.subplots(11, 1, figsize=(12, 10), sharex=True,
                         gridspec_kw={'height_ratios': [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'wspace':0.3}) 
                         #gridspec_kw={'hspace': 0.1, 'height_ratios': [2] + [1]*n_params})

axes[0].plot(t, A, label='Magnification $A(t)$', color='black')
axes[0].set_ylabel('Magnification')
#axes[0].legend(loc='upper right')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
ax_in = inset_axes(axes[0],
    width="70%", # 
    height="70%", 
    bbox_transform=axes[0].transAxes,
    bbox_to_anchor=(0.05, 0.05, .9, .9),
    #bbox_to_anchor=(-0.45, 0.05, .9, .9),
)
ax_in.set_aspect(1)
ax_in.set_aspect(1)
ax_in.set(xlabel="Re$(w)$", ylabel="Im$(w)$")

s  = 1.1  # separation between the two lenses in units of total ang. Einstein radii
ax_in.set_aspect(1)
ax_in.set_aspect(1)
ax_in.set(xlabel="Re$(w)$", ylabel="Im$(w)$")

_params = {"q": q, "s": s, "q3": q3, "r3": jnp.abs(r3_complex), "psi": psi}

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

labels = [
    r'$\frac{\partial A}{\partial u_0}$', r'$\frac{\partial A}{\partial t_0}$',r'$\frac{\partial A}{\partial t_E}$',
    r'$\frac{\partial A}{\partial q}$', r'$\frac{\partial A}{\partial s}$', r'$\frac{\partial A}{\partial \alpha}$',
    r'$\frac{\partial A}{\partial \rho}$',
    r'$\frac{\partial A}{\partial q_3}$', r'$\frac{\partial A}{\partial r_3}$', r'$\frac{\partial A}{\partial \psi}$'
]
for i, l in enumerate(labels):
    axes[i+1].plot(t, jac[i])
    axes[i+1].set_ylabel(l)

axes[-1].set_xlabel('Time (day)')

plt.tight_layout()
plt.savefig("example/triple-lens-jacobian/full_jacobian_plot.png", dpi=300)
plt.show()