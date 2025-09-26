import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax 
import jax
jax.config.update("jax_enable_x64", True)

try:
    if not jax.devices("cuda"):
        print("[Warning] No CUDA device detected by JAX.\n"
              "          This example may be impractically slow on CPU.")
except Exception:
    pass
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import critical_and_caustic_curves

N_POINTS = 500
N_LIMB   = 500
R_NUM = 500
TH_NUM = 500
TIME_CHUNK = 50         # chunk_size
MAX_FULL_CALLS = 500

# Parameters
t0, tE, u0 = 0.0, 10.0, 0.1
q, s, alpha = 0.1, 1.1, jnp.deg2rad(50) 
rho = 0.02
q3, r3_complex = 0.03, 0.3 + 1.2j 
psi = jnp.arctan2(r3_complex.imag, r3_complex.real)
times = t0 + jnp.linspace(-0.5*tE, tE, N_POINTS)

def map_chunked(func, arr, chunk_size: int):
    M = arr.shape[0]
    pad = (-M) % chunk_size
    arr_p = jnp.pad(arr, (0, pad))
    num_chunks = arr_p.shape[0] // chunk_size
    chunks = arr_p.reshape(num_chunks, chunk_size)
    def f_chunk(x):
        return vmap(func)(x)                # shape: (chunk_size,)
    outs = lax.map(f_chunk, chunks)         # shape: (num_chunks, chunk_size)
    outs = outs.reshape(-1)[:M]             # (M,)
    return outs

@jit
def get_mag(params):
    t0, tE, u0, q, s, alpha, rho, q3, r3, psi = params
    tau = (times - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
    w_points = jnp.array(y1 + 1j * y2, dtype=complex)
    _params = {"q": q, "s": s, "q3": q3, "r3": r3, "psi": psi}
    def mag_mj(w):
        return mag_uniform(w, rho, nlenses=3, **_params, Nlimb=N_LIMB, 
                           r_resolution=R_NUM, th_resolution=TH_NUM)
    magnifications = map_chunked(mag_mj, w_points, chunk_size=TIME_CHUNK)
    return magnifications

def jvp_param_loop(f, x):
    nparam = x.shape[0]
    outs = []
    for k in range(nparam):
        e = jnp.zeros_like(x).at[k].set(1.0)
        _, jvp_val = jax.jvp(f, (x,), (e,))
        outs.append(jvp_val)
    return jnp.stack(outs, axis=0)

if(1):
    import time
    params = jnp.array([t0, tE, u0, q, s, alpha, rho, q3, jnp.abs(r3_complex), psi])
    start = time.time() 
    _ = get_mag(params)
    _.block_until_ready()
    end = time.time()
    print("warmup (JIT): %.3f sec"%(end - start))
    start = time.time()
    A = get_mag(params)
    A.block_until_ready()
    end = time.time()
    print("mag finish: %.3f sec"%(end - start))
    start = time.time()
    _ = jvp_param_loop(get_mag, params) 
    _.block_until_ready()
    end = time.time()
    print("warmup (JVP, JIT): %.3f sec"%(end - start))
    start = time.time()
    J = jvp_param_loop(get_mag, params)
    J.block_until_ready()
    end = time.time()
    print("jac (param-loop) finish: %.3f sec"%(end - start))

    t_np = np.array(A).T
    jac_np = np.array(J)
    np.savetxt("example/triple-lens-jacobian/magnification.csv", t_np, delimiter=",")
    np.save("example/triple-lens-jacobian/jacobian_full.npy", jac_np)

A = np.loadtxt("example/triple-lens-jacobian/magnification.csv", delimiter=",")
tau = (times - t0)/tE
y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha)
w_points = jnp.array(y1 + 1j * y2, dtype=complex)
jac = np.load("example/triple-lens-jacobian/jacobian_full.npy")
param_names = ['t0', 'tE', 'u0', 'q', 's', 'alpha', 'rho', 'q3', 'r3', 'psi']
n_params = jac.shape[0]

fig, axes = plt.subplots(11, 1, figsize=(12, 10), sharex=True,
                         gridspec_kw={'height_ratios': [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'wspace':0.3}) 
axes[0].plot(times, A, label='Magnification $A(t)$', color='black')
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
    r'$\frac{\partial A}{\partial t_0}$', 
    r'$\frac{\partial A}{\partial t_E}$',
    r'$\frac{\partial A}{\partial u_0}$',
    r'$\frac{\partial A}{\partial q}$', 
    r'$\frac{\partial A}{\partial s}$', 
    r'$\frac{\partial A}{\partial \alpha}$',
    r'$\frac{\partial A}{\partial \rho}$',
    r'$\frac{\partial A}{\partial q_3}$', 
    r'$\frac{\partial A}{\partial r_3}$', 
    r'$\frac{\partial A}{\partial \psi}$'
]
for i, l in enumerate(labels):
    axes[i+1].plot(times, jac[i])
    axes[i+1].set_ylabel(l)
axes[-1].set_xlabel('Time (day)')
plt.savefig("example/triple-lens-jacobian/full_jac.png", dpi=300, bbox_inches='tight')
plt.show()