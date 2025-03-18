import jax
jax.config.update("jax_enable_x64", True)
#jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random, lax
import gc

from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import critical_and_caustic_curves
import MulensModel as mm

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def mag_vbb_binary(w0, rho, s, q, u1=0.0, accuracy=1e-05):
    e1 = 1/(1 + q)
    e2 = 1 - e1
    bl = mm.BinaryLens(e1, e2, s)
    return bl.vbbl_magnification(
        w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1
    )

s, q = 1.0, 0.1
# 1000  points on caustic curve
npts = 250
critical_curves, caustic_curves = critical_and_caustic_curves(
    npts=npts, nlenses=2, s=s, q=q
)
caustic_curves = caustic_curves.reshape(-1)

acc_vbb = 1e-05
r_resolution  = 500
th_resolution = 2000
Nlimb = 500
cubic = True

mags_vbb_list = []
mags_list = []
w_test_list = []

#rho_list = [1e-03, 8e-04, 5e-04, 3e-04, 1e-4]
rho_list = [1e-01, 1e-02, 1e-03]
#rho_list = [1e-01, 1e-02, 1e-03, 1e-04]

for i, rho in enumerate(rho_list):
    print(f"rho = {rho}")
    # Generate 1000 random test points within 2 source radii away from the caustic points 
    key = random.PRNGKey(1)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-jnp.pi, maxval=jnp.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=1.5*rho)
    w_test = caustic_curves + r*jnp.exp(1j*phi)

    mags_vbb = jnp.array([mag_vbb_binary(complex(w), rho, s, q, u1=0.0, accuracy=acc_vbb)
                          for w in w_test
                          ])
    #margin_th = [0.1, 1.0, 10.0, 100.0]
    mag_mj  = lambda w: mag_uniform(w, rho, s=s, q=q, 
                                    r_resolution=r_resolution, 
                                    th_resolution=th_resolution, 
                                    Nlimb=Nlimb, bins_r=50, bins_th=120, 
                                    margin_r=0.5, margin_th=0.5, cubic=cubic)
    #mag_mj  = lambda w: mag_uniform(w, rho, s=s, q=q, r_resolution=r_resolution, 
    #                                th_resolution=th_resolution, cubic=cubic, 
    #                                Nlimb=Nlimb, offset_r=0.1, offset_th=0.1)
    #magn    = jax.jit(jax.vmap(mag_mj, in_axes=(0,)))
    def chunked_vmap(func, data, chunk_size):
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            results.append(jax.vmap(func)(chunk))
        return jnp.concatenate(results)

    chunk_size = 2000  # メモリ消費を調整するため適宜変更
    mags = chunked_vmap(mag_mj, w_test, chunk_size)

    w_test_list.append(w_test) 
    mags_vbb_list.append(mags_vbb)
    mags_list.append(mags)

w_test_list = jnp.array(w_test_list)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
crit, cau = critical_and_caustic_curves(npts=1000, nlenses=2, s=s, q=q)
for cc in cau:
    ax.plot(cc.real, cc.imag, color='k', lw=1)
colors = ['r', 'g', 'b', 'c']
for i, w_test in enumerate(w_test_list):
    ax.plot(w_test.real, w_test.imag, ".", color=colors[i], zorder=i, ms=1)
ax.set_aspect('equal')
#fig.savefig("tests/integrate/inverse_ray/figs/accuracy_caustic.pdf",bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(1,len(rho_list)+1, figsize=(16, 4), gridspec_kw={'wspace':0.2})
labels = [r"$\rho_\star=0.1$", r"$\rho_\star=0.01$", r"$\rho_\star=10^{-3}$", r"$\rho_\star=10^{-4}$"]
for i in range(len(rho_list)):
    mags = mags_list[i]
    mags_vbb = mags_vbb_list[i]
    relative_error = jnp.abs((mags - mags_vbb)/mags_vbb) 
    ax[i].plot(relative_error, 'k-', alpha=0.9, zorder=-1, lw=0.3)
    ax[i].xaxis.set_minor_locator(AutoMinorLocator())
    ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    ax[i].set_yscale('log')
    ax[i].set_title(labels[i])
    ax[i].set_ylim(1e-05, 1e-1)
    ax[i].set_xlabel("Point index", labelpad=25)
    #ax[i].set_xlim(-10, 1010)
    ax[i].set_rasterization_zorder(0)
    if jnp.any(relative_error > 1e-03):
        mask = relative_error > 1e-03
        print(f"rho = {rho_list[i]}, w_test = {w_test[mask]}")
        #print(f"rho = {rho_list[i]}, w_test = {w_test[mask]}", relative_error[mask])
colors = ['r', 'g', 'b', 'c']
for cc in cau:
    ax[-1].plot(cc.real, cc.imag, color='k', lw=1)
for i, w_test in enumerate(w_test_list):
    ax[-1].plot(w_test.real, w_test.imag, ".", color=colors[i], zorder=i, ms=1, label="$\\rho=$%.3f"%(rho_list[i]))
ax[-1].set_aspect('equal')
ax[-1].legend()
ax[0].set_ylabel("Relative error")
if cubic:
    fig.savefig("tests/integrate/inverse_ray/accuracy/accuracy_r%d_th%d_Nl%d_cub2.pdf"%(r_resolution, th_resolution, Nlimb),bbox_inches="tight")
else:
    fig.savefig("tests/integrate/inverse_ray/accuracy/accuracy_r%d_th%d_Nl%d_lin2.pdf"%(r_resolution, th_resolution, Nlimb),bbox_inches="tight")
plt.show()