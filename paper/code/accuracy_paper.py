import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax, vmap, jit

from microjax.inverse_ray.extended_source import mag_uniform
from microjax.point_source import critical_and_caustic_curves
import MulensModel as mm

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def mag_vbb_binary(w0, rho, s, q, u1=0.0, accuracy=1e-05):
    e1 = 1/(1 + q)
    e2 = 1 - e1
    bl = mm.BinaryLens(e1, e2, s)
    return bl.vbbl_magnification(w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)

# Binary lens params
s, q = 1.0, 0.1
npts = 250
critical_curves, caustic_curves = critical_and_caustic_curves(npts=npts, nlenses=2, s=s, q=q)
caustic_curves = caustic_curves.reshape(-1)

acc_vbb = 1e-05
r_resolution = 1000
Nlimb = 100
margin_r = 1.0
margin_th = 1.0
bins_r = 50
bins_th = 120
cubic = True

# Parameters to loop over
rho_list = [1e-01, 1e-02, 1e-03, 1e-04]
th_res_list = [1000, 8000]
colors = ['r', 'g']

# Result containers
mags_list = [[] for _ in th_res_list]
mags_vbb_list = []
w_test_list = []

# Compute VBB magnification once for all rho
for i, rho in enumerate(rho_list):
    key = random.PRNGKey(0)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-jnp.pi, maxval=jnp.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=1.0*rho)
    w_test = caustic_curves + r * jnp.exp(1j * phi)

    mags_vbb = jnp.array([mag_vbb_binary(complex(w), rho, s, q, u1=0.0, accuracy=acc_vbb) for w in w_test])
    mags_vbb_list.append(mags_vbb)
    w_test_list.append(w_test)

    # Run both th_resolution cases
    for j, th_resolution in enumerate(th_res_list):
        def mag_mj(w):
            return mag_uniform(w, rho, s=s, q=q,
                               r_resolution=r_resolution,
                               th_resolution=th_resolution,
                               Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th,
                               margin_r=margin_r, margin_th=margin_th, cubic=cubic)
        mag_mj_jit = jit(mag_mj)

        def chunked_vmap_map(func, data, chunk_size):
            N = data.shape[0]
            pad_len = (-N) % chunk_size
            padded = jnp.pad(data, [(0, pad_len)] + [(0, 0)] * (data.ndim - 1))
            chunks = padded.reshape(-1, chunk_size, *data.shape[1:])
            def apply_vmap(chunk):
                return jit(vmap(func))(chunk)
            results = lax.map(apply_vmap, chunks)
            return results.reshape(-1, *results.shape[2:])[:N]

        mags = chunked_vmap_map(mag_mj_jit, w_test, chunk_size=100)
        mags_list[j].append(mags)

# Plotting
fig, ax = plt.subplots(1, len(rho_list) + 1, figsize=(22, 4), gridspec_kw={'wspace': 0.2})
labels = [r"$\rho_\star=0.1$", r"$\rho_\star=0.01$", r"$\rho_\star=10^{-3}$", r"$\rho_\star=10^{-4}$"]

for i in range(len(rho_list)):
    mags_vbb = mags_vbb_list[i]
    for j, th_resolution in enumerate(th_res_list):
        mags = mags_list[j][i]
        relative_error = jnp.abs((mags - mags_vbb) / mags_vbb)
        ax[i].plot(relative_error, '-', color=colors[j], alpha=0.8, lw=0.4, label=f"bins={th_resolution}")
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
        ax[i].set_yscale('log')
        ax[i].set_title(labels[i])
        ax[i].set_ylim(1e-05, 1e-2)
        ax[i].set_xlabel("Point index")
        ax[i].set_rasterization_zorder(0)
    ax[i].legend(loc='upper right')
    ax[i].set_ylabel("Relative error" if i == 0 else "")

# Final caustic plot
crit, cau = critical_and_caustic_curves(npts=1000, nlenses=2, s=s, q=q)
for cc in cau:
    ax[-1].plot(cc.real, cc.imag, color='k', lw=1)
colors = ['r', 'g', 'b', 'c']
for i, w_test in enumerate(w_test_list):
    ax[-1].plot(w_test.real, w_test.imag, ".", color=colors[i], zorder=i, ms=1, label=labels[i])
ax[-1].set_aspect('equal')
ax[-1].legend()
ax[-1].set_title("Sample points")

# Save
if cubic:
    fig.savefig("paper/figure/accuracy_compare_r%d_Nl%d_mr%.1f_mth%.1f_cub.pdf"
                % (r_resolution, Nlimb, margin_r, margin_th), bbox_inches="tight", dpi=200)
else:
    fig.savefig("paper/figure/accuracy_compare_r%d_Nl%d_mr%.1f_mth%.1f_lin.pdf"
                % (r_resolution, Nlimb, margin_r, margin_th), bbox_inches="tight", dpi=200)

plt.close()