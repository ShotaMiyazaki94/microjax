"""Compare microlux-based contour integration with VBBinaryLensing.

This mirrors ``compare_vbbl.py`` but uses ``microjax.contour.mag_binary``
(microlux-based contour integration) instead of the caustics backend.
"""

import time
import os

# Force CPU to avoid slow GPU plugin probing on machines without CUDA
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import VBBinaryLensing

from microjax.contour import mag_binary
from microjax.point_source import critical_and_caustic_curves

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- Lens/source configuration (MulensModel/VBBL convention) ---
q = 0.05
s = 1.0
alpha_deg = 45.0
t_E = 30.0
t_0 = 0.0
u_0 = 0.0
rho = 0.03
u1 = 0.5
n_annuli = 10

# --- Observation times ---
num_points = 1000
times = jnp.linspace(-0.5 * t_E, 0.5 * t_E, num_points)

# Source trajectory in the source plane (for plotting)
alpha_rad = jnp.deg2rad(alpha_deg)
tau = (times - t_0) / t_E
y1 = -u_0 * jnp.sin(alpha_rad) + tau * jnp.cos(alpha_rad)
y2 = u_0 * jnp.cos(alpha_rad) + tau * jnp.sin(alpha_rad)
w_points = (y1 + 1j * y2).astype(jnp.complex128)

# --- Warm-up JIT ---
_ = mag_binary(
    w_points,
    rho,
    s=s,
    q=q,
    tol=1e-2,
    retol=1e-3,
    analytic=True,
).block_until_ready()
_ = mag_binary(
    w_points,
    rho,
    s=s,
    q=q,
    tol=1e-2,
    retol=1e-3,
    analytic=True,
    limb_darkening_coeff=u1,
    n_annuli=n_annuli,
).block_until_ready()

# --- microlux evaluation ---
start = time.time()
mag_mj_uniform = mag_binary(
    w_points,
    rho,
    s=s,
    q=q,
    tol=1e-2,
    retol=1e-3,
    analytic=True,
).block_until_ready()
elapsed_mj = time.time() - start
print(f"microjax (microlux, uniform): {elapsed_mj:.3f} s total ({1e3*elapsed_mj/num_points:.3f} ms/pt)")

start = time.time()
mag_mj_ld = mag_binary(
    w_points,
    rho,
    s=s,
    q=q,
    tol=1e-2,
    retol=1e-3,
    analytic=True,
    limb_darkening_coeff=u1,
    n_annuli=n_annuli,
).block_until_ready()
elapsed_mj = time.time() - start
print(f"microjax (microlux, LD u1={u1:.2f}): {elapsed_mj:.3f} s total ({1e3*elapsed_mj/num_points:.3f} ms/pt)")

# --- VBBinaryLensing reference ---
_vbbl_solver = VBBinaryLensing.VBBinaryLensing()
_vbbl_solver.RelTol = 1e-4


def mag_vbbl(points, a1):
    _vbbl_solver.a1 = float(a1)  # limb darkening coefficient
    mags = [
        _vbbl_solver.BinaryMag2(s, q, float(w.real), float(w.imag), float(rho))
        for w in jnp.asarray(points)
    ]
    return jnp.array(mags)


start = time.time()
mag_vb_uniform = mag_vbbl(w_points, 0.0)
elapsed_vb = time.time() - start
print(f"VBBinaryLensing (uniform): {elapsed_vb:.3f} s total ({1e3*elapsed_vb/num_points:.3f} ms/pt)")

start = time.time()
mag_vb_ld = mag_vbbl(w_points, u1)
elapsed_vb = time.time() - start
print(f"VBBinaryLensing (LD u1={u1:.2f}): {elapsed_vb:.3f} s total ({1e3*elapsed_vb/num_points:.3f} ms/pt)")

# --- Caustic curves for inset ---
critical_curves, caustic_curves = critical_and_caustic_curves(
    nlenses=2, npts=200, s=s, q=q
)

# --- Plot ---
def plot_case(mag_vb, mag_mj, title, out_path, label_mj="microjax (microlux)"):
    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw=dict(hspace=0.1, height_ratios=[4, 1]),
    )

    ax.plot(times, mag_vb, "-", lw=1.2, label="VBBinaryLensing")
    ax.plot(times, mag_mj, ".", ms=2.0, label=label_mj)
    ax.set_ylabel("magnification")
    ax.set_title(title)
    ax.grid(ls=":")
    ax.legend(loc="upper left")

    residual = jnp.abs(mag_mj - mag_vb) / mag_vb
    ax_res.plot(times, residual, color="tab:blue")
    ax_res.set_yscale("log")
    ax_res.set_ylabel("relative diff")
    ax_res.set_xlabel("time (days)")
    ax_res.set_ylim(1e-6, 1e-2)
    ax_res.grid(ls=":")

    ax_in = inset_axes(
        ax,
        width="60%",
        height="60%",
        bbox_to_anchor=(0.35, 0.35, 0.6, 0.6),
        bbox_transform=ax.transAxes,
    )
    ax_in.set_aspect(1)
    for cc in caustic_curves:
        ax_in.plot(cc.real, cc.imag, color="red", lw=0.7)
    # plot source trajectory and finite-source discs
    ax_in.plot(w_points.real, w_points.imag, color="tab:blue", lw=1.0, zorder=1)
    circles = [
        plt.Circle((xi, yi), radius=rho, fill=False, facecolor=None, ec="tab:blue", lw=0.8)
        for xi, yi in zip(w_points.real, w_points.imag)
    ]
    ax_in.add_collection(
        mpl.collections.PatchCollection(circles, match_original=True, alpha=0.35, zorder=0)
    )
    ax_in.scatter((-q / (1 + q)) * s, 0.0, c="k", s=15)
    ax_in.scatter((1.0 / (1 + q)) * s, 0.0, c="k", s=15)
    ax_in.set(xlabel="Re(w)", ylabel="Im(w)", xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"output: {out_path}")
    plt.close(fig)


plot_case(
    mag_vb_uniform,
    mag_mj_uniform,
    f"Microlux contour vs VBBL (uniform, rho={rho:.3f}, s={s:.2f}, q={q:.3f})",
    "example/contour_integrating/compare_binary_uniform_microlux.png",
)
plot_case(
    mag_vb_ld,
    mag_mj_ld,
    f"Microlux contour vs VBBL (LD u1={u1:.2f}, rho={rho:.3f}, s={s:.2f}, q={q:.3f})",
    "example/contour_integrating/compare_binary_ld_microlux.png",
    label_mj=f"microjax (microlux, LD u1={u1:.2f})",
)
