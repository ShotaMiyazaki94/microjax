"""Compare ``microjax`` caustic magnifications with VBBinaryLensing.

Inspired by ``example/compare-vbbl/compare_binary_uniform.py`` this local helper
evaluates the finite-source light curve with ``microjax.caustics.magnifications``
and benchmarks it against ``VBBinaryLensing.BinaryMag2``.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import VBBinaryLensing

from microjax.caustics.lightcurve import magnifications
from microjax.point_source import critical_and_caustic_curves

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


# --- Lens/source configuration (same convention as the example script) ---
q = 0.05
s = 1.0
alpha = jnp.deg2rad(45.0)
t_E = 30.0
t_0 = 0.0
u_0 = 0.0
rho = 0.005
nlenses = 2

a = 0.5 * s
e1 = q / (1.0 + q)
params = {"s": s, "q": q}


# --- Source trajectory ---
num_points = 1000
t = jnp.linspace(-0.5 * t_E, 0.5 * t_E, num_points)
tau = (t - t_0) / t_E
y1 = -u_0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
y2 = u_0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
w_points = (y1 + 1j * y2).astype(jnp.complex128)

npts_limb = 200  # number of points to sample the limb of the source

# --- Warm-up JIT compilation ---
_ = magnifications(
    w_points,
    rho,
    nlenses=nlenses,
    npts_limb=npts_limb,
    limb_darkening=False,
    **params,
).block_until_ready()


# --- microJAX evaluation ---
start = time.time()
mag_mj = magnifications(
    w_points,
    rho,
    nlenses=nlenses,
    npts_limb=npts_limb,
    limb_darkening=False,
    **params,
).block_until_ready()
end = time.time()
print(
    "microjax: %.3f s total (%.3f ms/point)"
    % (end - start, 1e3 * (end - start) / num_points)
)


# --- VBBinaryLensing reference ---
_vbbl_solver = VBBinaryLensing.VBBinaryLensing()
_vbbl_solver.RelTol = 1e-4
_vbbl_solver.a1 = 0.0  # limb darkening coefficient


def mag_vbbl(points):
    points_np = jnp.asarray(points)
    mags = [
        _vbbl_solver.BinaryMag2(s, q, float(w.real), float(w.imag), float(rho))
        for w in points_np
    ]
    return jnp.array(mags)


start = time.time()
mag_vb = mag_vbbl(w_points)
end = time.time()
print(
    "VBBinaryLensing: %.3f s total (%.3f ms/point)"
    % (end - start, 1e3 * (end - start) / num_points)
)


# --- Plotting ---
critical_curves, caustic_curves = critical_and_caustic_curves(
    nlenses=nlenses, npts=200, s=s, q=q
)

fig, (ax, ax_res) = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw=dict(hspace=0.1, height_ratios=[4, 1]),
)

ax.plot(t, mag_vb, "-", lw=1.2, label="VBBinaryLensing")
ax.plot(t, mag_mj, ".", ms=2.0, label="microjax (caustics)")
ax.set_ylabel("magnification")
ax.set_title(f"Uniform source, rho={rho:.3f}, s={s:.2f}, q={q:.3f}")
ax.grid(ls=":")
ax.legend(loc="upper left")

residual = jnp.abs(mag_mj - mag_vb) / mag_vb
ax_res.plot(t, residual, color="tab:blue")
ax_res.set_yscale("log")
ax_res.set_ylabel("relative diff")
ax_res.set_xlabel("time (days)")
ax_res.set_ylim(1e-6, 1e-2)
ax_res.grid(ls=":")

ax_in = inset_axes(ax, width="60%", height="60%", bbox_to_anchor=(0.35, 0.35, 0.6, 0.6), bbox_transform=ax.transAxes)
ax_in.set_aspect(1)
for cc in caustic_curves:
    ax_in.plot(cc.real, cc.imag, color="red", lw=0.7)
ax_in.plot(w_points.real, w_points.imag, color="tab:blue", lw=1.0)
ax_in.scatter((-q / (1 + q)) * s, 0.0, c="k", s=15)
ax_in.scatter((1.0 / (1 + q)) * s, 0.0, c="k", s=15)
ax_in.set(xlabel="Re(w)", ylabel="Im(w)", xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
fig.savefig("example/contour_integrating/compare_binary_uniform.png", dpi=200, bbox_inches="tight")
print("output: example/contour_integrating/compare_binary_uniform.png")
plt.show()