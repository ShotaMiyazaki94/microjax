import time
import statistics
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import VBBinaryLensing
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from microjax.inverse_ray import BinaryMagConfig, mag_binary
from microjax.multipole import _mag_hexadecapole
from microjax.point_source import _images_point_source, critical_and_caustic_curves, mag_point_source
from plot_boundary_construction import plot_boundary_construction

jax.config.update("jax_enable_x64", True)

q = 0.03
s = 0.85
alpha = jnp.deg2rad(45.0)
tE = 30.0
t0 = 0.0
u0 = 0.0
rho = 5e-3
nlenses = 2
u1 = 0.5

a = 0.5 * s
e1 = q / (1.0 + q)
_params = {"a": a, "e1": e1}
x_cm = a * (1.0 - q) / (1.0 + q)

num_points = 1000
t = jnp.linspace(-0.5 * tE, 0.5 * tE, num_points)
tau = (t - t0) / tE
y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
w_points = (y1 + 1j * y2).astype(jnp.complex128)

n_limb = 500
benchmark_repeats = 7

_vbbl_solver = VBBinaryLensing.VBBinaryLensing()
accuracy = 1e-4
_vbbl_solver.RelTol = accuracy
_vbbl_solver.Tol = accuracy
_vbbl_solver.a1 = u1


def mag_vbbl_(w0, rho):
    # Configure solver tolerances and limb darkening for each evaluation.
    return _vbbl_solver.BinaryMag2(s, q, float(w0.real), float(w0.imag), float(rho))


def mag_vbbl(w0):
    return jnp.array([mag_vbbl_(w, rho) for w in w0])


config = BinaryMagConfig(n_limb=n_limb)

# ---- Warmup (JIT compile) ----
_ = mag_binary(
    w_points,
    rho,
    config=config,
    s=s,
    q=q,
    u1=u1,
).block_until_ready()
_ = mag_point_source(w_points, s=s, q=q).block_until_ready()
z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
_, _ = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, u1=u1, **_params)

print("number of data points: %d" % (num_points))
start = time.time()
mags_poi = mag_point_source(w_points, s=s, q=q)
mags_poi.block_until_ready()
end = time.time()
print(
    "computation time: %.3f sec (%.3f ms per point) for point-source in microJAX"
    % (end - start, 1000 * (end - start) / num_points)
)

start = time.time()
z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, u1=u1, **_params)
mu_multi.block_until_ready()
end = time.time()
print(
    "computation time: %.3f sec (%.3f ms per point) for hexadecapole in microJAX"
    % (end - start, 1000 * (end - start) / num_points)
)

start = time.time()
mag_VB = mag_vbbl(w_points)
mag_VB.block_until_ready()
end = time.time()
print(
    "computation time: %.3f sec (%.3f ms per point) with VBBinaryLensing"
    % (end - start, 1000 * (end - start) / num_points)
)

boundary_samples = []
for _ in range(benchmark_repeats):
    start = time.perf_counter()
    mag_jax = mag_binary(
        w_points,
        rho,
        config=config,
        s=s,
        q=q,
        u1=u1,
    ).block_until_ready()
    boundary_samples.append(time.perf_counter() - start)
boundary_seconds = statistics.median(boundary_samples)
nonfinite = int(jnp.sum(~jnp.isfinite(mag_jax)))
if nonfinite:
    raise RuntimeError(f"microJAX returned {nonfinite} non-finite magnifications")
print(
    "computation time: %.3f sec (%.3f ms per point), median of %d, "
    "with microJAX mag_binary, n_limb=%d"
    % (boundary_seconds, 1000 * boundary_seconds / num_points, benchmark_repeats, n_limb)
)
relative_difference = jnp.abs(mag_jax - mag_VB) / mag_VB
max_residual_index = int(jnp.nanargmax(relative_difference))
print(
    "relative difference vs VBBinaryLensing: median=%.3e, p95=%.3e, max=%.3e"
    % (
        jnp.median(relative_difference),
        jnp.quantile(relative_difference, 0.95),
        jnp.max(relative_difference),
    )
)
plot_boundary_construction(
    w_points[max_residual_index],
    rho,
    s=s,
    q=q,
    n_limb=n_limb,
    time_value=float(t[max_residual_index]),
    relative_residual=float(relative_difference[max_residual_index]),
    limb_darkening=u1,
    output_path=Path("example/compare-vbbl/compare_binary_limb_dark_max_residual_icrs.png"),
)
critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)

fig, ax_ = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw=dict(hspace=0.1, height_ratios=[4, 1]))
ax = ax_[0]
ax1 = ax_[1]
ax_in = inset_axes(ax, width="60%", height="60%", bbox_transform=ax.transAxes, bbox_to_anchor=(0.35, 0.35, 0.6, 0.6))
ax_in.set_aspect(1)
ax_in.set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$")
for cc in caustic_curves:
    ax_in.plot(cc.real, cc.imag, color="red", lw=0.7)
circles = [
    plt.Circle((xi, yi), radius=rho, fill=False, facecolor=None, ec="blue", zorder=2)
    for xi, yi in zip(w_points.real, w_points.imag)
]
c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.5)
ax_in.add_collection(c)
ax_in.set_aspect(1)
ax_in.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
ax_in.plot(-q / (1 + q) * s, 0, ".", c="k")
ax_in.plot((1.0) / (1 + q) * s, 0, ".", c="k")

ax.plot(t, mag_jax, ".", label="microJAX", zorder=1)
ax.plot(t, mag_VB, "-", label="VBBinaryLensing", zorder=2)
ylim = ax.get_ylim()
ax.set_title("Limb-darkened source, rho=%.3f, s=%.2f, q=%.3f, u1=%.2f" % (rho, s, q, u1))
ax.grid(ls=":")
ax.set_ylabel("magnification")
# ax.plot(t, mags_poi, "--", label="point_source", zorder=-1, color="gray")
# ax.plot(t, mu_multi, ":", label="hexadecapole", zorder=-2, color="orange")
ax.set_ylim(ylim[0], ylim[1])
ax1.plot(t, relative_difference, "-", ms=1)
ax1.grid(ls=":")
ax1.set_yticks(10 ** jnp.arange(-4, -2, 1))
ax1.set_ylabel("relative diff")
ax1.set_yscale("log")
ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0, 10**-2, 10**-4, 10**-6], numticks=10))
ax1.set_ylim(1e-6, 1e-2)
ax.legend(loc="upper left")
ax1.set_xlabel("time (days)")
fig.savefig("example/compare-vbbl/compare_binary_limb_dark.png", dpi=200)
print("output: example/compare-vbbl/compare_binary_limb_dark.png")
plt.close()
