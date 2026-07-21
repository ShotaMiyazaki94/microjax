"""Plot the boundary-root ICRS geometry for one binary-source position."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from microjax.inverse_ray.geometry.limb import calc_source_limb
from microjax.inverse_ray.geometry.topology import define_radial_topology
from microjax.inverse_ray.roots.angular import angular_intervals_binary_roots
from microjax.inverse_ray.roots.level_set import binary_level_set
from microjax.point_source import critical_and_caustic_curves


def _display_ring_indices(
    radii: np.ndarray,
    angular_measure: np.ndarray,
    interval_ids: np.ndarray,
    count: int = 30,
) -> np.ndarray:
    """Select live rings while retaining every non-empty radial interval."""

    live = np.isfinite(angular_measure) & (angular_measure > 0.0)
    guaranteed = []
    for interval_id in np.unique(interval_ids[live]):
        candidates = np.flatnonzero(live & (interval_ids == interval_id))
        guaranteed.append(candidates[len(candidates) // 2])

    guaranteed = np.asarray(guaranteed, dtype=int)
    if guaranteed.size >= count:
        return guaranteed
    remaining = np.setdiff1d(np.flatnonzero(live), guaranteed, assume_unique=False)
    extra_count = min(count - guaranteed.size, remaining.size)
    if extra_count:
        positions = np.linspace(0, remaining.size - 1, extra_count).round().astype(int)
        guaranteed = np.concatenate((guaranteed, remaining[positions]))
    return np.sort(np.unique(guaranteed))


def plot_boundary_construction(
    w_center: complex,
    rho: float,
    *,
    s: float,
    q: float,
    n_limb: int,
    margin_r: float,
    time_value: float,
    relative_residual: float,
    limb_darkening: float,
    output_path: Path,
) -> dict:
    """Save image-plane boundary geometry at one light-curve sample."""

    output_path = Path(output_path)
    a = 0.5 * s
    e1 = q / (1.0 + q)
    shifted = a * (1.0 - q) / (1.0 + q)
    w_center = jnp.asarray(w_center, dtype=jnp.complex128)
    w_shifted = w_center - shifted
    limb_count = 2 * n_limb - 1
    image_limb, mask_limb = calc_source_limb(w_center, rho, limb_count, nlenses=2, s=s, q=q, a=a, e1=e1)
    origin_inside = binary_level_set(0.0j, w_shifted, rho, shifted, a=a, e1=e1) <= 0.0
    topology = jax.jit(
        lambda images, masks: define_radial_topology(
            images,
            masks,
            rho,
            margin_r=margin_r,
            origin_inside=origin_inside,
            binary_margin_parameters=(shifted, a, e1),
        )
    )(image_limb, mask_limb)
    topology = jax.block_until_ready(topology)
    n_intervals = int(topology.n_intervals)
    if n_intervals == 0:
        raise RuntimeError("maximum-residual source produced no radial topology")
    radial_intervals = np.asarray(topology.intervals[:n_intervals])
    radial_min = float(radial_intervals[:, 0].min())
    radial_max = float(radial_intervals[:, 1].max())
    radial_span = max(radial_max - radial_min, 0.05 * radial_max)
    radial_padding = 0.05 * radial_span
    profile_lower = max(0.0, radial_min - radial_padding)
    profile_upper = radial_max + radial_padding
    samples_per_interval = 24
    interval_fractions = (np.arange(samples_per_interval) + 0.5) / samples_per_interval
    profile_radii_by_interval = radial_intervals[:, :1] + (
        radial_intervals[:, 1:] - radial_intervals[:, :1]
    ) * interval_fractions[None, :]
    profile_radii = profile_radii_by_interval.reshape(-1)
    profile_interval_ids = np.repeat(np.arange(n_intervals), samples_per_interval)
    cell_tolerance = 64.0 * jnp.finfo(jnp.float64).eps
    angular = jax.jit(
        jax.vmap(
            lambda radius: angular_intervals_binary_roots(
                radius,
                0.0,
                2.0 * jnp.pi,
                w_shifted,
                rho,
                shifted,
                cell_tolerance,
                a=a,
                e1=e1,
                robust_roots=False,
            )
        )
    )(jnp.asarray(profile_radii))
    angular = jax.block_until_ready(angular)
    profile_intervals = np.asarray(angular.intervals)
    profile_counts = np.asarray(angular.n_intervals)
    angular_measure = np.asarray(
        [np.sum(bounds[:count, 1] - bounds[:count, 0]) for bounds, count in zip(profile_intervals, profile_counts)]
    )
    ring_indices = _display_ring_indices(profile_radii, angular_measure, profile_interval_ids)
    ring_radii = profile_radii[ring_indices]
    ring_intervals = profile_intervals[ring_indices]
    ring_counts = profile_counts[ring_indices]

    critical, caustics = critical_and_caustic_curves(nlenses=2, npts=1000, s=s, q=q)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    image_axis = axes[0]
    boundary_roots_label = "angular boundary roots"
    for radius, intervals, count in zip(ring_radii, ring_intervals, ring_counts, strict=True):
        for theta_lo, theta_hi in intervals[:count]:
            theta = np.linspace(theta_lo, theta_hi, 80)
            image_axis.plot(radius * np.cos(theta), radius * np.sin(theta), color="darkorange", lw=1.2, alpha=0.9)
            image_axis.scatter(
                radius * np.cos([theta_lo, theta_hi]),
                radius * np.sin([theta_lo, theta_hi]),
                color="tab:cyan",
                edgecolor="black",
                linewidth=0.25,
                s=7,
                zorder=4,
                label=boundary_roots_label,
            )
            boundary_roots_label = None
    image_limb_np = np.asarray(image_limb)
    mask_np = np.asarray(mask_limb)
    image_axis.scatter(
        image_limb_np[mask_np].real,
        image_limb_np[mask_np].imag,
        s=1.0,
        color="purple",
        alpha=0.65,
        label="mapped source limb",
    )
    for curve in np.asarray(critical):
        image_axis.plot(curve.real, curve.imag, color="seagreen", lw=0.55, alpha=0.75)
    for index, curve in enumerate(np.asarray(caustics)):
        image_axis.plot(
            curve.real,
            curve.imag,
            color="crimson",
            lw=0.7,
            label="caustic" if index == 0 else None,
        )
    source_theta = np.linspace(0.0, 2.0 * np.pi, 500)
    source_limb = complex(w_center) + rho * np.exp(1j * source_theta)
    image_axis.plot(source_limb.real, source_limb.imag, color="tab:blue", lw=1.0, label="source limb")
    image_axis.scatter(
        [complex(w_center).real],
        [complex(w_center).imag],
        s=12,
        color="tab:blue",
        zorder=4,
    )
    lens_positions = np.asarray([-q / (1.0 + q) * s, 1.0 / (1.0 + q) * s])
    image_axis.scatter(lens_positions, np.zeros(2), marker="x", color="black", s=35, label="lenses")
    image_axis.set_aspect("equal")
    image_axis.set_xlabel(r"image-plane $x$ [$R_E$]")
    image_axis.set_ylabel(r"image-plane $y$ [$R_E$]")
    image_axis.set_title("Exact inside-angle arcs on selected radii")
    image_axis.grid(alpha=0.2)
    image_axis.legend(frameon=False, fontsize=8, loc="lower right")

    radial_axis = axes[1]
    measure_by_interval = angular_measure.reshape(n_intervals, samples_per_interval)
    for interval_radii, interval_measure in zip(
        profile_radii_by_interval,
        measure_by_interval,
        strict=True,
    ):
        positive_measure = np.where(interval_measure > 0.0, interval_measure, np.nan)
        radial_axis.plot(interval_radii, positive_measure, color="tab:blue", lw=1.4)
    for lower, upper in radial_intervals:
        radial_axis.axvspan(lower, upper, color="tab:blue", alpha=0.055)
        radial_axis.axvline(lower, color="0.55", lw=0.35)
    radial_axis.axvline(radial_intervals[-1, 1], color="0.55", lw=0.35)
    radial_axis.set_xlabel(r"image-plane radius $r$ [$R_E$]")
    radial_axis.set_ylabel(r"inside angular measure $\Delta\theta(r)$")
    radial_axis.set_yscale("log")
    radial_axis.set_xlim(profile_lower, profile_upper)
    radial_axis.set_title("Radial topology and angular measure")
    radial_axis.grid(alpha=0.2)
    profile_label = "uniform" if limb_darkening == 0.0 else rf"$u_1={limb_darkening:g}$"
    fig.suptitle(
        rf"Maximum residual: $t={time_value:.6g}$, rel. diff $={relative_residual:.3e}$, "
        rf"{profile_label}; $s={s:g}$, $q={q:g}$, $\rho={rho:g}$"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    angular_status = np.asarray(angular.status)
    summary = {
        "time": float(time_value),
        "source_real": float(w_center.real),
        "source_imag": float(w_center.imag),
        "relative_residual": float(relative_residual),
        "limb_darkening_u1": float(limb_darkening),
        "n_limb_requested": int(n_limb),
        "n_limb_traced": int(limb_count),
        "radial_intervals": n_intervals,
        "radial_candidates_raw": int(topology.n_candidates_raw),
        "radial_status": int(topology.status),
        "angular_profile_samples": int(profile_radii.size),
        "display_rings": int(ring_radii.size),
        "angular_failures": int(np.count_nonzero(angular_status)),
        "radial_plot_min": profile_lower,
        "radial_plot_max": profile_upper,
    }
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"maximum-residual ICRS output: {output_path}")
    return summary


__all__ = ["plot_boundary_construction"]
