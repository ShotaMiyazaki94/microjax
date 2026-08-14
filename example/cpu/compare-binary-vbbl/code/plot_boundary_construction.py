"""Plot the production CPU ICRS state for one binary-source position."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from microjax.inverse_ray.cpu.angular_limb_dark import (
    _profile_ray_moments,
    _profile_ray_moments_gk15,
)
from microjax.inverse_ray.cpu.angular_moment import (
    _angular_support_cells_from_trace,
    _ray_moments,
    binary_radial_level_set_coefficients,
)
from microjax.inverse_ray.cpu.cartesian_limb_dark import _trace_limb_dark_cartesian_primary
from microjax.inverse_ray.cpu.cartesian_moment import (
    _strip_widths,
    binary_line_level_set_coefficients,
)
from microjax.inverse_ray.cpu.one_shot import _prepare_uniform_one_shot
from microjax.inverse_ray.geometry.lens import binary_geometry
from microjax.inverse_ray.cpu.quadrature import GK15_X
from microjax.point_source import critical_and_caustic_curves

jax.config.update("jax_enable_x64", True)


def _production_trace(w_center, rho, *, s, q, n_limb, limb_darkening):
    """Return the trace, Cartesian chart, and diagnostics used in production."""

    if limb_darkening == 0.0:
        prepared = _prepare_uniform_one_shot(
            w_center,
            rho,
            s=s,
            q=q,
            n_limb=n_limb,
            external_available=True,
        )
        neighbors = (
            prepared.previous_limb,
            prepared.following_limb,
            prepared.previous_mask,
            prepared.following_mask,
        )
        return {
            "image_limb": prepared.image_limb,
            "physical_mask": prepared.physical_mask,
            "neighbors": neighbors,
            "topology": prepared.state.topology_uncertain,
            "ghost": prepared.state.ghost_residual_ratio * rho,
            "limb_topology": prepared.state.limb_topology,
            "axis": prepared.primary_axis,
            "cartesian_support": prepared.primary_support,
            "route": int(prepared.route),
        }

    traced = _trace_limb_dark_cartesian_primary(
        w_center,
        rho,
        s=s,
        q=q,
        n_limb=n_limb,
    )
    (
        primary_axis,
        _scout_axis,
        primary_support,
        image_limb,
        physical_mask,
        topology,
        ghost,
        limb_topology,
        neighbors,
    ) = traced
    return {
        "image_limb": image_limb,
        "physical_mask": physical_mask,
        "neighbors": neighbors,
        "topology": topology,
        "ghost": ghost,
        "limb_topology": limb_topology,
        "axis": primary_axis,
        "cartesian_support": primary_support,
        # The public tier is the authoritative LD route label.
        "route": None,
    }


def _quadrature_nodes(bounds, nodes):
    nodes = np.asarray(nodes)
    transform = 0.25 * np.pi * (nodes + 1.0)
    lower, upper = bounds
    return lower + (upper - lower) * np.sin(transform) ** 2


def _cartesian_nodes(selected_tier, selected_n_slices, n_active):
    if selected_tier == 6:
        return np.asarray(GK15_X), "nested Gauss-7/Kronrod-15"
    nodes_per_cell = selected_n_slices / max(n_active, 1)
    if nodes_per_cell <= 8.5:
        order = max(1, int(round(nodes_per_cell)))
        return np.polynomial.legendre.leggauss(order)[0], f"Gauss-Legendre-{order}"
    return np.asarray(GK15_X), "nested Gauss-7/Kronrod-15"


def _polar_nodes(selected_tier, limb_darkening):
    if limb_darkening == 0.0:
        if selected_tier == 7:
            return (
                np.polynomial.legendre.leggauss(32)[0],
                "independent Gauss-Legendre 24/32",
            )
        if selected_tier == 9:
            fine_intervals = 48
            nodes = np.cos(np.pi * np.arange(fine_intervals + 1) / fine_intervals)[::-1]
            return nodes[1:-1], "nested Clenshaw-Curtis 24/48"
        return np.asarray(GK15_X), "nested Gauss-7/Kronrod-15"
    if selected_tier == 7:
        return np.asarray(GK15_X), "angular Gauss-Kronrod-15"
    order = 24 if selected_tier == 9 else 12
    return np.polynomial.legendre.leggauss(order)[0], f"angular Gauss-Legendre-{order}"


def _plot_image_plane(axis, trace, critical, s, q):
    image_limb = np.asarray(trace["image_limb"])
    physical_mask = np.asarray(trace["physical_mask"])
    for index, curve in enumerate(np.asarray(critical)):
        axis.plot(
            curve.real,
            curve.imag,
            color="seagreen",
            lw=0.65,
            label="critical curve" if index == 0 else None,
        )
    axis.scatter(
        image_limb[physical_mask].real,
        image_limb[physical_mask].imag,
        s=2.0,
        color="purple",
        alpha=0.72,
        label="source-limb images",
    )
    lenses = np.asarray([-q / (1.0 + q) * s, 1.0 / (1.0 + q) * s])
    axis.scatter(lenses, np.zeros(2), marker="x", s=35, color="black", label="lenses")
    axis.set_aspect("equal")
    axis.set(xlabel=r"image-plane $x$ [$R_E$]", ylabel=r"image-plane $y$ [$R_E$]")
    axis.set_title("Image profile: mapped source limb")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7, loc="lower center", ncol=3)


def _contiguous_domains(cells):
    """Merge adjacent support cells for the compact domain strip."""

    if len(cells) == 0:
        return np.empty((0, 2), dtype=float)
    tolerance = 512.0 * np.finfo(cells.dtype).eps * max(
        1.0, float(np.max(np.abs(cells)))
    )
    separated = cells[1:, 0] > cells[:-1, 1] + tolerance
    starts = np.concatenate(([0], np.flatnonzero(separated) + 1))
    ends = np.concatenate((np.flatnonzero(separated), [len(cells) - 1]))
    return np.column_stack((cells[starts, 0], cells[ends, 1]))


def _cartesian_width_profile(abscissa, projection_axis, w_center, rho, s, q):
    """Evaluate the production strip-width kernel for plot-only samples."""

    lens = binary_geometry(s, q)
    lens_radius = jnp.maximum(
        jnp.abs(lens.shifted - lens.a), jnp.abs(lens.shifted + lens.a)
    )
    radial_offset = jnp.abs(w_center) + rho - lens_radius
    ordinate_bound = lens_radius + 0.5 * (
        radial_offset + jnp.sqrt(radial_offset**2 + 4.0)
    )
    ordinate_bound *= 1.0 + 32.0 * jnp.finfo(w_center.real.dtype).eps
    coefficients = jax.vmap(
        lambda u: binary_line_level_set_coefficients(
            u * projection_axis,
            1.0j * projection_axis,
            w_center,
            rho,
            s=s,
            q=q,
        )
    )(jnp.asarray(abscissa, dtype=w_center.real.dtype))
    widths, invalid = _strip_widths(
        coefficients,
        continuation="ea_fixed28",
        ordinate_bound=ordinate_bound,
        source_radius=rho,
    )
    return np.asarray(jax.block_until_ready(widths)), np.asarray(invalid)


def _polar_integrand_profile(
    angles,
    w_center,
    rho,
    s,
    q,
    selected_tier,
    limb_darkening,
):
    """Evaluate the production radial moment passed to angular quadrature."""

    angles = jnp.asarray(angles, dtype=w_center.real.dtype)
    coefficients = jax.vmap(
        lambda theta: binary_radial_level_set_coefficients(
            theta,
            w_center,
            rho,
            s=s,
            q=q,
        )
    )(angles)
    if limb_darkening == 0.0:
        profile, invalid = _ray_moments(
            coefficients,
            root_mode="companion",
            stable_context=(angles, w_center, rho, s, q),
        )
    elif selected_tier == 7:
        uniform, residual, _coarse, invalid = _profile_ray_moments_gk15(
            angles,
            coefficients,
            w_center,
            rho,
            s=s,
            q=q,
        )
        profile = (1.0 - limb_darkening) * uniform + limb_darkening * residual
    else:
        n_radial = 12 if selected_tier == 9 else 8
        uniform, residual, invalid = _profile_ray_moments(
            angles,
            coefficients,
            w_center,
            rho,
            s=s,
            q=q,
            n_radial=n_radial,
            estimate_error=False,
        )
        profile = (1.0 - limb_darkening) * uniform + limb_darkening * residual
    return (
        np.asarray(jax.block_until_ready(profile)),
        np.asarray(jax.block_until_ready(invalid)),
    )


def _plot_cartesian_chart(
    axis,
    domain_axis,
    trace,
    w_center,
    rho,
    s,
    q,
    selected_tier,
    selected_n_slices,
):
    projection_axis = complex(trace["axis"])
    cells, active, *_ = trace["cartesian_support"]
    cells = np.asarray(cells)
    active = np.asarray(active, dtype=bool)
    active_cells = cells[active]
    nodes, rule = _cartesian_nodes(selected_tier, selected_n_slices, len(active_cells))

    profile_count = 48
    fractions = (np.arange(profile_count) + 0.5) / profile_count
    profile_abscissa = active_cells[:, :1] + (
        active_cells[:, 1:] - active_cells[:, :1]
    ) * fractions[None, :]
    profile_widths, profile_invalid = _cartesian_width_profile(
        profile_abscissa.reshape(-1), projection_axis, w_center, rho, s, q
    )
    profile_widths = profile_widths.reshape(len(active_cells), profile_count)
    profile_invalid = profile_invalid.reshape(len(active_cells), profile_count)

    cell_colors = ("#cfe8f5", "#9ecae1")
    for index, bounds in enumerate(active_cells):
        axis.axvspan(
            bounds[0],
            bounds[1],
            color=cell_colors[index % len(cell_colors)],
            alpha=0.38,
            lw=0,
        )
        abscissa = _quadrature_nodes(bounds, nodes)
        axis.axvline(bounds[0], color="0.55", lw=0.4, alpha=0.65)
        axis.vlines(
            abscissa,
            0.0,
            1.0,
            transform=axis.get_xaxis_transform(),
            color="tab:orange",
            lw=0.42,
            alpha=0.6,
        )
        valid_width = np.where(
            (profile_invalid[index] == 0) & (profile_widths[index] > 0.0),
            profile_widths[index],
            np.nan,
        )
        axis.plot(
            profile_abscissa[index],
            valid_width,
            color="tab:blue",
            lw=1.35,
            label=r"$L(u)$" if index == 0 else None,
            zorder=3,
        )
    axis.axvline(active_cells[-1, 1], color="0.55", lw=0.4, alpha=0.65)

    domains = _contiguous_domains(active_cells)
    for lower, upper in domains:
        domain_axis.barh(
            0.5,
            upper - lower,
            left=lower,
            height=0.72,
            color=cell_colors[0],
            edgecolor="tab:blue",
            linewidth=0.8,
        )
    shared_boundaries = active_cells[:-1, 1][
        np.isclose(
            active_cells[:-1, 1],
            active_cells[1:, 0],
            rtol=0.0,
            atol=512.0
            * np.finfo(active_cells.dtype).eps
            * max(1.0, float(np.max(np.abs(active_cells)))),
        )
    ]
    domain_axis.vlines(
        shared_boundaries, 0.14, 0.86, color="tab:blue", linewidth=0.65
    )

    span = active_cells[-1, 1] - active_cells[0, 0]
    padding = 0.05 * max(span, np.finfo(active_cells.dtype).eps)
    axis.set(
        xlim=(active_cells[0, 0] - padding, active_cells[-1, 1] + padding),
        ylabel=r"inside strip width $L(u)$",
        title=f"Strip width on non-overlapping Cartesian cells ({rule})",
    )
    axis.set_yscale("log")
    axis.tick_params(axis="x", which="both", labelbottom=False)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=8, loc="upper right")

    domain_axis.set_ylim(0.0, 1.0)
    domain_axis.set_yticks([])
    domain_axis.set_xlabel(r"strip coordinate $u=\mathrm{Re}(z\bar a)$")
    domain_axis.set_ylabel(
        "integration\ndomain", rotation=0, ha="right", va="center", fontsize=8
    )
    domain_axis.grid(False)
    for side in ("left", "right", "top"):
        domain_axis.spines[side].set_visible(False)

    angle = np.rad2deg(np.angle(projection_axis))
    return {
        "chart": "cartesian",
        "projection_axis_real": projection_axis.real,
        "projection_axis_imag": projection_axis.imag,
        "projection_angle_deg": angle,
        "n_support_cells": int(len(active_cells)),
        "quadrature_rule": rule,
    }


def _plot_polar_chart(axis, trace, w_center, rho, s, q, selected_tier, limb_darkening):
    support = _angular_support_cells_from_trace(
        w_center,
        rho,
        s=s,
        q=q,
        physical_limb=trace["image_limb"],
        physical_mask=trace["physical_mask"],
        topology_uncertain=trace["topology"],
        minimum_ghost_residual=trace["ghost"],
        limb_topology=trace["limb_topology"],
        neighbors=trace["neighbors"],
    )
    cells = np.asarray(support.cells)
    active = np.asarray(support.active, dtype=bool)
    active_cells = cells[active]
    nodes, rule = _polar_nodes(selected_tier, limb_darkening)

    profile_count = 96
    fractions = (np.arange(profile_count) + 0.5) / profile_count
    profile_angles = active_cells[:, :1] + (
        active_cells[:, 1:] - active_cells[:, :1]
    ) * fractions[None, :]
    profile, profile_invalid = _polar_integrand_profile(
        profile_angles.reshape(-1),
        w_center,
        rho,
        s,
        q,
        selected_tier,
        limb_darkening,
    )
    profile = profile.reshape(len(active_cells), profile_count)
    profile_invalid = profile_invalid.reshape(len(active_cells), profile_count)

    for index, bounds in enumerate(active_cells):
        axis.axvspan(bounds[0], bounds[1], color=("#dbeaf4", "#edf4f8")[index % 2], alpha=0.75)
        theta = _quadrature_nodes(bounds, nodes)
        axis.vlines(
            theta,
            0.0,
            1.0,
            transform=axis.get_xaxis_transform(),
            color="tab:orange",
            lw=0.42,
            alpha=0.6,
        )
        valid_profile = np.where(
            (profile_invalid[index] == 0) & (profile[index] > 0.0),
            profile[index],
            np.nan,
        )
        axis.plot(
            profile_angles[index],
            valid_profile,
            color="tab:blue",
            lw=1.35,
            label=(
                r"$M_0(\theta)$"
                if index == 0 and limb_darkening == 0.0
                else r"$M_{\rm LD}(\theta)$"
                if index == 0
                else None
            ),
            zorder=3,
        )
    axis.set(
        xlim=(0.0, 2.0 * np.pi),
        xlabel=r"image angle $\theta$",
        ylabel=(
            r"radial area moment $M_0(\theta)$"
            if limb_darkening == 0.0
            else r"radial brightness moment $M_{\rm LD}(\theta)$"
        ),
        title=f"Radial moment on polar cells ({rule})",
    )
    axis.set_xticks(np.linspace(0.0, 2.0 * np.pi, 5), ("0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"))
    axis.set_yscale("log")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    return {
        "chart": "polar",
        "n_support_cells": int(len(active_cells)),
        "quadrature_rule": rule,
        "integrand": "M0(theta)" if limb_darkening == 0.0 else "MLD(theta)",
    }


def plot_boundary_construction(
    w_center: complex,
    rho: float,
    *,
    s: float,
    q: float,
    n_limb: int,
    time_value: float,
    relative_residual: float,
    limb_darkening: float,
    selected_tier: int,
    selected_n_slices: int,
    output_path: Path,
) -> dict:
    """Save an exact production-route diagnostic at one ICRS sample."""

    if selected_tier not in (5, 6, 7, 8, 9):
        raise ValueError(f"tier {selected_tier} is not a CPU ICRS route")
    output_path = Path(output_path)
    w_center = jnp.asarray(w_center, dtype=jnp.complex128)
    rho_array = jnp.asarray(rho, dtype=jnp.float64)
    trace = jax.block_until_ready(
        _production_trace(
            w_center,
            rho_array,
            s=jnp.asarray(s, dtype=jnp.float64),
            q=jnp.asarray(q, dtype=jnp.float64),
            n_limb=n_limb,
            limb_darkening=limb_darkening,
        )
    )
    critical, _ = critical_and_caustic_curves(nlenses=2, npts=1000, s=s, q=q)
    fig = plt.figure(figsize=(12.0, 5.4), layout="constrained")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.10),
        hspace=0.06,
        wspace=0.08,
    )
    image_axis = fig.add_subplot(grid[:, 0])
    _plot_image_plane(image_axis, trace, critical, s, q)
    if selected_tier in (5, 6):
        chart_axis = fig.add_subplot(grid[0, 1])
        domain_axis = fig.add_subplot(grid[1, 1], sharex=chart_axis)
        chart_summary = _plot_cartesian_chart(
            chart_axis,
            domain_axis,
            trace,
            w_center,
            rho_array,
            jnp.asarray(s, dtype=jnp.float64),
            jnp.asarray(q, dtype=jnp.float64),
            selected_tier,
            selected_n_slices,
        )
    else:
        chart_axis = fig.add_subplot(grid[:, 1])
        chart_summary = _plot_polar_chart(
            chart_axis,
            trace,
            w_center,
            rho_array,
            jnp.asarray(s),
            jnp.asarray(q),
            selected_tier,
            limb_darkening,
        )

    profile_label = "uniform" if limb_darkening == 0.0 else rf"$u_1={limb_darkening:g}$"
    fig.suptitle(
        rf"Largest status-zero ICRS residual: $t={time_value:.6g}$, "
        rf"rel. diff $={relative_residual:.3e}$, tier {selected_tier} "
        rf"({chart_summary['chart']})" "\n"
        rf"profile: {profile_label}; "
        rf"$s={s:g}$, $q={q:g}$, $\rho={rho:g}$"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "time": float(time_value),
        "source_real": float(w_center.real),
        "source_imag": float(w_center.imag),
        "relative_residual": float(relative_residual),
        "limb_darkening_u1": float(limb_darkening),
        "selected_tier": int(selected_tier),
        "selected_n_slices": int(selected_n_slices),
        "n_limb": int(n_limb),
        "production_route": chart_summary,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"largest-residual {chart_summary['chart']} CPU ICRS diagnostic: {output_path}"
    )
    return summary


__all__ = ["plot_boundary_construction"]
