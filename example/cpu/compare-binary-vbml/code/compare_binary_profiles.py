"""Compare uniform and limb-darkened CPU ICRS on one binary-lens track."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import VBMicrolensing
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from microjax.inverse_ray import BinaryMagConfig, mag_binary
from microjax.multipole import _mag_hexadecapole
from microjax.point_source import (
    _images_point_source,
    critical_and_caustic_curves,
    mag_point_source,
)
from plot_boundary_construction import plot_boundary_construction

jax.config.update("jax_enable_x64", True)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "outputs"
VBML_TIMING_TOL = 1.0e-4
VBML_RESIDUAL_TOL = 1.0e-6


class ProfileResult(NamedTuple):
    name: str
    u1: float
    vbml: np.ndarray
    vbml_timing: np.ndarray
    cpu: np.ndarray
    relative_error: np.ndarray
    vbml_reference_error: np.ndarray
    certified: np.ndarray
    cpu_result: object
    vbml_seconds: float
    vbml_residual_seconds: float
    cpu_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare uniform and limb-darkened CPU ICRS in one figure."
    )
    parser.add_argument("--s", type=float, default=0.9)
    parser.add_argument("--q", type=float, default=1e-2)
    parser.add_argument("--rho", type=float, default=5e-3)
    parser.add_argument("--u1", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--alpha-deg", type=float, default=40.0)
    parser.add_argument("--t-e", type=float, default=30.0)
    parser.add_argument("--u0", type=float, default=0.0)
    parser.add_argument("--half-span-te", type=float, default=0.5)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--n-limb", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def trajectory(args: argparse.Namespace) -> tuple[jax.Array, jax.Array]:
    time_days = jnp.linspace(
        -args.half_span_te * args.t_e,
        args.half_span_te * args.t_e,
        args.points,
    )
    tau = time_days / args.t_e
    alpha = jnp.deg2rad(args.alpha_deg)
    real = -args.u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
    imag = args.u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
    return time_days, (real + 1.0j * imag).astype(jnp.complex128)


def median_runtime(function, *, repeats: int) -> tuple[object, float]:
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        if hasattr(result, "magnification"):
            result.magnification.block_until_ready()
        else:
            jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def benchmark_approximations(
    points: jax.Array,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    """Time the shared point-source and profile-dependent hexadecapole calls."""

    a = 0.5 * args.s
    e1 = args.q / (1.0 + args.q)
    x_cm = a * (1.0 - args.q) / (1.0 + args.q)
    lens_parameters = {"a": a, "e1": e1}

    mag_point_source(points, s=args.s, q=args.q).block_until_ready()
    start = time.perf_counter()
    mag_point_source(points, s=args.s, q=args.q).block_until_ready()
    point_seconds = time.perf_counter() - start

    def evaluate_hexadecapole(u1: float):
        images, image_mask = _images_point_source(
            points - x_cm,
            nlenses=2,
            **lens_parameters,
        )
        magnification, _ = _mag_hexadecapole(
            images,
            image_mask,
            args.rho,
            nlenses=2,
            u1=u1,
            **lens_parameters,
        )
        magnification.block_until_ready()

    hexadecapole_seconds = {}
    for profile, u1 in (("uniform", 0.0), ("limb_dark", args.u1)):
        evaluate_hexadecapole(u1)
        start = time.perf_counter()
        evaluate_hexadecapole(u1)
        hexadecapole_seconds[profile] = time.perf_counter() - start
    return point_seconds, hexadecapole_seconds


def evaluate_profile(
    name: str,
    u1: float,
    points: jax.Array,
    args: argparse.Namespace,
) -> ProfileResult:
    config = BinaryMagConfig(n_limb=args.n_limb)
    timing_solver = VBMicrolensing.VBMicrolensing()
    timing_solver.RelTol = VBML_TIMING_TOL
    timing_solver.Tol = VBML_TIMING_TOL
    timing_solver.a1 = u1
    residual_solver = VBMicrolensing.VBMicrolensing()
    residual_solver.RelTol = VBML_RESIDUAL_TOL
    residual_solver.Tol = VBML_RESIDUAL_TOL
    residual_solver.a1 = u1

    def evaluate_vbml(solver):
        # VBML 5.5 stores the linear-LD coefficient in ``solver.a1``.
        # BinaryMag2 dispatches to BinaryMagDark with the configured tolerance;
        # BinaryMagDark's sixth Python argument is Tolnew, despite its docstring.
        return np.asarray(
            [
                solver.BinaryMag2(
                    args.s,
                    args.q,
                    float(point.real),
                    float(point.imag),
                    args.rho,
                )
                for point in np.asarray(points)
            ]
        )

    def evaluate_cpu():
        return mag_binary(
            points,
            args.rho,
            config=config,
            s=args.s,
            q=args.q,
            u1=u1,
            backend="cpu",
            return_info=True,
        )

    # Compile before measuring steady-state CPU execution.
    evaluate_cpu().magnification.block_until_ready()
    vbml_timing, vbml_seconds = median_runtime(
        lambda: evaluate_vbml(timing_solver), repeats=args.repeats
    )
    vbml, vbml_residual_seconds = median_runtime(
        lambda: evaluate_vbml(residual_solver), repeats=1
    )
    cpu_result, cpu_seconds = median_runtime(evaluate_cpu, repeats=args.repeats)
    cpu = np.asarray(cpu_result.magnification)
    if not np.all(np.isfinite(cpu)):
        raise RuntimeError(f"{name}: CPU ICRS returned non-finite magnifications")
    relative_error = np.abs(cpu - vbml) / vbml
    vbml_reference_error = np.abs(vbml_timing - vbml) / vbml
    certified = np.asarray(cpu_result.status) == 0
    return ProfileResult(
        name,
        u1,
        vbml,
        vbml_timing,
        cpu,
        relative_error,
        vbml_reference_error,
        certified,
        cpu_result,
        vbml_seconds,
        vbml_residual_seconds,
        cpu_seconds,
    )


def count_dictionary(values) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return dict(zip(map(str, unique.tolist()), counts.tolist()))


def profile_summary(
    result: ProfileResult,
    args: argparse.Namespace,
    *,
    hexadecapole_seconds: float,
) -> dict:
    def error_statistics(values: np.ndarray) -> tuple[dict[str, float], int]:
        accepted_error = values[result.certified]
        residuals_above_target = int(
            np.sum(result.certified & (values > args.rtol))
        )
        return (
            {
                "median": float(np.median(accepted_error)),
                "p95": float(np.quantile(accepted_error, 0.95)),
                "p99": float(np.quantile(accepted_error, 0.99)),
                "maximum": float(np.max(accepted_error)),
            },
            residuals_above_target,
        )

    residual_statistics, residuals_above_target = error_statistics(
        result.relative_error
    )
    reference_residual_statistics, reference_above_rtol = error_statistics(
        result.vbml_reference_error
    )
    summary = {
        "u1": result.u1,
        "hexadecapole_seconds": hexadecapole_seconds,
        "microjax_seconds": result.cpu_seconds,
        "vbml_seconds": result.vbml_seconds,
        "vbml_residual_seconds": result.vbml_residual_seconds,
        "microjax_ms_per_point": 1.0e3 * result.cpu_seconds / args.points,
        "vbml_ms_per_point": 1.0e3 * result.vbml_seconds / args.points,
        "vbml_residual_ms_per_point": 1.0e3 * result.vbml_residual_seconds / args.points,
        "cpu_over_vbml": result.cpu_seconds / result.vbml_seconds,
        "tier_counts": count_dictionary(result.cpu_result.tier),
        "status_counts": count_dictionary(result.cpu_result.status),
        "nonzero_status": int(np.sum(~result.certified)),
        "structurally_valid_residuals_above_rtol": residuals_above_target,
        "vbml_reference_above_rtol": reference_above_rtol,
        "relative_error_reference": f"VBMicrolensing Tol=RelTol={VBML_RESIDUAL_TOL:.1e}",
        "vbml_reference_relative_error": (
            f"VBML ({VBML_TIMING_TOL:.1e}) vs VBML ({VBML_RESIDUAL_TOL:.1e})"
        ),
        "status_zero_relative_error": residual_statistics,
        "status_zero_vbml_reference_relative_error": reference_residual_statistics,
    }
    print(f"\n{result.name}")
    print(
        "computation time: %.3f sec (%.3f ms per point) "
        "for hexadecapole in microJAX"
        % (hexadecapole_seconds, 1.0e3 * hexadecapole_seconds / args.points)
    )
    print(
        "computation time: %.3f sec (%.3f ms per point), median of %d, "
        "with VBMicrolensing (Tol=RelTol=%.1e; timing reference)"
        % (
            result.vbml_seconds,
            summary["vbml_ms_per_point"],
            args.repeats,
            VBML_TIMING_TOL,
        )
    )
    print(
        "computation time: %.3f sec (%.3f ms per point), single run, "
        "with VBMicrolensing (Tol=RelTol=%.1e; residual reference)"
        % (
            result.vbml_residual_seconds,
            summary["vbml_residual_ms_per_point"],
            VBML_RESIDUAL_TOL,
        )
    )
    print(
        "computation time: %.3f sec (%.3f ms per point), median of %d, "
        "with microJAX one-shot CPU ICRS"
        % (result.cpu_seconds, summary["microjax_ms_per_point"], args.repeats)
    )
    print(
        "CPU / VBML steady-state ratio (Tol=RelTol=%.1e): %.2fx"
        % (VBML_TIMING_TOL, summary["cpu_over_vbml"])
    )
    print("  tiers:", summary["tier_counts"])
    print("  statuses:", summary["status_counts"])
    print(
        "  status-zero relative error (VBML Tol=RelTol=%.1e): "
        "median=%.3e, p95=%.3e, p99=%.3e, max=%.3e"
        % (
            VBML_RESIDUAL_TOL,
            *(
                summary["status_zero_relative_error"][key]
                for key in ("median", "p95", "p99", "maximum")
            ),
        )
    )
    print(
        "  structurally-valid residuals above rtol:",
        residuals_above_target,
    )
    print(
        "  VBML-reference relative residual (%.1e vs %.1e): "
        "median=%.3e, p95=%.3e, p99=%.3e, max=%.3e"
        % (
            VBML_TIMING_TOL,
            VBML_RESIDUAL_TOL,
            *(
                summary["status_zero_vbml_reference_relative_error"][key]
                for key in ("median", "p95", "p99", "maximum")
            ),
        )
    )
    print("  VBML-reference residuals above rtol:", reference_above_rtol)
    return summary


def add_geometry_inset(
    axis,
    points: jax.Array,
    *,
    s: float,
    q: float,
    rho: float,
) -> None:
    _, caustics = critical_and_caustic_curves(nlenses=2, npts=160, s=s, q=q)
    inset = inset_axes(axis, width="35%", height="58%", loc="upper right", borderpad=1.0)
    for caustic in caustics:
        inset.plot(caustic.real, caustic.imag, color="tab:red", lw=0.8)
    inset.plot(points.real, points.imag, color="0.4", lw=0.7)
    indices = np.linspace(0, points.size - 1, 13, dtype=int)
    circles = [
        plt.Circle(
            (float(points.real[index]), float(points.imag[index])),
            radius=rho,
            fill=False,
            ec="tab:blue",
            lw=0.55,
        )
        for index in indices
    ]
    inset.add_collection(
        mpl.collections.PatchCollection(circles, match_original=True, alpha=0.55)
    )
    inset.plot(-q / (1.0 + q) * s, 0.0, ".", color="black", ms=3)
    inset.plot(1.0 / (1.0 + q) * s, 0.0, ".", color="black", ms=3)
    geometry = np.concatenate(
        [np.asarray(points)] + [np.asarray(caustic).reshape(-1) for caustic in caustics]
    )
    x_min = float(np.min(geometry.real) - rho)
    x_max = float(np.max(geometry.real) + rho)
    y_min = float(np.min(geometry.imag) - rho)
    y_max = float(np.max(geometry.imag) + rho)
    x_padding = 0.06 * max(x_max - x_min, rho)
    y_padding = 0.06 * max(y_max - y_min, rho)
    inset.set_xlim(x_min - x_padding, x_max + x_padding)
    inset.set_ylim(y_min - y_padding, y_max + y_padding)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xlabel(r"$\mathrm{Re}(w)$", fontsize=11)
    inset.set_ylabel(r"$\mathrm{Im}(w)$", fontsize=11)
    inset.tick_params(labelsize=10)
    inset.grid(ls=":", lw=0.5)


def plot_residual(
    axis,
    time_days,
    values: np.ndarray,
    certified: np.ndarray,
    *,
    rtol: float,
    color: str,
    linestyle: str,
    label: str,
    mark_status: bool = False,
):
    floor = np.finfo(float).tiny
    residual = np.maximum(values, floor)
    axis.plot(time_days, residual, color=color, lw=1.15, ls=linestyle, label=label)
    if mark_status:
        axis.plot(
            np.asarray(time_days)[~certified],
            residual[~certified],
            "x",
            ms=4.5,
            color="tab:red",
            label="non-zero status",
        )
    axis.axhline(rtol, color="0.35", ls="--", lw=1.0)
    axis.set_yscale("log")
    axis.set_xlabel("time (days)", fontsize=15)
    axis.tick_params(labelsize=13)
    axis.grid(ls=":", lw=0.6)
    axis.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))


def plot_comparison(
    time_days: jax.Array,
    points: jax.Array,
    uniform: ProfileResult,
    limb_dark: ProfileResult,
    args: argparse.Namespace,
) -> Path:
    figure = plt.figure(figsize=(10.5, 8))
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(3.7, 1.0, 1.0),
        hspace=0.07,
    )
    light_curve = figure.add_subplot(grid[0, 0])
    uniform_residual = figure.add_subplot(grid[1, 0], sharex=light_curve)
    limb_residual = figure.add_subplot(
        grid[2, 0], sharex=light_curve, sharey=uniform_residual
    )

    profile_styles = (
        (uniform, "tab:blue"),
        (limb_dark, "tab:orange"),
    )
    for result, color in profile_styles:
        light_curve.plot(
            time_days,
            result.vbml,
            color=color,
            lw=1.35,
            label=f"{result.name} — VBML (Tol={VBML_RESIDUAL_TOL:.0e})",
        )
        light_curve.plot(
            time_days,
            result.cpu,
            ".",
            color=color,
            ms=2.1,
            alpha=0.75,
            label=f"{result.name} — microJAX",
        )
    light_curve.set_title(
        "Binary finite-source magnification  "
        rf"($\rho={args.rho:.3g},\ s={args.s:.2f},\ "
        rf"q={args.q:.3g},\ u_1={args.u1:.2f}$)",
        fontsize=17,
        pad=10,
    )
    light_curve.set_ylabel("magnification", fontsize=16)
    light_curve.grid(ls=":", lw=0.7)
    light_curve.legend(
        loc="upper left",
        ncol=1,
        fontsize=12,
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=2.2,
    )
    light_curve.tick_params(labelbottom=False, labelsize=13)
    light_curve.set_yscale("log")
    add_geometry_inset(light_curve, points, s=args.s, q=args.q, rho=args.rho)

    for axis, result, color in (
        (uniform_residual, uniform, "tab:blue"),
        (limb_residual, limb_dark, "tab:orange"),
    ):
        plot_residual(
            axis,
            time_days,
            result.relative_error,
            result.certified,
            rtol=args.rtol,
            color=color,
            linestyle="-",
            label=f"microJAX - VBML (Tol={VBML_RESIDUAL_TOL:.0e})",
            mark_status=True,
        )
        plot_residual(
            axis,
            time_days,
            result.vbml_reference_error,
            result.certified,
            rtol=args.rtol,
            color=color,
            linestyle=":",
            label=(
                f"VBML (Tol={VBML_TIMING_TOL:.0e}) - "
                f"VBML (Tol={VBML_RESIDUAL_TOL:.0e})"
            ),
        )
    uniform_residual.set_ylabel("Uniform\nrelative res.", fontsize=14)
    uniform_residual.set_xlabel("")
    uniform_residual.tick_params(labelbottom=False)
    limb_residual.set_ylabel("LD\nrelative res.", fontsize=14)
    maximum = max(
        np.max(uniform.relative_error),
        np.max(uniform.vbml_reference_error),
        np.max(limb_dark.relative_error),
        np.max(limb_dark.vbml_reference_error),
    )
    upper = max(10.0 * args.rtol, 1.5 * maximum)
    lower = 1.0e-6
    uniform_residual.set_ylim(lower, upper)
    uniform_residual.legend(loc="upper right", ncol=3, fontsize=9)
    limb_residual.legend(loc="upper right", ncol=3, fontsize=9)

    output = OUTPUT_DIRECTORY / "compare_binary_uniform_limb_dark.png"
    figure.subplots_adjust(left=0.145, right=0.975, bottom=0.075, top=0.935)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def write_maximum_residual_diagnostic(
    result: ProfileResult,
    points: jax.Array,
    time_days: jax.Array,
    args: argparse.Namespace,
) -> Path:
    icrs = result.certified & np.isin(
        np.asarray(result.cpu_result.tier), (5, 6, 7, 8, 9)
    )
    candidates = np.flatnonzero(icrs if np.any(icrs) else result.certified)
    index = int(candidates[np.argmax(result.relative_error[candidates])])
    suffix = "uniform" if result.u1 == 0.0 else "limb_dark"
    output = OUTPUT_DIRECTORY / f"compare_binary_profiles_{suffix}_max_residual_icrs.png"
    plot_boundary_construction(
        points[index],
        args.rho,
        s=args.s,
        q=args.q,
        time_value=float(time_days[index]),
        relative_residual=float(result.relative_error[index]),
        limb_darkening=result.u1,
        selected_tier=int(result.cpu_result.tier[index]),
        selected_n_slices=int(result.cpu_result.n_radial_nodes[index]),
        n_limb=int(result.cpu_result.n_limb[index]),
        output_path=output,
    )
    return output


def main() -> None:
    args = parse_args()
    if args.points <= 1 or args.n_limb <= 0 or args.repeats <= 0:
        raise ValueError("points, n-limb, and repeats must be positive")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    time_days, points = trajectory(args)
    print("number of data points:", args.points)
    print(
        "configuration: rho=%.3e, rtol=%.1e, s=%.3f, q=%.3g, u1(LD)=%.2f"
        % (args.rho, args.rtol, args.s, args.q, args.u1)
    )
    point_seconds, hexadecapole_seconds = benchmark_approximations(points, args)
    print(
        "computation time: %.3f sec (%.3f ms per point) "
        "for point-source in microJAX"
        % (point_seconds, 1.0e3 * point_seconds / args.points)
    )
    uniform = evaluate_profile("Uniform", 0.0, points, args)
    limb_dark = evaluate_profile("Limb darkening", args.u1, points, args)
    summaries = {
        "uniform": profile_summary(
            uniform,
            args,
            hexadecapole_seconds=hexadecapole_seconds["uniform"],
        ),
        "limb_dark": profile_summary(
            limb_dark,
            args,
            hexadecapole_seconds=hexadecapole_seconds["limb_dark"],
        ),
    }
    payload = {
        "configuration": vars(args),
        "reference": "VBMicrolensing",
        "vbml_timing_tolerance": VBML_TIMING_TOL,
        "vbml_residual_tolerance": VBML_RESIDUAL_TOL,
        "vbmicrolensing_version": getattr(VBMicrolensing, "__version__", "unknown"),
        "point_source_seconds": point_seconds,
        "profiles": summaries,
    }
    json_output = OUTPUT_DIRECTORY / "benchmark_binary_profiles.json"
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    figure_output = plot_comparison(time_days, points, uniform, limb_dark, args)
    diagnostic_outputs = [
        write_maximum_residual_diagnostic(result, points, time_days, args)
        for result in (uniform, limb_dark)
    ]
    print("\noutput:", figure_output)
    print("benchmark:", json_output)
    for output in diagnostic_outputs:
        print("diagnostic:", output)
    residuals_above_target = sum(
        summary["structurally_valid_residuals_above_rtol"]
        for summary in summaries.values()
    )
    if residuals_above_target:
        raise RuntimeError(
            "combined comparison found "
            f"{residuals_above_target} structurally-valid residuals above rtol"
        )


if __name__ == "__main__":
    main()
