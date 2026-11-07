"""LD benchmark estimates with stratified configuration-bootstrap uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np

from cpu_benchmark_stats import estimate

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, EMPTY = "#172536", "#596879", "#E2E8EE", "#E4E8ED"
BLUE, ORANGE, RED = "#176B9A", "#BC6024", "#B52F46"
TARGET = 1e-3
MIN_CONFIGS = 15
BOOTSTRAPS = 4000
SEED = 20261106


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "#F8FAFC",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C8D1DC",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "savefig.facecolor": "#F8FAFC",
            "svg.fonttype": "none",
        }
    )


def load_data(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["state"] != "complete":
        raise ValueError("Benchmark is incomplete; do not present a partial run as full coverage")
    with (directory / "cpu.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata["profiles"]["ld"]["selected_points"]:
        raise ValueError("Incomplete LD accuracy rows")
    fields = ("q", "s", "rho", "reference", "magnification", "relative_error")
    points = {key: np.asarray([float(row[key]) for row in rows]) for key in fields}
    for key in ("config_id", "stratum_id"):
        points[key] = np.asarray([int(row[key]) for row in rows])
    if not np.all(np.isfinite(points["magnification"]) & (points["magnification"] > 0)):
        raise ValueError("Report nonfinite/nonpositive CPU estimates before plotting")
    reference_valid = np.isfinite(points["reference"]) & (points["reference"] > 0)
    points = {key: values[reference_valid] for key, values in points.items()}
    with (directory / "timing.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        references = np.asarray(json.loads(row["reference_magnifications"]))
        cpu_values = np.asarray(json.loads(row["cpu_magnifications"]))
        if row["speed_eligible"] != "True" or not np.all(np.abs(cpu_values / references - 1) <= TARGET):
            continue
        record = {key: float(row[key]) for key in ("q", "s", "rho", "cpu_seconds_per_point", "vbm_seconds_per_point")}
        record.update({key: int(row[key]) for key in ("config_id", "stratum_id")})
        record["reference"] = float(references[0])
        record["ratio"] = record["vbm_seconds_per_point"] / record["cpu_seconds_per_point"]
        records.append(record)
    groups = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    if len(np.unique(groups["config_id"])) != len(groups["config_id"]):
        raise ValueError("The bounded benchmark must have one measured source per configuration")
    metadata["plotted"] = {
        "accuracy_positions": len(points["q"]),
        "missing_references": int(np.sum(~reference_valid)),
        "matched_timings": len(records),
        "excluded_timings": len(rows) - len(records),
    }
    return points, groups, metadata


def stat(data, key, selected=None, *, quantile=0.5, log=False):
    if selected is None:
        selected = np.ones(len(data[key]), dtype=bool)
    return estimate(
        data[key][selected],
        data["config_id"][selected],
        data["stratum_id"][selected],
        quantile=quantile,
        log=log,
        min_configs=MIN_CONFIGS,
        min_points=50 if key == "relative_error" else 0,
        repeats=BOOTSTRAPS,
        seed=SEED,
        frame_config_ids=data["config_id"],
        frame_strata=data["stratum_id"],
    )


def in_bin(values, edges, index):
    return (values >= edges[index]) & (
        (values <= edges[index + 1]) if index == len(edges) - 2 else (values < edges[index + 1])
    )


def statistics(points, groups, metadata):
    edges = {key: np.asarray(metadata["sampling"][key + "_edges"]) for key in ("q", "rho")}
    edges["reference"] = np.asarray(metadata["sampling"]["magnification_edges"])
    combinations = (("q", "rho"), ("reference", "rho"), ("reference", "q"))
    cells = []
    for metric, data, key, quantile, log in (
        ("speed", groups, "ratio", 0.5, True),
        ("accuracy", points, "relative_error", 0.95, False),
    ):
        for x, y in combinations:
            for iy in range(len(edges[y]) - 1):
                for ix in range(len(edges[x]) - 1):
                    selected = in_bin(data[x], edges[x], ix) & in_bin(data[y], edges[y], iy)
                    population = [
                        row
                        for row in metadata["sampling"]["coverage"]
                        if (
                            row["q_bin"] == ix and row["rho_bin"] == iy
                            if x == "q"
                            else row["mag_bin"] == ix and row[y + "_bin"] == iy
                        )
                    ]
                    frame_configs = (
                        next(
                            row["population_configs"]
                            for row in metadata["sampling"]["strata"]
                            if row["stratum_id"] == iy * 5 + ix
                        )
                        if x == "q"
                        else sum(row["configurations"] for row in population)
                    )
                    cells.append(
                        {
                            "metric": metric,
                            "x": x,
                            "y": y,
                            "ix": ix,
                            "iy": iy,
                            "x_low": float(edges[x][ix]),
                            "x_high": float(edges[x][ix + 1]),
                            "y_low": float(edges[y][iy]),
                            "y_high": float(edges[y][iy + 1]),
                            "n_frame_configurations": frame_configs,
                            "n_frame_positions": sum(row["source_points"] for row in population),
                            **stat(data, key, selected, quantile=quantile, log=log),
                        }
                    )
    return edges, cells


def percent(value, _=None):
    return f"{100 * value:g}%"


def ratio_label(value, _=None):
    return f"1/{round(1 / value):d}×" if value < 1 else f"{value:g}×"


def log_axis(ax, axis):
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"{axis}axis").set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", alpha=0.8)


def save(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=190)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def interval(ax, result, coordinate, color, *, horizontal=False, scale=1):
    if not result["eligible"]:
        return
    center = result["estimate"] * scale
    lo95, hi95 = np.asarray(result["ci95"]) * scale
    lo68, hi68 = np.asarray(result["ci68"]) * scale
    if horizontal:
        ax.hlines(coordinate, lo95, hi95, color=color, alpha=0.38, lw=1.8)
        ax.hlines(coordinate, lo68, hi68, color=color, lw=5)
        ax.scatter(center, coordinate, s=28, color=color, edgecolor="white", linewidth=0.65, zorder=3)
    else:
        ax.vlines(coordinate, lo95, hi95, color=color, alpha=0.38, lw=1.8)
        ax.vlines(coordinate, lo68, hi68, color=color, lw=5)
        ax.scatter(coordinate, center, s=28, color=color, edgecolor="white", linewidth=0.65, zorder=3)


def overview(points, groups, metadata, edges, summary, directory):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.79, bottom=0.16, hspace=0.45, wspace=0.29)
    fig.text(0.085, 0.95, "microJAX CPU · limb darkening", fontsize=24, weight="bold")
    fig.text(0.085, 0.91, "Accuracy, speed and uncertainty of the estimates", fontsize=14, color=MUTED)
    fig.text(
        0.085,
        0.866,
        f"a = 0.5   ·   {len(points['q']):,} positions   ·   "
        f"{metadata['sampling']['selected_configs']:,} grid configurations   ·   "
        f"{len(groups['q'])} matched timings",
        fontsize=11,
    )
    fig.text(
        0.085,
        0.833,
        "Full parameter-grid coverage · one source / configuration · ● estimate  ━ 68.27%  ─ 95% bootstrap interval",
        fontsize=10,
        color=MUTED,
    )

    ax = axes[0, 0]
    ax.set_title("A   Accuracy distribution", loc="left", pad=14)
    error = np.sort(np.maximum(points["relative_error"], 1e-13))
    ax.step(error, np.arange(1, len(error) + 1) / len(error), where="post", lw=2.3, color=BLUE)
    ax.axvline(TARGET, color=RED, ls="--", lw=1.2)
    ax.annotate("0.1% target", (TARGET, 0.55), xytext=(6, 0), textcoords="offset points", rotation=90, color=RED)
    log_axis(ax, "x")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.set(
        xlabel="Relative difference from saved LD reference",
        ylabel="Fraction of reference positions",
        xlim=(max(1e-13, error[0] / 2), max(3e-3, error[-1] * 2)),
        ylim=(0, 1.035),
    )
    ax.text(
        0.03, 0.08, f"{int(np.sum(error > TARGET))} / {len(error):,} above target", transform=ax.transAxes, color=MUTED
    )

    ax = axes[0, 1]
    ax.set_title("B   Error distribution by source size", loc="left", pad=14)
    zero_errors = 0
    for i in range(5):
        selected = in_bin(points["rho"], edges["rho"], i)
        errors = points["relative_error"][selected]
        positive = errors[errors > 0]
        zero_errors += int(np.sum(errors == 0))
        center = np.sqrt(edges["rho"][i : i + 2].prod())
        if positive.size:
            # Estimate density per log-error unit. Transform both coordinates
            # back so violin widths are equal on the logarithmic rho axis.
            violin = ax.violinplot(
                np.log10(positive),
                positions=[np.log10(center)],
                widths=0.72 * np.diff(np.log10(edges["rho"][i : i + 2])),
                showextrema=False,
                points=160,
                bw_method="scott",
            )
            body = violin["bodies"][0]
            for path in body.get_paths():
                path.vertices[:] = 10**path.vertices
            body.set_facecolor(BLUE)
            body.set_edgecolor(BLUE)
            body.set_alpha(0.28)
        result = stat(points, "relative_error", selected, quantile=0.95)
        interval(ax, result, center, ORANGE)
        ax.text(
            center,
            0.025,
            f"{result['n_configurations']} cfg\n{result['n_points']} pts",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )
    ax.axhline(TARGET, color=RED, ls="--", lw=1.2)
    ax.text(
        0.02,
        0.98,
        "Violin: log-error density · orange: p95 + CI",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    if zero_errors:
        ax.text(
            0.98,
            0.12,
            f"{zero_errors} exact zeros omitted from density",
            transform=ax.transAxes,
            ha="right",
            fontsize=7.5,
            color=MUTED,
        )
    log_axis(ax, "x")
    log_axis(ax, "y")
    ax.yaxis.set_major_formatter(FuncFormatter(percent))
    ax.set(
        xlabel=r"Source radius $\rho$",
        ylabel="Relative difference from reference",
        xlim=(edges["rho"][0], edges["rho"][-1]),
        ylim=(max(1e-13, error[0] / 2), max(3e-3, error[-1] * 2)),
    )

    ax = axes[1, 0]
    ax.set_title("C   Median warm runtime", loc="left", pad=14)
    for y, key, color in ((1, "cpu_runtime", BLUE), (0, "vbm_runtime", ORANGE)):
        interval(ax, summary[key], y, color, horizontal=True, scale=1000)
        ax.annotate(
            f"{summary[key]['estimate'] * 1000:.3f} ms",
            (summary[key]["estimate"] * 1000, y),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            color=color,
        )
    ax.grid(True, axis="x")
    ax.set(
        yticks=[1, 0],
        yticklabels=["microJAX\nfull solve", "VBM\nBinaryMag2"],
        ylim=(-0.6, 1.7),
        xlabel="Warm milliseconds / position · estimate and CI",
    )

    ax = axes[1, 1]
    ax.set_title("D   Paired speed ratios", loc="left", pad=14)
    ratios = np.sort(groups["ratio"])
    ax.step(ratios, np.arange(1, len(ratios) + 1) / len(ratios), where="post", lw=2.2, color=ORANGE)
    ax.axvline(1, color=INK, ls="--", lw=1.2)
    interval(ax, summary["speed"], 0.5, ORANGE, horizontal=True)
    value = summary["speed"]
    ax.text(
        0.03,
        0.94,
        f"Median {value['estimate']:.2f}×\n95% CI [{value['ci95'][0]:.2f}, {value['ci95'][1]:.2f}]",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    log_axis(ax, "x")
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=8))
    ax.xaxis.set_major_formatter(FuncFormatter(ratio_label))
    ax.set(xlabel=r"$t_{\rm VBM}\,/\,t_{\rm microJAX}$", ylabel="Fraction of matched configurations", ylim=(0, 1.035))
    ax.text(0.02, 0.035, "← VBM faster", transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.text(0.98, 0.035, "microJAX faster →", transform=ax.transAxes, ha="right", color=MUTED, fontsize=9)
    footnotes(fig, 0.085, metadata)
    save(fig, directory / "cpu_overview")


def footnotes(fig, left, metadata):
    repeats = metadata["timing"]["repeats"]
    lines = [
        f"{BOOTSTRAPS:,} bootstrap draws within q–ρ strata; whole configurations stay together. CI describes estimator uncertainty, not point scatter.",
        "Source selection balances occupied magnification bins; intervals describe configuration-resampling stability, conditional on this design.",
        f"Full CPU solve · identical single-source inputs · {repeats} warm repeats · VBM Tol = RelTol = 10⁻⁴ · compilation / IPC excluded from timings.",
    ]
    for y, line in zip((0.073, 0.048, 0.023), lines):
        fig.text(left, y, line, fontsize=8.8, color=MUTED)


def parameter_maps(points, groups, metadata, edges, cells, directory):
    fig = plt.figure(figsize=(20.0, 12.0))
    fig.text(0.065, 0.963, "Regional estimates and their uncertainty · a = 0.5", fontsize=22, weight="bold")
    fig.text(
        0.065,
        0.927,
        "Each cell: estimate  /  [68.27% bootstrap interval]  /  measured / available configurations",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.897,
        f"{len(points['q']):,} LD reference positions · {len(groups['q'])} matched timing groups · "
        f"gray = fewer than {MIN_CONFIGS} configurations (or 50 accuracy positions)",
        fontsize=10,
    )
    speed_bounds = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16])
    speed_cmap = ListedColormap(
        ["#205980", "#61A2BB", "#9AC7D2", "#D2E6EB", "#F6E3C7", "#EFBE83", "#D58B4B", "#AC592C"]
    )
    error_bounds = np.array([0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    error_cmap = ListedColormap(
        ["#176943", "#39935D", "#6EAE70", "#A8CE86", "#DAE8A5", "#F4CC7B", "#E68059", "#AC3346"]
    )
    names = {"q": r"Mass ratio $q$", "rho": r"Source radius $\rho$", "reference": "Reference magnification"}
    combinations = (("q", "rho"), ("reference", "rho"), ("reference", "q"))
    meshes = []
    for row, metric in enumerate(("speed", "accuracy")):
        cmap, bounds = (speed_cmap, speed_bounds) if row == 0 else (error_cmap, error_bounds)
        cmap.set_bad(EMPTY)
        norm = BoundaryNorm(bounds, cmap.N, clip=True)
        for col, (x, y) in enumerate(combinations):
            ax = fig.add_axes([0.065 + col * 0.322, 0.594 if row == 0 else 0.192, 0.258, 0.25])
            selected = [c for c in cells if c["metric"] == metric and c["x"] == x and c["y"] == y]
            matrix = np.full((len(edges[y]) - 1, len(edges[x]) - 1), np.nan)
            for cell in selected:
                if cell["eligible"]:
                    matrix[cell["iy"], cell["ix"]] = cell["estimate"]
            mesh = ax.pcolormesh(
                edges[x],
                edges[y],
                np.ma.masked_invalid(matrix),
                cmap=cmap,
                norm=norm,
                shading="flat",
                edgecolors="white",
                linewidth=0.7,
            )
            if col == 0:
                meshes.append(mesh)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(names[x])
            ax.set_ylabel(names[y])
            ax.set_title(
                f"{chr(65 + row*3 + col)}   " + ("Median speed ratio" if row == 0 else "p95 error (ppm)"),
                loc="left",
                pad=12,
            )
            for cell in selected:
                ix, iy = cell["ix"], cell["iy"]
                count_label = f"{cell['n_configurations']}/{cell['n_frame_configurations']}"
                label, color = (
                    "No source\nin frame" if cell["n_frame_configurations"] == 0 else "n=" + count_label
                ), MUTED
                if cell["eligible"]:
                    v, (lo, hi) = cell["estimate"], cell["ci68"]
                    label = (
                        f"{v:.2g}×\n[{lo:.2g}, {hi:.2g}]" if row == 0 else f"{1e6*v:.0f}\n[{1e6*lo:.0f}, {1e6*hi:.0f}]"
                    ) + f"\n{count_label}"
                    rgba = cmap(norm(v))
                    brightness = np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
                    color = "white" if brightness < 0.53 else INK
                ax.text(
                    np.sqrt(edges[x][ix : ix + 2].prod()),
                    np.sqrt(edges[y][iy : iy + 2].prod()),
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    color=color,
                    linespacing=1.4,
                )
    for index, (bounds, ticks, y) in enumerate(
        ((speed_bounds, speed_bounds, 0.508), (error_bounds, error_bounds[1:], 0.108))
    ):
        cbar = fig.colorbar(
            meshes[index],
            cax=fig.add_axes([0.065, y, 0.904, 0.014]),
            orientation="horizontal",
            spacing="uniform",
            ticks=ticks,
            extend="both" if index == 0 else "max",
        )
        cbar.ax.set_xticklabels([ratio_label(v) if index == 0 else f"{1e6*v:g}" for v in ticks])
        cbar.set_label(
            (
                "VBM / microJAX time · <1 VBM faster · >1 microJAX faster"
                if index == 0
                else "p95 relative error in ppm (10⁻⁶) · 1,000 ppm = 0.1% target"
            ),
            labelpad=6,
        )
    fig.text(
        0.065,
        0.029,
        "One measured source per configuration; its actual reference magnification determines the speed cell. All configuration combinations are covered.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.009,
        "Intervals are conditional bootstrap estimates, not worst-case bounds. Exact counts, standard errors and 95% CIs: regional_statistics.csv.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, directory / "cpu_parameter_maps_ld")


def regional_intervals(edges, cells, directory):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 12.2))
    fig.subplots_adjust(left=0.18, right=0.92, top=0.83, bottom=0.13, wspace=0.36)
    fig.text(0.055, 0.96, "How certain is each regional estimate?", fontsize=23, weight="bold")
    fig.text(0.055, 0.924, "LD · a = 0.5 · 25 mass-ratio / source-radius regions", fontsize=13, color=MUTED)
    fig.text(
        0.055,
        0.89,
        "● estimate     ━ 68.27% CI     ─ 95% CI      ·      labels: configurations / positions",
        fontsize=10,
    )
    labels = [f"q{i+1}  ·  ρ{j+1}" for j in range(5) for i in range(5)]
    for ax, metric, color in zip(axes, ("speed", "accuracy"), (ORANGE, BLUE)):
        selected = [c for c in cells if c["metric"] == metric and c["x"] == "q" and c["y"] == "rho"]
        for cell in selected:
            row = cell["iy"] * 5 + cell["ix"]
            if cell["iy"] % 2 == 0:
                ax.axhspan(row - 0.5, row + 0.5, color="#EAF0F5", alpha=0.6, zorder=0)
            interval(ax, cell, row, color, horizontal=True, scale=100 if metric == "accuracy" else 1)
            label = str(cell["n_configurations"])
            if metric == "accuracy":
                label += f" / {cell['n_points']}"
            if not cell["eligible"]:
                label += " *"
            ax.text(1.025, row, label, transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=MUTED)
        ax.set_ylim(24.7, -0.7)
        ax.set_yticks(range(25), labels if metric == "speed" else [""] * 25)
        log_axis(ax, "x")
        ax.set_title("Median speed ratio" if metric == "speed" else "p95 relative error", loc="left", pad=16)
        ax.set_xlabel("VBM / microJAX time" if metric == "speed" else "Relative error (%)")
        if metric == "speed":
            ax.axvline(1, color=INK, lw=1, ls="--")
            ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2]))
            ax.xaxis.set_major_formatter(FuncFormatter(ratio_label))
        else:
            ax.xaxis.set_major_locator(FixedLocator([0.004, 0.006, 0.008, 0.01, 0.015]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.text(
                0.98,
                1.012,
                "Target 0.1% is beyond this zoom",
                transform=ax.transAxes,
                ha="right",
                color=RED,
                fontsize=8,
            )
    fig.text(0.055, 0.083, "q edges: " + "  /  ".join(f"{v:.2g}" for v in edges["q"]), fontsize=9, color=MUTED)
    fig.text(0.055, 0.061, "ρ edges: " + "  /  ".join(f"{v:.2g}" for v in edges["rho"]), fontsize=9, color=MUTED)
    fig.text(
        0.055,
        0.035,
        f"{BOOTSTRAPS:,} stratified configuration resamples. * Interval suppressed below {MIN_CONFIGS} configurations; no interpolation.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.013,
        "Intervals show configuration-resampling stability, conditional on the grid and source-selection design; they do not bound unmeasured source positions.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, directory / "cpu_regional_intervals_ld")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/cpu/figures-benchmark/data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/cpu/figures-benchmark/figures")
    args = parser.parse_args()
    points, groups, metadata = load_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges, cells = statistics(points, groups, metadata)
    summary = {
        "coefficient": 0.5,
        "relative_error_target": TARGET,
        "bootstrap_repeats": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "minimum_configurations": MIN_CONFIGS,
        "accuracy_minimum_points": 50,
        "uncertainty": "stratified configuration-resampling stability, conditional on the grid and coverage-balanced source selection",
        "coverage": metadata["sampling"],
        "exclusions": metadata["plotted"],
        "usable_references": len(points["q"]),
        "configurations": len(np.unique(points["config_id"])),
        "matched_timing_configurations": len(groups["q"]),
        "maximum_relative_error": float(points["relative_error"].max()),
        "above_target": int(np.sum(points["relative_error"] > TARGET)),
        "speed": stat(groups, "ratio", log=True),
        "accuracy_p95": stat(points, "relative_error", quantile=0.95),
        "cpu_runtime": stat(groups, "cpu_seconds_per_point"),
        "vbm_runtime": stat(groups, "vbm_seconds_per_point"),
        "edges": {key: value.tolist() for key, value in edges.items()},
    }
    style()
    overview(points, groups, metadata, edges, summary, args.output_dir)
    parameter_maps(points, groups, metadata, edges, cells, args.output_dir)
    regional_intervals(edges, cells, args.output_dir)
    (args.output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "regional_statistics.json").write_text(json.dumps(cells, indent=2, allow_nan=False) + "\n")
    flat = []
    for cell in cells:
        row = {key: value for key, value in cell.items() if key not in ("ci68", "ci95")}
        for level in ("ci68", "ci95"):
            pair = cell[level] or (None, None)
            row[level + "_low"], row[level + "_high"] = pair
        flat.append(row)
    with (args.output_dir / "regional_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(args.output_dir)


if __name__ == "__main__":
    main()
