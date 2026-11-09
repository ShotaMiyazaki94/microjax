"""LD benchmark estimates with stratified configuration-bootstrap uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np

from cpu_benchmark_stats import estimate

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, EMPTY = "#222222", "#777777", "#dddddd", "#eeeeee"
BLUE, ORANGE = "#31688e", "#b45f2a"
TARGET = 1e-3
MIN_CONFIGS = 15
BOOTSTRAPS = 4000
SEED = 20261106


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#888888",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "savefig.facecolor": "white",
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


def ratio_label(value, _=None):
    return f"1/{round(1 / value):d}×" if value < 1 else f"{value:g}×"


def log_axis(ax, axis):
    getattr(ax, f"set_{axis}scale")("log")
    getattr(ax, f"{axis}axis").set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", alpha=0.8)


def save(fig, destination):
    fig.savefig(destination.with_suffix(".png"), dpi=190, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
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


def overview(points, groups, edges, summary, directory):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), layout="constrained")
    error = np.sort(np.maximum(points["relative_error"], 1e-13))
    error_limits = (max(1e-13, error[0] / 2), max(1e-3, error[-1] * 1.5))

    ax = axes[0, 0]
    ax.set_title("a   Relative error", loc="left")
    ax.step(error, np.arange(1, len(error) + 1) / len(error), where="post", lw=1.7, color=BLUE)
    ax.axvline(1e-4, color=MUTED, ls="--", lw=0.9)
    interval(ax, summary["accuracy_p95"], 0.95, BLUE, horizontal=True)
    log_axis(ax, "x")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.set(xlabel=r"$|A/A_{\rm ref}-1|$", ylabel="Cumulative fraction", xlim=error_limits, ylim=(0, 1.04))

    ax = axes[0, 1]
    ax.set_title("b   By source radius", loc="left")
    for i in range(len(edges["rho"]) - 1):
        selected = in_bin(points["rho"], edges["rho"], i)
        errors = points["relative_error"][selected]
        positive = errors[errors > 0]
        center = np.sqrt(edges["rho"][i:i+2].prod())
        if positive.size:
            violin = ax.violinplot(
                np.log10(positive), positions=[np.log10(center)],
                widths=0.72 * np.diff(np.log10(edges["rho"][i:i+2])),
                showextrema=False, points=160, bw_method="scott",
            )
            body = violin["bodies"][0]
            for path in body.get_paths():
                path.vertices[:] = 10**path.vertices
            body.set_facecolor(BLUE)
            body.set_edgecolor(BLUE)
            body.set_alpha(0.3)
            ax.scatter(center, np.quantile(errors, 0.95), s=21, color=ORANGE, zorder=3)
    ax.axhline(1e-4, color=MUTED, ls="--", lw=0.9)
    log_axis(ax, "x")
    log_axis(ax, "y")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.set(xlabel=r"Source radius $\rho$", ylabel=r"$|A/A_{\rm ref}-1|$",
           xlim=(edges["rho"][0], edges["rho"][-1]), ylim=error_limits)

    ax = axes[1, 0]
    ax.set_title("c   Time per source", loc="left")
    for y, key, color in ((1, "cpu_runtime", BLUE), (0, "vbm_runtime", ORANGE)):
        interval(ax, summary[key], y, color, horizontal=True, scale=1000)
        value = summary[key]["estimate"] * 1000
        ax.annotate(f"{value:.2f}", (value, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color=color)
    ax.grid(True, axis="x")
    upper_time = max(summary[key]["ci95"][1] for key in ("cpu_runtime", "vbm_runtime"))
    ax.set(yticks=[1, 0], yticklabels=["microJAX", "VBM"], ylim=(-0.6, 1.6),
           xlim=(0, upper_time * 1150), xlabel="Time [ms]")

    ax = axes[1, 1]
    ax.set_title("d   Relative speed", loc="left")
    ratios = np.sort(groups["ratio"])
    ax.step(ratios, np.arange(1, len(ratios) + 1) / len(ratios), where="post", lw=1.7, color=ORANGE)
    ax.axvline(1, color=MUTED, ls="--", lw=0.9)
    interval(ax, summary["speed"], 0.5, ORANGE, horizontal=True)
    log_axis(ax, "x")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.set(xlabel=r"$t_{\rm VBM}/t_{\rm microJAX}$", ylabel="Cumulative fraction", ylim=(0, 1.04))
    save(fig, directory / "cpu_overview")


def parameter_maps(edges, cells, directory):
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.3), sharex="col", layout="constrained")
    speed_bounds = np.array([1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16])
    speed_cmap = ListedColormap(["#235b7c", "#5791ad", "#96becd", "#d5e4e9",
                               "#f5e4cc", "#e8bd8c", "#cc8857", "#9a532d"])
    names = {"q": r"Mass ratio $q$", "rho": r"Source radius $\rho$", "reference": r"$A_{\rm ref}$"}
    combinations = (("q", "rho"), ("reference", "rho"), ("reference", "q"))
    for row, metric in enumerate(("speed", "accuracy")):
        cmap = speed_cmap if row == 0 else plt.get_cmap("viridis_r")
        norm = BoundaryNorm(speed_bounds, cmap.N) if row == 0 else LogNorm(1e-5, 1e-3)
        for col, (x, y) in enumerate(combinations):
            ax = axes[row, col]
            matrix = np.full((len(edges[y])-1, len(edges[x])-1), np.nan)
            for cell in cells:
                if cell["metric"] == metric and cell["x"] == x and cell["y"] == y and cell["eligible"]:
                    matrix[cell["iy"], cell["ix"]] = cell["estimate"]
            ax.set_facecolor(EMPTY)
            mesh = ax.pcolormesh(edges[x], edges[y], np.ma.masked_invalid(matrix),
                                 cmap=cmap, norm=norm, shading="flat", edgecolors="white", linewidth=0.45)
            ax.set(xscale="log", yscale="log", ylabel=names[y])
            if row == 1:
                ax.set_xlabel(names[x])
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title(chr(97 + row*3 + col), loc="left")
        colorbar = fig.colorbar(mesh, ax=axes[row, :], pad=0.025, fraction=0.025, aspect=22)
        if row == 0:
            colorbar.set_ticks([1/16, 1/4, 1, 4, 16])
            colorbar.ax.yaxis.set_major_formatter(FuncFormatter(ratio_label))
            colorbar.set_label(r"Median $t_{\rm VBM}/t_{\rm microJAX}$")
        else:
            colorbar.set_ticks([1e-5, 1e-4, 1e-3])
            colorbar.set_label("95th-percentile relative error")
    save(fig, directory / "cpu_parameter_maps_ld")


def regional_intervals(cells, directory):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 8.8), sharey=True, layout="constrained")
    labels = [rf"$q_{{{i+1}}},\rho_{{{j+1}}}$" for j in range(5) for i in range(5)]
    for ax, metric, color in zip(axes, ("speed", "accuracy"), (ORANGE, BLUE)):
        for j in range(0, 5, 2):
            ax.axhspan(j*5-0.5, j*5+4.5, color="#f2f2f2", zorder=0)
        for cell in cells:
            if cell["metric"] == metric and cell["x"] == "q" and cell["y"] == "rho":
                interval(ax, cell, cell["iy"]*5+cell["ix"], color, horizontal=True)
        ax.set(ylim=(24.7, -0.7), yticks=range(25), yticklabels=labels)
        log_axis(ax, "x")
        ax.set_title("a   Relative speed" if metric == "speed" else "b   Relative error", loc="left")
        if metric == "speed":
            ax.axvline(1, color=MUTED, lw=0.9, ls="--")
            ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2]))
            ax.xaxis.set_major_formatter(FuncFormatter(ratio_label))
            ax.set_xlabel(r"Median $t_{\rm VBM}/t_{\rm microJAX}$")
        else:
            lo, hi = ax.get_xlim()
            ax.set_xlim(min(lo, 3.8e-5), max(hi, 1.03e-4))
            ax.axvline(1e-4, color=MUTED, lw=0.9, ls="--")
            ax.xaxis.set_major_locator(FixedLocator([4e-5, 6e-5, 1e-4]))
            ax.set_xlabel(r"95th percentile of $|A/A_{\rm ref}-1|$")
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
    overview(points, groups, edges, summary, args.output_dir)
    parameter_maps(edges, cells, args.output_dir)
    regional_intervals(cells, args.output_dir)
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
