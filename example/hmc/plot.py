"""Plot saved injection-recovery results without JAX or another sampling run."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import corner


LABELS = (
    r"$t_0$ [day]",
    r"$t_E$ [day]",
    r"$u_0$",
    r"$10^3 q$",
    r"$s$",
    r"$\alpha$ [deg]",
    r"$\log_{10}\rho$",
    r"$F_s$",
    r"$F_b$",
)
DISPLAY_SCALE = np.array([1, 1, 1, 1e3, 1, 180 / np.pi, 1, 1, 1])
BLUE, ORANGE = "#31688e", "#b45f2a"
DIVERGENCE = "#9b3d61"
OUTPUT = Path(__file__).resolve().parents[2] / "results" / "hmc" / "binary"


def plot_results():
    output = OUTPUT
    config = json.loads((output / "run.json").read_text())
    names = config["parameter_names"]
    with np.load(output / "data.npz") as saved:
        times, flux, error, injected_data = (saved[key] for key in ("times", "flux", "error", "injected_flux"))
    with np.load(output / "samples.npz") as saved:
        samples = {name: saved[name] for name in names}
    with np.load(output / "curves.npz") as saved:
        model_times, injected = saved["times"], saved["injected_flux"]
        lower, median, upper = np.quantile(saved["model_flux"], [0.16, 0.5, 0.84], axis=0)
        flux_draws = saved["flux_draws"]
    residual = (flux - injected_data) / error
    model_error = config["relative_noise"] * injected
    residual_lower, residual_median, residual_upper = (
        (values - injected) / model_error for values in (lower, median, upper)
    )
    with np.load(output / "sampler.npz") as saved:
        divergent = saved["diverging"].ravel()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 4.6),
        sharex="col",
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1], "width_ratios": [1.3, 1]},
    )
    for column, (left, right) in enumerate(((times[0], times[-1]), (-5.0, 5.0))):
        visible = (times >= left) & (times <= right)
        model_visible = (model_times >= left) & (model_times <= right)
        x = times[visible]
        model_x = model_times[model_visible]
        ax = axes[0, column]
        ax.errorbar(x, flux[visible], yerr=error[visible], fmt=".", ms=3, color="0.5", elinewidth=0.6, label="Data")
        ax.fill_between(model_x, lower[model_visible], upper[model_visible], color=BLUE, alpha=0.25)
        ax.plot(model_x, median[model_visible], color=BLUE, lw=1.3, label="Posterior")
        ax.plot(model_x, injected[model_visible], color=ORANGE, lw=1, ls="--", label="Injected")
        ax.set_xlim(left, right)
        axes[1, column].axhline(0, color=ORANGE, lw=0.8, ls="--")
        axes[1, column].fill_between(
            model_x, residual_lower[model_visible], residual_upper[model_visible], color=BLUE, alpha=0.25
        )
        axes[1, column].plot(model_x, residual_median[model_visible], color=BLUE, lw=1.3)
        axes[1, column].plot(x, residual[visible], ".", ms=3, color="0.5")
        axes[1, column].set(xlabel="Time [day]")
    axes[0, 0].set_ylabel("Flux")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].set_title("Peak", fontsize=10)
    axes[1, 0].set_ylabel(r"$(F-F_{\rm inj})/\sigma$")
    fig.savefig(output / "lightcurve.png", dpi=190)
    plt.close(fig)

    event_draws = np.column_stack([samples[name].ravel() for name in names])
    draws = np.column_stack((event_draws, flux_draws)) * DISPLAY_SCALE
    truth = np.concatenate((config["truth"], config["flux_truth"])) * DISPLAY_SCALE
    draws[:, 6] = np.log10(draws[:, 6])
    truth[6] = np.log10(truth[6])
    low = np.minimum(draws.min(axis=0), truth)
    high = np.maximum(draws.max(axis=0), truth)
    padding = 0.05 * (high - low)
    fig = corner.corner(
        draws,
        labels=LABELS,
        truths=truth,
        truth_color=ORANGE,
        color=BLUE,
        range=list(zip(low - padding, high + padding)),
        bins=30,
        smooth=0.7,
        smooth1d=0.7,
        levels=(0.68, 0.95),
        fill_contours=True,
        plot_density=False,
        max_n_ticks=3,
        labelpad=0.14,
        label_kwargs={"fontsize": 11},
        data_kwargs={"ms": 1, "alpha": 0.15},
        contour_kwargs={"linewidths": 0.9},
        fig=plt.figure(figsize=(12.5, 12.5)),
    )
    for ax in fig.axes:
        ax.tick_params(labelsize=8)
    handles = [Line2D([], [], color=BLUE, label="Posterior"), Line2D([], [], color=ORANGE, label="Injected")]
    if divergent.any():
        corner.overplot_points(fig, draws[divergent], color=DIVERGENCE, marker="x", ms=3, alpha=0.7)
        handles.append(Line2D([], [], color=DIVERGENCE, marker="x", ls="none", label="Divergent transition"))
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.89, 0.91),
        fontsize=12,
    )
    fig.savefig(output / "corner.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python example/hmc/plot.py")
    plot_results()
