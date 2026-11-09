"""Plot saved injection-recovery results without JAX or another sampling run."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = (r"$t_0$ [day]", r"$t_E$ [day]", r"$u_0$", r"$q$", r"$s$", r"$\alpha$ [rad]", r"$\rho$", r"$F_s$", r"$F_b$")
BLUE, ORANGE = "#31688e", "#b45f2a"
OUTPUT = Path(__file__).resolve().parents[2] / "results" / "hmc" / "binary"


def plot_results():
    output = OUTPUT
    config = json.loads((output / "run.json").read_text())
    with np.load(output / "data.npz") as saved:
        times, flux, error, injected = (saved[key] for key in ("times", "flux", "error", "injected_flux"))
    with np.load(output / "samples.npz") as saved:
        samples = {name: saved[name] for name in config["parameter_names"]}
    with np.load(output / "curves.npz") as saved:
        lower, median, upper = np.quantile(saved["model_flux"], [0.16, 0.5, 0.84], axis=0)

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
        2, 1, figsize=(8, 5), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [3, 1]}
    )
    axes[0].errorbar(times, flux, yerr=error, fmt=".", ms=3, color="0.55", elinewidth=0.6, label="Data")
    axes[0].fill_between(times, lower, upper, color=BLUE, alpha=0.25, label="68% model interval")
    axes[0].plot(times, median, color=BLUE, lw=1.2, label="Posterior median")
    axes[0].plot(times, injected, color=ORANGE, lw=1, ls="--", label="Injected")
    axes[0].set_ylabel("Flux")
    axes[0].legend(fontsize=9)
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].plot(times, (flux - median) / error, ".", ms=3, color=BLUE)
    axes[1].set(xlabel="Time [day]", ylabel=r"Residual / $\sigma$")
    fig.savefig(output / "lightcurve.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(9, 7), layout="constrained")
    for ax, name, label, truth in zip(axes.flat, config["parameter_names"], LABELS, config["truth"]):
        ax.hist(samples[name].ravel(), bins=35, density=True, color=BLUE, alpha=0.6)
        ax.axvline(truth, color=ORANGE, lw=1.5, label="Injected")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useOffset=False)
    axes[0, 0].legend(fontsize=9)
    fig.savefig(output / "recovery.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(9, 7), layout="constrained")
    for ax, name, label, truth in zip(axes.flat, config["parameter_names"], LABELS, config["truth"]):
        for i, chain in enumerate(samples[name]):
            ax.plot(chain, lw=0.6, alpha=0.8, label=f"Chain {i + 1}")
        ax.axhline(truth, color="0.35", lw=0.8, ls="--")
        ax.set_ylabel(label)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useOffset=False)
    for ax in axes[-1]:
        ax.set_xlabel("Retained draw")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / "traces.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python example/hmc/plot.py")
    plot_results()
