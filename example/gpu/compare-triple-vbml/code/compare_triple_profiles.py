"""Compare uniform and limb-darkened GPU ICRS on one triple-lens track."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
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

from microjax.inverse_ray import TripleMagConfig, mag_triple
from microjax.point_source import critical_and_caustic_curves
from plot_triple_boundary_construction import plot_triple_boundary_construction

jax.config.update("jax_enable_x64", True)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "outputs"


class ProfileResult(NamedTuple):
    name: str
    u1: float
    vbml: np.ndarray
    gpu: np.ndarray
    relative_error: np.ndarray
    finite: np.ndarray
    warmup_seconds: float
    vbml_seconds: float
    gpu_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accuracy",
        type=float,
        default=1e-4,
        help="VBMicrolensing accuracy target (default: %(default)s)",
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--points", "--n-points", dest="points", type=int, default=1000)
    parser.add_argument("--n-limb", type=int, default=TripleMagConfig().n_limb)
    parser.add_argument("--u1", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--diagnostic-n-limb",
        type=int,
        default=80,
        help="source-limb samples for each maximum-residual ICRS diagnostic",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 24 trajectory points and 80 limb samples for a smoke run",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def source_trajectory(
    times: np.ndarray,
    *,
    t0: float,
    t_e: float,
    u0: float,
    alpha: float,
) -> np.ndarray:
    """Return source coordinates in the first two lenses' COM frame."""

    tau = (times - t0) / t_e
    real = -u0 * np.sin(alpha) + tau * np.cos(alpha)
    imag = u0 * np.cos(alpha) + tau * np.sin(alpha)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def configure_vbmicrolensing(
    *,
    s: float,
    q: float,
    q3: float,
    r3: float,
    psi: float,
    u1: float,
    accuracy: float,
) -> VBMicrolensing.VBMicrolensing:
    """Configure VBMicrolensing in microJAX's public coordinate frame."""

    solver = VBMicrolensing.VBMicrolensing()
    a = 0.5 * s
    binary_com_shift = a * (1.0 - q) / (1.0 + q)
    third_lens = r3 * np.exp(1j * psi)
    solver.SetLensGeometry(
        [
            -a + binary_com_shift,
            0.0,
            1.0,
            a + binary_com_shift,
            0.0,
            q,
            third_lens.real + binary_com_shift,
            third_lens.imag,
            q3,
        ]
    )
    solver.Tol = accuracy
    solver.RelTol = accuracy
    solver.a1 = u1
    return solver


def evaluate_vbmicrolensing(
    solver: VBMicrolensing.VBMicrolensing,
    positions: np.ndarray,
    *,
    rho: float,
    u1: float,
    accuracy: float,
) -> np.ndarray:
    """Evaluate one serial VBMicrolensing light curve."""

    if u1 == 0.0:

        def evaluate(position):
            return solver.MultiMag2(float(position.real), float(position.imag), rho)

    else:

        def evaluate(position):
            return solver.MultiMagDark(
                float(position.real),
                float(position.imag),
                rho,
                accuracy,
            )

    return np.asarray([evaluate(position) for position in positions])


def median_runtime(function, *, repeats: int) -> tuple[object, float]:
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def evaluate_profile(
    name: str,
    u1: float,
    positions: np.ndarray,
    params: dict[str, float],
    args: argparse.Namespace,
) -> ProfileResult:
    jax_positions = jnp.asarray(positions)
    config = TripleMagConfig(n_limb=args.n_limb)

    def evaluate_gpu():
        return mag_triple(
            jax_positions,
            params["rho"],
            s=params["s"],
            q=params["q"],
            q3=params["q3"],
            r3=params["r3"],
            psi=params["psi"],
            u1=u1,
            config=config,
        )

    gpu_call = jax.jit(evaluate_gpu)
    start = time.perf_counter()
    gpu_call().block_until_ready()
    warmup_seconds = time.perf_counter() - start
    gpu_result, gpu_seconds = median_runtime(gpu_call, repeats=args.repeats)
    gpu = np.asarray(gpu_result)

    solver = configure_vbmicrolensing(
        s=params["s"],
        q=params["q"],
        q3=params["q3"],
        r3=params["r3"],
        psi=params["psi"],
        u1=u1,
        accuracy=args.accuracy,
    )

    def evaluate_reference():
        return evaluate_vbmicrolensing(
            solver,
            positions,
            rho=params["rho"],
            u1=u1,
            accuracy=args.accuracy,
        )

    vbml_result, vbml_seconds = median_runtime(
        evaluate_reference,
        repeats=args.repeats,
    )
    vbml = np.asarray(vbml_result)
    finite = np.isfinite(gpu) & np.isfinite(vbml)
    if not np.any(finite):
        raise RuntimeError(f"{name}: no finite paired magnifications")
    relative_error = np.full_like(vbml, np.nan)
    relative_error[finite] = np.abs(gpu[finite] - vbml[finite]) / np.abs(vbml[finite])
    return ProfileResult(
        name,
        u1,
        vbml,
        gpu,
        relative_error,
        finite,
        warmup_seconds,
        vbml_seconds,
        gpu_seconds,
    )


def profile_summary(result: ProfileResult, args: argparse.Namespace) -> dict:
    errors = result.relative_error[result.finite]
    summary = {
        "u1": result.u1,
        "microjax_warmup_seconds": result.warmup_seconds,
        "microjax_seconds": result.gpu_seconds,
        "vbmicrolensing_seconds": result.vbml_seconds,
        "microjax_ms_per_point": 1e3 * result.gpu_seconds / args.points,
        "vbmicrolensing_ms_per_point": 1e3 * result.vbml_seconds / args.points,
        "gpu_over_vbmicrolensing": result.gpu_seconds / result.vbml_seconds,
        "nonfinite_pairs": int(np.sum(~result.finite)),
        "finite_residuals_above_rtol": int(np.sum(errors > args.rtol)),
        "finite_relative_error": {
            "median": float(np.median(errors)),
            "p95": float(np.quantile(errors, 0.95)),
            "p99": float(np.quantile(errors, 0.99)),
            "maximum": float(np.max(errors)),
        },
    }
    print(f"\n{result.name}")
    print(
        "VBMicrolensing: %.3f s (%.3f ms/point); microJAX GPU: %.3f s "
        "(%.3f ms/point); GPU/VBML: %.2fx"
        % (
            result.vbml_seconds,
            summary["vbmicrolensing_ms_per_point"],
            result.gpu_seconds,
            summary["microjax_ms_per_point"],
            summary["gpu_over_vbmicrolensing"],
        )
    )
    stats = summary["finite_relative_error"]
    print(
        "relative error: median=%.3e, p95=%.3e, p99=%.3e, max=%.3e; "
        "nonfinite=%d; above rtol=%d"
        % (
            stats["median"],
            stats["p95"],
            stats["p99"],
            stats["maximum"],
            summary["nonfinite_pairs"],
            summary["finite_residuals_above_rtol"],
        )
    )
    return summary


def add_geometry_inset(axis, positions: np.ndarray, params: dict[str, float]) -> None:
    _, caustics = critical_and_caustic_curves(
        nlenses=3,
        npts=500,
        s=params["s"],
        q=params["q"],
        q3=params["q3"],
        r3=params["r3"],
        psi=params["psi"],
    )
    inset = inset_axes(
        axis, width="37%", height="60%", loc="upper right", borderpad=1.0
    )
    for caustic in caustics:
        inset.plot(
            np.asarray(caustic.real), np.asarray(caustic.imag), color="tab:red", lw=0.7
        )
    inset.plot(positions.real, positions.imag, color="0.4", lw=0.7)
    indices = np.linspace(0, positions.size - 1, 13, dtype=int)
    circles = [
        plt.Circle(
            (positions[index].real, positions[index].imag),
            radius=params["rho"],
            fill=False,
            ec="tab:blue",
            lw=0.5,
        )
        for index in indices
    ]
    inset.add_collection(
        mpl.collections.PatchCollection(circles, match_original=True, alpha=0.5)
    )
    a = 0.5 * params["s"]
    shift = a * (1.0 - params["q"]) / (1.0 + params["q"])
    third = params["r3"] * np.exp(1j * params["psi"])
    lenses = np.asarray([-a + shift, a + shift, third + shift])
    inset.plot(lenses.real, lenses.imag, "x", color="black", ms=4)
    inset.set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$")
    inset.set_aspect("equal", adjustable="box")
    inset.tick_params(labelsize=9)
    inset.grid(ls=":", lw=0.5)


def plot_residual(
    axis, times: np.ndarray, result: ProfileResult, args: argparse.Namespace
) -> None:
    floor = np.finfo(float).tiny
    axis.plot(times, np.maximum(result.relative_error, floor), lw=1.15)
    if np.any(~result.finite):
        axis.plot(
            times[~result.finite],
            np.full(np.sum(~result.finite), args.rtol),
            "x",
            color="tab:red",
        )
    axis.axhline(
        args.rtol, color="0.35", ls="--", lw=1.0, label=f"rtol={args.rtol:.0e}"
    )
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
    axis.grid(ls=":", lw=0.6)
    axis.legend(loc="upper right", fontsize=9)


def plot_comparison(
    times: np.ndarray,
    positions: np.ndarray,
    uniform: ProfileResult,
    limb_dark: ProfileResult,
    params: dict[str, float],
    args: argparse.Namespace,
) -> Path:
    figure = plt.figure(figsize=(10.5, 8.0))
    grid = figure.add_gridspec(3, 1, height_ratios=(3.7, 1.0, 1.0), hspace=0.07)
    light_curve = figure.add_subplot(grid[0, 0])
    uniform_residual = figure.add_subplot(grid[1, 0], sharex=light_curve)
    limb_residual = figure.add_subplot(
        grid[2, 0], sharex=light_curve, sharey=uniform_residual
    )

    for result, color in ((uniform, "tab:blue"), (limb_dark, "tab:orange")):
        light_curve.plot(
            times, result.vbml, color=color, lw=1.35, label=f"{result.name} — VBML"
        )
        light_curve.plot(
            times,
            result.gpu,
            ".",
            color=color,
            ms=2.1,
            alpha=0.75,
            label=f"{result.name} — microJAX",
        )
    light_curve.set_title(
        "Triple finite-source magnification  "
        rf"($\rho={params['rho']:.3g},\ s={params['s']:.2f},\ q={params['q']:.2g},\ "
        rf"q_3={params['q3']:.2g},\ u_1={args.u1:.2f}$)"
    )
    light_curve.set_ylabel("magnification")
    light_curve.set_yscale("log")
    light_curve.grid(ls=":", lw=0.7)
    light_curve.legend(loc="upper left", fontsize=10)
    light_curve.tick_params(labelbottom=False)
    add_geometry_inset(light_curve, positions, params)

    plot_residual(uniform_residual, times, uniform, args)
    plot_residual(limb_residual, times, limb_dark, args)
    uniform_residual.set_ylabel("Uniform\nrelative res.")
    uniform_residual.tick_params(labelbottom=False)
    limb_residual.set_ylabel("LD\nrelative res.")
    limb_residual.set_xlabel("time (days)")
    maximum = max(
        np.nanmax(uniform.relative_error),
        np.nanmax(limb_dark.relative_error),
    )
    uniform_residual.set_ylim(1e-7, max(10.0 * args.rtol, 1.5 * maximum))
    output = OUTPUT_DIRECTORY / "compare_triple_uniform_limb_dark.png"
    figure.subplots_adjust(left=0.145, right=0.975, bottom=0.075, top=0.935)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def write_maximum_residual_diagnostic(
    result: ProfileResult,
    positions: np.ndarray,
    times: np.ndarray,
    params: dict[str, float],
    args: argparse.Namespace,
) -> Path:
    candidates = np.flatnonzero(result.finite)
    index = int(candidates[np.argmax(result.relative_error[candidates])])
    suffix = "uniform" if result.u1 == 0.0 else "limb_dark"
    output = (
        OUTPUT_DIRECTORY / f"compare_triple_profiles_{suffix}_max_residual_icrs.png"
    )
    plot_triple_boundary_construction(
        positions[index],
        params["rho"],
        s=params["s"],
        q=params["q"],
        q3=params["q3"],
        r3=params["r3"],
        psi=params["psi"],
        n_limb=min(args.n_limb, args.diagnostic_n_limb),
        time_value=float(times[index]),
        relative_residual=float(result.relative_error[index]),
        limb_darkening=result.u1,
        output_path=output,
    )
    return output


def main() -> None:
    args = parse_args()
    if args.quick:
        args.points = 24
        args.n_limb = 80
    if args.accuracy <= 0.0 or args.rtol <= 0.0:
        raise ValueError("accuracy and rtol must be positive")
    if min(args.points, args.n_limb, args.repeats, args.diagnostic_n_limb) < 1:
        raise ValueError(
            "points, n-limb, diagnostic-n-limb, and repeats must be positive"
        )

    t0, t_e, u0 = 0.0, 10.0, 0.1
    q, s, alpha, rho = 0.1, 1.1, np.deg2rad(50.0), 0.01
    q3, third_lens = 0.01, 0.3 + 1.2j
    params = {
        "t0": t0,
        "tE": t_e,
        "u0": u0,
        "alpha": alpha,
        "rho": rho,
        "s": s,
        "q": q,
        "q3": q3,
        "r3": abs(third_lens),
        "psi": np.angle(third_lens),
    }
    times = np.linspace(t0 - 0.5 * t_e, t0 + t_e, args.points)
    positions = source_trajectory(times, t0=t0, t_e=t_e, u0=u0, alpha=alpha)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print("number of data points:", args.points)
    uniform = evaluate_profile("Uniform", 0.0, positions, params, args)
    limb_dark = evaluate_profile("Limb darkening", args.u1, positions, params, args)
    summaries = {
        "uniform": profile_summary(uniform, args),
        "limb_dark": profile_summary(limb_dark, args),
    }
    payload = {
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "vbmicrolensing_version": importlib.metadata.version("VBMicrolensing"),
        "parameters": {**params, "limb_dark_u1": args.u1},
        "configuration": {
            "points": args.points,
            "n_limb": args.n_limb,
            "repeats": args.repeats,
            "rtol": args.rtol,
            "vbmicrolensing_accuracy": args.accuracy,
            "diagnostic_n_limb": min(args.n_limb, args.diagnostic_n_limb),
            "quick": args.quick,
        },
        "profiles": summaries,
    }
    benchmark = OUTPUT_DIRECTORY / "benchmark_triple_profiles.json"
    benchmark.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    outputs = []
    if not args.no_plot:
        outputs.append(
            plot_comparison(times, positions, uniform, limb_dark, params, args)
        )
        outputs.extend(
            write_maximum_residual_diagnostic(result, positions, times, params, args)
            for result in (uniform, limb_dark)
        )
    print("\nbenchmark:", benchmark)
    for output in outputs:
        print("output:", output)
    failures = sum(item["finite_residuals_above_rtol"] for item in summaries.values())
    if failures:
        raise RuntimeError(
            f"combined comparison found {failures} finite residuals above rtol"
        )


if __name__ == "__main__":
    main()
