"""Compare microJAX and VBMicrolensing for a finite-source triple lens."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import VBMicrolensing

from microjax.inverse_ray import TripleMagConfig, mag_triple
from microjax.point_source import critical_and_caustic_curves
from plot_triple_boundary_construction import plot_triple_boundary_construction

jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limb-darkening",
        type=float,
        default=0.0,
        metavar="U1",
        help="linear limb-darkening coefficient (default: uniform source)",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=1e-4,
        help="VBMicrolensing accuracy target (default: %(default)s)",
    )
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--n-limb", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--diagnostic-n-limb",
        type=int,
        default=80,
        help="source-limb samples for the max-residual ICRS diagnostic",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use 24 trajectory points and 80 limb samples for a smoke run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
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
    """Return source coordinates in the binary centre-of-mass frame."""

    tau = (times - t0) / t_e
    y1 = -u0 * np.sin(alpha) + tau * np.cos(alpha)
    y2 = u0 * np.cos(alpha) + tau * np.sin(alpha)
    return np.asarray(y1 + 1j * y2, dtype=np.complex128)


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

    # microJAX exposes source coordinates relative to the centre of mass of
    # the first two lenses. VBMicrolensing accepts explicit lens positions and
    # unnormalised masses, so translate all midpoint-frame lens positions by
    # the binary COM shift and use masses in the ratio 1:q:q3.
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
    """Evaluate one VBMicrolensing light curve serially."""

    if u1 == 0.0:

        def evaluate(w):
            return solver.MultiMag2(float(w.real), float(w.imag), rho)

    else:

        def evaluate(w):
            return solver.MultiMagDark(
                float(w.real),
                float(w.imag),
                rho,
                accuracy,
            )
    return np.asarray([evaluate(position) for position in positions])


def timed_microjax(function, repeats: int) -> tuple[np.ndarray, float, list[float]]:
    """Compile once, then time repeated compiled microJAX evaluations."""

    start = time.perf_counter()
    result = function().block_until_ready()
    warmup_seconds = time.perf_counter() - start
    run_seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function().block_until_ready()
        run_seconds.append(time.perf_counter() - start)
    return np.asarray(result), warmup_seconds, run_seconds


def timed_vbmicrolensing(function, repeats: int) -> tuple[np.ndarray, list[float]]:
    """Time repeated serial VBMicrolensing light-curve evaluations."""

    run_seconds = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        run_seconds.append(time.perf_counter() - start)
    assert result is not None
    return result, run_seconds


def save_plot(
    output_path: Path,
    *,
    times: np.ndarray,
    positions: np.ndarray,
    microjax_magnification: np.ndarray,
    vbml_magnification: np.ndarray,
    relative_difference: np.ndarray,
    params: dict[str, float],
) -> None:
    """Save the light curves, residuals, caustics, and source trajectory."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    s, q = params["s"], params["q"]
    q3, r3, psi = params["q3"], params["r3"], params["psi"]
    rho, u1 = params["rho"], params["u1"]
    _, caustic_curves = critical_and_caustic_curves(
        npts=1000,
        nlenses=3,
        s=s,
        q=q,
        q3=q3,
        r3=r3,
        psi=psi,
    )

    fig, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.1},
    )
    axis.plot(times, microjax_magnification, ".", ms=3, label="microJAX", zorder=1)
    axis.plot(times, vbml_magnification, "-", lw=1.2, label="VBMicrolensing", zorder=2)
    y_limits = axis.get_ylim()
    source_name = "uniform" if u1 == 0.0 else rf"linear limb darkening, $u_1={u1:g}$"
    axis.set_title(
        rf"Triple lens, {source_name}: $\rho={rho:g}$, "
        rf"$s={s:g}$, $q={q:g}$, $q_3={q3:g}$"
    )
    axis.set_ylabel("magnification")
    axis.grid(ls=":")
    axis.legend(loc="upper left")
    axis.set_ylim(*y_limits)

    inset = inset_axes(
        axis,
        width="60%",
        height="60%",
        bbox_transform=axis.transAxes,
        bbox_to_anchor=(0.35, 0.35, 0.6, 0.6),
    )
    for curve in caustic_curves:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="tab:red", lw=0.6)
    source_circles = [
        plt.Circle(
            (position.real, position.imag),
            radius=rho,
            fill=False,
            edgecolor="tab:blue",
        )
        for position in positions
    ]
    inset.add_collection(
        mpl.collections.PatchCollection(
            source_circles,
            match_original=True,
            alpha=0.35,
        )
    )
    a = 0.5 * s
    shift = a * (1.0 - q) / (1.0 + q)
    third_lens = r3 * np.exp(1j * psi)
    lenses = np.asarray([-a + shift, a + shift, third_lens + shift])
    inset.plot(lenses.real, lenses.imag, ".", color="black", ms=5)
    inset.set(
        xlabel=r"$\mathrm{Re}(w)$",
        ylabel=r"$\mathrm{Im}(w)$",
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
    )
    inset.set_aspect("equal")

    residual_floor = np.finfo(float).tiny
    residual_axis.plot(times, np.maximum(relative_difference, residual_floor))
    residual_axis.set_yscale("log")
    residual_axis.yaxis.set_major_locator(
        ticker.LogLocator(
            base=10.0,
            subs=[1.0, 10**-2, 10**-4, 10**-6],
            numticks=10,
        )
    )
    residual_axis.set_ylabel("relative diff")
    residual_axis.set_xlabel("time (days)")
    residual_axis.grid(ls=":")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_points = 24
        args.n_limb = 80
    if not 0.0 <= args.limb_darkening < 1.0:
        raise ValueError("--limb-darkening must satisfy 0 <= U1 < 1")
    if args.accuracy <= 0.0:
        raise ValueError("--accuracy must be positive")
    if min(args.n_points, args.n_limb, args.repeats, args.diagnostic_n_limb) < 1:
        raise ValueError(
            "--n-points, --n-limb, --diagnostic-n-limb, and --repeats must be positive"
        )

    t0, t_e, u0 = 0.0, 10.0, 0.1
    q, s, alpha, rho = 0.1, 1.1, np.deg2rad(50.0), 0.01
    q3, third_lens = 0.01, 0.3 + 1.2j
    r3, psi = abs(third_lens), np.angle(third_lens)
    times = np.linspace(t0 - 0.5 * t_e, t0 + t_e, args.n_points)
    positions = source_trajectory(times, t0=t0, t_e=t_e, u0=u0, alpha=alpha)
    jax_positions = jnp.asarray(positions)
    config = TripleMagConfig(n_limb=args.n_limb)

    microjax_call = jax.jit(
        lambda: mag_triple(
            jax_positions,
            rho,
            s=s,
            q=q,
            q3=q3,
            r3=r3,
            psi=psi,
            u1=args.limb_darkening,
            config=config,
        )
    )
    vbml_solver = configure_vbmicrolensing(
        s=s,
        q=q,
        q3=q3,
        r3=r3,
        psi=psi,
        u1=args.limb_darkening,
        accuracy=args.accuracy,
    )
    def vbml_call():
        return evaluate_vbmicrolensing(
            vbml_solver,
            positions,
            rho=rho,
            u1=args.limb_darkening,
            accuracy=args.accuracy,
        )

    print(f"number of data points: {args.n_points}")
    print("warming up microJAX...", flush=True)
    microjax_magnification, warmup_seconds, microjax_runs = timed_microjax(
        microjax_call,
        args.repeats,
    )
    print("benchmarking VBMicrolensing...", flush=True)
    vbml_magnification, vbml_runs = timed_vbmicrolensing(vbml_call, args.repeats)

    if not np.all(np.isfinite(microjax_magnification)):
        raise RuntimeError("microJAX returned non-finite magnifications")
    if not np.all(np.isfinite(vbml_magnification)):
        raise RuntimeError("VBMicrolensing returned non-finite magnifications")
    relative_difference = np.abs(microjax_magnification - vbml_magnification) / np.abs(vbml_magnification)
    maximum_index = int(np.argmax(relative_difference))
    metrics = {
        "median": float(np.median(relative_difference)),
        "p95": float(np.quantile(relative_difference, 0.95)),
        "maximum": float(relative_difference[maximum_index]),
        "maximum_index": maximum_index,
        "maximum_time": float(times[maximum_index]),
    }

    profile = "uniform" if args.limb_darkening == 0.0 else "limb_dark"
    stem = f"compare_triple_{profile}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / f"{stem}.csv",
        np.column_stack(
            (
                times,
                microjax_magnification,
                vbml_magnification,
                relative_difference,
            )
        ),
        delimiter=",",
        header="time,microjax,vbmicrolensing,relative_difference",
        comments="",
    )

    params = {
        "t0": t0,
        "tE": t_e,
        "u0": u0,
        "alpha": alpha,
        "rho": rho,
        "s": s,
        "q": q,
        "q3": q3,
        "r3": r3,
        "psi": psi,
        "u1": args.limb_darkening,
    }
    report = {
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "vbmicrolensing_version": importlib.metadata.version("VBMicrolensing"),
        "parameters": params,
        "configuration": {
            "n_points": args.n_points,
            "n_limb": args.n_limb,
            "repeats": args.repeats,
            "vbmicrolensing_accuracy": args.accuracy,
            "diagnostic_n_limb": min(args.n_limb, args.diagnostic_n_limb),
            "quick": args.quick,
        },
        "timing_seconds": {
            "microjax_warmup": warmup_seconds,
            "microjax_runs": microjax_runs,
            "microjax_median": statistics.median(microjax_runs),
            "vbmicrolensing_runs": vbml_runs,
            "vbmicrolensing_median": statistics.median(vbml_runs),
        },
        "relative_difference": metrics,
    }
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.no_plot:
        plot_triple_boundary_construction(
            positions[maximum_index],
            rho,
            s=s,
            q=q,
            q3=q3,
            r3=r3,
            psi=psi,
            n_limb=min(args.n_limb, args.diagnostic_n_limb),
            time_value=float(times[maximum_index]),
            relative_residual=float(relative_difference[maximum_index]),
            limb_darkening=args.limb_darkening,
            output_path=args.output_dir / f"{stem}_max_residual_icrs.png",
        )
        save_plot(
            args.output_dir / f"{stem}.png",
            times=times,
            positions=positions,
            microjax_magnification=microjax_magnification,
            vbml_magnification=vbml_magnification,
            relative_difference=relative_difference,
            params=params,
        )

    microjax_median = statistics.median(microjax_runs)
    vbml_median = statistics.median(vbml_runs)
    vbml_method = "MultiMag2" if args.limb_darkening == 0.0 else "MultiMagDark"
    print(
        "computation time: %.3f sec (%.3f ms per point), median of %d, "
        "with VBMicrolensing.%s"
        % (
            vbml_median,
            1000 * vbml_median / args.n_points,
            args.repeats,
            vbml_method,
        )
    )
    print(
        "computation time: %.3f sec (%.3f ms per point), median of %d, "
        "with microJAX mag_triple, n_limb=%d"
        % (
            microjax_median,
            1000 * microjax_median / args.n_points,
            args.repeats,
            args.n_limb,
        )
    )
    print(f"microJAX JIT warm-up time: {warmup_seconds:.3f} sec")
    print(
        "relative difference: median=%.3e, p95=%.3e, max=%.3e at t=%.6g"
        % (
            metrics["median"],
            metrics["p95"],
            metrics["maximum"],
            metrics["maximum_time"],
        )
    )
    print(f"outputs: {args.output_dir / stem}.[csv,json,png]")


if __name__ == "__main__":
    main()
