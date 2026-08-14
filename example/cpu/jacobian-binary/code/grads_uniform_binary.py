"""Forward-mode Jacobian of a binary-lens light curve on the CPU backend."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from microjax.inverse_ray import mag_binary
from microjax.point_source import critical_and_caustic_curves

jax.config.update("jax_enable_x64", True)

PARAMETER_NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho")
PARAMETER_LABELS = (
    r"$\partial A / \partial t_0$",
    r"$\partial A / \partial t_E$",
    r"$\partial A / \partial u_0$",
    r"$\partial A / \partial q$",
    r"$\partial A / \partial s$",
    r"$\partial A / \partial \alpha$",
    r"$\partial A / \partial \rho$",
)


def source_trajectory(params: jax.Array, times: jax.Array) -> jax.Array:
    """Return the complex source trajectory in the binary centre-of-mass frame."""

    t0, t_e, u0, _, _, alpha, _ = params
    tau = (times - t0) / t_e
    y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
    y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
    return y1 + 1.0j * y2


def make_model(times: jax.Array, *, u1: float, rtol: float):
    """Build the seven-parameter CPU light-curve model."""

    def model(params: jax.Array) -> jax.Array:
        _, _, _, q, s, _, rho = params
        result = mag_binary(
            source_trajectory(params, times),
            rho,
            q=q,
            s=s,
            u1=u1,
            backend="cpu",
            return_info=True,
        )
        return result.magnification

    return model


def make_diagnostics(times: jax.Array, *, u1: float, rtol: float):
    """Build an evaluator exposing the CPU scheduler's tier and status arrays."""

    def diagnostics(params: jax.Array):
        _, _, _, q, s, _, rho = params
        return mag_binary(
            source_trajectory(params, times),
            rho,
            q=q,
            s=s,
            u1=u1,
            backend="cpu",
            return_info=True,
        )

    return diagnostics


def timed_call(function, params: jax.Array):
    """Time one JAX call, synchronising before returning."""

    start = time.perf_counter()
    result = function(params)
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def benchmark(function, params: jax.Array, repeats: int):
    """Return compile-plus-first-call time and repeated compiled timings."""

    result, warmup_seconds = timed_call(function, params)
    run_seconds = []
    for _ in range(repeats):
        result, elapsed = timed_call(function, params)
        run_seconds.append(elapsed)
    return result, warmup_seconds, run_seconds


def trajectory(params: np.ndarray, times: np.ndarray) -> np.ndarray:
    """NumPy trajectory used only for plotting."""

    return np.asarray(source_trajectory(jnp.asarray(params), jnp.asarray(times)))


def save_jacobian_plot(
    output_path: Path,
    times: np.ndarray,
    magnification: np.ndarray,
    jacobian: np.ndarray,
    params: np.ndarray,
):
    """Save magnification and all seven parameter sensitivities."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    _, _, _, q, s, _, rho = params
    positions = trajectory(params, times)
    critical, caustics = critical_and_caustic_curves(npts=500, nlenses=2, q=q, s=s)
    fig, axes = plt.subplots(
        len(PARAMETER_NAMES) + 1,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [7] + [1] * len(PARAMETER_NAMES)},
    )
    axes[0].plot(times, magnification, color="black")
    axes[0].set_ylabel("Magnification")

    inset = inset_axes(
        axes[0],
        width="68%",
        height="68%",
        bbox_transform=axes[0].transAxes,
        bbox_to_anchor=(0.25, 0.05, 0.9, 0.9),
    )
    for curve in caustics:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="red", lw=0.7)
    for curve in critical:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="green", lw=0.7)
    circles = [plt.Circle((point.real, point.imag), radius=rho, fill=False) for point in positions]
    inset.add_collection(
        mpl.collections.PatchCollection(
            circles,
            match_original=True,
            alpha=0.08,
            edgecolor="blue",
            linewidth=0.5,
        )
    )
    inset.plot(-q / (1.0 + q) * s, 0.0, "x", color="black", ms=3)
    inset.plot(1.0 / (1.0 + q) * s, 0.0, "x", color="black", ms=3)
    inset.set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    inset.set_aspect("equal")

    for index, label in enumerate(PARAMETER_LABELS):
        axes[index + 1].plot(times, jacobian[:, index])
        axes[index + 1].set_ylabel(label)
    axes[-1].set_xlabel("Time (day)")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_plot(
    output_path: Path,
    warmups: dict[str, float],
    medians: dict[str, float],
):
    """Plot first-call and steady-state times for the primal and Jacobian."""

    import matplotlib.pyplot as plt

    names = ("magnification", "forward Jacobian")
    keys = ("value", "forward")
    positions = np.arange(len(keys))
    width = 0.36
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    first = axis.bar(positions - width / 2, [warmups[key] for key in keys], width, label="compile + first")
    steady = axis.bar(positions + width / 2, [medians[key] for key in keys], width, label="compiled median")
    axis.set_xticks(positions, names)
    axis.set_yscale("log")
    axis.set_ylabel("Execution time (s, log scale)")
    axis.set_title("CPU binary-lens forward-mode AD")
    axis.legend()
    axis.bar_label(first, fmt="%.3g", padding=2, fontsize=8)
    axis.bar_label(steady, fmt="%.3g", padding=2, fontsize=8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use 24 points for a smoke run")
    parser.add_argument("--n-points", type=int)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "uniform",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def run_example(
    *,
    u1: float,
    params: jax.Array,
    output_dir: Path,
    plot_name: str,
    args,
):
    """Run one uniform or fixed-limb-darkening CPU Jacobian example."""

    n_points = args.n_points if args.n_points is not None else (24 if args.quick else 500)
    if n_points < 1 or args.repeats < 1 or not 0.0 < args.rtol <= 1.0e-2:
        raise ValueError("require n_points >= 1, repeats >= 1, and 0 < rtol <= 1e-2")

    times = params[0] + jnp.linspace(-0.5 * params[1], 0.5 * params[1], n_points)
    #times = params[0] + jnp.linspace(-0.5 * params[1], params[1], n_points)
    model = make_model(times, u1=u1, rtol=args.rtol)
    value_function = jax.jit(model)
    forward_function = jax.jit(jax.jacfwd(model))
    results = {}
    for name, function in (("value", value_function), ("forward", forward_function)):
        print(f"benchmarking {name}...", flush=True)
        result, warmup, runs = benchmark(function, params, args.repeats)
        results[name] = (result, warmup, runs)
        print(f"  compile + first={warmup:.3f} s, median={np.median(runs):.3f} s", flush=True)

    magnification = np.asarray(results["value"][0])
    forward = np.asarray(results["forward"][0])
    diagnostics = jax.jit(make_diagnostics(times, u1=u1, rtol=args.rtol))(params)
    jax.block_until_ready(diagnostics)
    status = np.asarray(diagnostics.status)
    if not np.all(np.isfinite(magnification)) or not np.all(np.isfinite(forward)):
        raise RuntimeError("CPU magnification or forward Jacobian contains non-finite values")
    if np.any(status != 0):
        values, counts = np.unique(status, return_counts=True)
        raise RuntimeError(f"CPU ICRS returned nonzero statuses: {dict(zip(values.tolist(), counts.tolist()))}")

    medians = {name: float(np.median(result[2])) for name, result in results.items()}
    warmups = {name: float(result[1]) for name, result in results.items()}
    tier_values, tier_counts = np.unique(np.asarray(diagnostics.tier), return_counts=True)
    status_values, status_counts = np.unique(status, return_counts=True)
    report = {
        "device": str(jax.devices()[0]),
        "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "api": "microjax.inverse_ray.mag_binary(backend='cpu')",
        "automatic_differentiation": "jax.jacfwd",
        "reverse_mode_supported": False,
        "differentiated_parameter_names": list(PARAMETER_NAMES),
        "limb_darkening_coefficient_u1": u1,
        "u1_is_differentiated": False,
        "config": {"n_points": n_points, "rtol": args.rtol, "repeats": args.repeats},
        "warmup_seconds": warmups,
        "run_seconds": {name: result[2] for name, result in results.items()},
        "median_seconds": medians,
        "forward_over_value": medians["forward"] / medians["value"],
        "tier_counts": dict(zip(map(str, tier_values.tolist()), tier_counts.tolist())),
        "status_counts": dict(zip(map(str, status_values.tolist()), status_counts.tolist())),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_dir / "magnification.csv",
        np.column_stack((np.asarray(times), magnification)),
        delimiter=",",
        header="time,magnification",
        comments="",
    )
    np.save(output_dir / "jacobian_forward.npy", forward)
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        save_jacobian_plot(output_dir / plot_name, np.asarray(times), magnification, forward, np.asarray(params))
        save_benchmark_plot(output_dir / "ad_benchmark.png", warmups, medians)

    print(f"device: {jax.devices()[0]}")
    print(f"selected CPU tiers: {report['tier_counts']}")
    print(f"CPU statuses: {report['status_counts']}")
    print(f"outputs: {output_dir}")


def main():
    args = parse_args()
    #params = jnp.asarray([0.0, 10.0, 0.01, 1e-4, 1.0, np.deg2rad(50.0), 0.005])
    params = jnp.asarray([0.0, 30.0, 0.0, 0.03, 0.85, np.deg2rad(45.0), 1e-2])
    run_example(
        u1=0.0,
        params=params,
        output_dir=args.output_dir,
        plot_name="binary_jacobian.png",
        args=args,
    )


if __name__ == "__main__":
    main()
