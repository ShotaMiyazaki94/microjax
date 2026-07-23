"""Triple-lens uniform-source magnification and forward Jacobian benchmark.

This is the triple-lens counterpart of
``example/binary-lens-jacobian/grads_uniform_binary.py``. It benchmarks the
current ``mag_triple`` API, separating JIT warm-up from compiled execution,
and differentiates all ten trajectory, source, and lens parameters with
forward-mode automatic differentiation.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from microjax.inverse_ray import TripleMagConfig, mag_triple
from microjax.point_source import critical_and_caustic_curves

jax.config.update("jax_enable_x64", True)

PARAMETER_NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho", "q3", "r3", "psi")
PARAMETER_LABELS = (
    r"$\partial A / \partial t_0$",
    r"$\partial A / \partial t_E$",
    r"$\partial A / \partial u_0$",
    r"$\partial A / \partial q$",
    r"$\partial A / \partial s$",
    r"$\partial A / \partial \alpha$",
    r"$\partial A / \partial \rho$",
    r"$\partial A / \partial q_3$",
    r"$\partial A / \partial r_3$",
    r"$\partial A / \partial \psi$",
)


def make_model(
    times: jax.Array,
    *,
    u1: float = 0.0,
    n_limb: int,
):
    """Build the triple boundary light-curve function used by forward AD."""

    config = TripleMagConfig(n_limb=n_limb)

    def get_magnification(params: jax.Array) -> jax.Array:
        t0, t_e, u0, q, s, alpha, rho, q3, r3, psi = params
        tau = (times - t0) / t_e
        y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
        y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
        source_positions = jnp.asarray(y1 + 1j * y2)
        return mag_triple(
            source_positions,
            rho,
            s=s,
            q=q,
            q3=q3,
            r3=r3,
            psi=psi,
            u1=u1,
            config=config,
        )

    return get_magnification


def timed_call(function, params: jax.Array):
    """Time one asynchronous JAX call, synchronising before returning."""

    start = time.perf_counter()
    result = function(params)
    result.block_until_ready()
    return result, time.perf_counter() - start


def benchmark(function, params: jax.Array, repeats: int):
    """Return one warm-up time and repeated post-compilation run times."""

    result, warmup_seconds = timed_call(function, params)
    run_seconds = []
    for _ in range(repeats):
        result, elapsed = timed_call(function, params)
        run_seconds.append(elapsed)
    return result, warmup_seconds, run_seconds


def trajectory(params: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return source positions for plotting outside the timed region."""

    t0, t_e, u0, _, _, alpha, _, _, _, _ = params
    tau = (times - t0) / t_e
    y1 = -u0 * np.sin(alpha) + tau * np.cos(alpha)
    y2 = u0 * np.cos(alpha) + tau * np.sin(alpha)
    return y1 + 1j * y2


def save_jacobian_plot(
    output_path: Path,
    times: np.ndarray,
    magnification: np.ndarray,
    jacobian: np.ndarray,
    params: np.ndarray,
    *,
    source_label: str | None = None,
):
    """Save the magnification and ten parameter-sensitivity panels."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    _, _, _, q, s, _, rho, q3, r3, psi = params
    positions = trajectory(params, times)
    critical_curves, caustic_curves = critical_and_caustic_curves(
        npts=1000,
        nlenses=3,
        q=q,
        s=s,
        q3=q3,
        r3=r3,
        psi=psi,
    )

    fig, axes = plt.subplots(
        len(PARAMETER_NAMES) + 1,
        1,
        figsize=(12, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [8] + [1.2] * len(PARAMETER_NAMES)},
    )
    axes[0].plot(times, magnification, color="black")
    axes[0].set_ylabel("Magnification")
    if source_label is not None:
        axes[0].set_title(source_label)

    inset = inset_axes(
        axes[0],
        width="70%",
        height="70%",
        bbox_transform=axes[0].transAxes,
        bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),
    )
    for curve in caustic_curves:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="red", lw=0.7)
    for curve in critical_curves:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="green", lw=0.7)
    circles = [plt.Circle((position.real, position.imag), radius=rho, fill=False) for position in positions]
    inset.add_collection(
        mpl.collections.PatchCollection(circles, match_original=True, alpha=0.05, edgecolor="blue", linewidth=0.5)
    )
    shift = 0.5 * s * (1.0 - q) / (1.0 + q)
    lens_positions = np.asarray([-0.5 * s + shift, 0.5 * s + shift, r3 * np.exp(1j * psi) + shift])
    inset.plot(lens_positions.real, lens_positions.imag, "x", color="black", ms=3)
    inset.set(
        xlabel=r"$\mathrm{Re}(w)$",
        ylabel=r"$\mathrm{Im}(w)$",
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
    )
    inset.set_aspect("equal")

    for index, label in enumerate(PARAMETER_LABELS):
        axes[index + 1].plot(times, jacobian[:, index])
        axes[index + 1].set_ylabel(label, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Time (day)")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a small configuration for a smoke benchmark",
    )
    parser.add_argument("--n-points", type=int)
    parser.add_argument("--n-limb", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolved_config(args) -> dict[str, int]:
    """Resolve CLI overrides against full-size or quick defaults."""

    defaults = {"n_points": 24, "n_limb": 80} if args.quick else {"n_points": 1000, "n_limb": 500}
    return {name: getattr(args, name) if getattr(args, name) is not None else value for name, value in defaults.items()}


def main():
    args = parse_args()
    config = resolved_config(args)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one")
    if any(value < 1 for value in config.values()):
        raise ValueError("point and limb arguments must be positive")

    devices = jax.devices()
    if not args.quick and not any(device.platform == "gpu" for device in devices):
        print("[Warning] No GPU detected. The default configuration is expensive; use --quick for a smoke run.")

    t0, t_e, u0 = 0.0, 10.0, 0.1
    q, s, alpha, rho = 0.1, 1.1, np.deg2rad(50.0), 0.01
    q3, r3_complex = 0.01, 0.3 + 1.2j
    r3, psi = np.abs(r3_complex), np.angle(r3_complex)
    params = jnp.asarray([t0, t_e, u0, q, s, alpha, rho, q3, r3, psi])
    times = t0 + jnp.linspace(-0.5 * t_e, t_e, config["n_points"])
    model = make_model(
        times,
        n_limb=config["n_limb"],
    )

    functions = {
        "value": jax.jit(model),
        "forward": jax.jit(jax.jacfwd(model)),
    }
    results = {}
    for name, function in functions.items():
        print(f"benchmarking {name}...", flush=True)
        result, warmup, runs = benchmark(function, params, args.repeats)
        results[name] = (result, warmup, runs)
        print(f"  warm-up={warmup:.3f} s, median={np.median(runs):.3f} s", flush=True)

    magnification = np.asarray(results["value"][0])
    forward = np.asarray(results["forward"][0])
    if not np.all(np.isfinite(magnification)):
        raise RuntimeError("magnification contains non-finite values")
    if not np.all(np.isfinite(forward)):
        raise RuntimeError("forward Jacobian contains non-finite values")

    medians = {name: float(np.median(results[name][2])) for name in results}
    report = {
        "device": str(devices[0]),
        "device_kind": getattr(devices[0], "device_kind", "unknown"),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "api": "microjax.inverse_ray.mag_triple",
        "automatic_differentiation": "jax.jacfwd",
        "parameter_names": list(PARAMETER_NAMES),
        "config": {
            **config,
            "repeats": args.repeats,
        },
        "warmup_seconds": {name: results[name][1] for name in results},
        "run_seconds": {name: results[name][2] for name in results},
        "median_seconds": medians,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "magnification.csv",
        np.column_stack((np.asarray(times), magnification)),
        delimiter=",",
        header="time,magnification",
        comments="",
    )
    np.save(args.output_dir / "jacobian_forward.npy", forward)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        save_jacobian_plot(
            args.output_dir / "triple_jacobian.png",
            np.asarray(times),
            magnification,
            forward,
            np.asarray(params),
        )

    print(f"device: {devices[0]}")
    print(
        f"median of {args.repeats} compiled runs: "
        f"value={medians['value']:.3f} s, forward={medians['forward']:.3f} s"
    )
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
