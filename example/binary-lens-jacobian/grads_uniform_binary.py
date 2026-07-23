"""Binary-lens magnification Jacobian and AD-mode benchmark.

This is the binary-lens counterpart of
``example/triple-lens-jacobian/grads_uniform_paper.py``.  It computes the
Jacobian of a uniform-source light curve with forward-mode automatic
differentiation and benchmarks steady-state execution separately from JIT
compilation. The expensive reverse-mode comparison is opt-in.
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
from jax import lax

from microjax.inverse_ray import BinaryMagConfig, mag_binary
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


def make_model(
    times: jax.Array,
    *,
    u1: float = 0.0,
    n_limb: int,
):
    """Build the new boundary-only light-curve function used by both AD modes."""

    config = BinaryMagConfig(n_limb=n_limb)

    def get_magnification(params: jax.Array) -> jax.Array:
        t0, t_e, u0, q, s, alpha, rho = params
        tau = (times - t0) / t_e
        y1 = -u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)
        y2 = u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
        source_positions = jnp.asarray(y1 + 1j * y2)
        return mag_binary(
            source_positions,
            rho,
            q=q,
            s=s,
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


def chunked_jacrev(function, output_size: int, chunk_size: int):
    """Build a memory-bounded reverse-mode Jacobian transform.

    ``jax.jacrev`` vmaps the pullback over every output basis vector at once.
    That is fast for small outputs but the 500-point production case requires
    more than 80 GiB for this model. Here the same VJP is vmapped over a small
    output block and the blocks are evaluated sequentially with ``lax.map``.
    """

    if chunk_size < 1:
        raise ValueError("reverse chunk size must be positive")

    def jacobian(params):
        output, pullback = jax.vjp(function, params)
        basis = jnp.eye(output_size, dtype=output.dtype)
        pad = (-output_size) % chunk_size
        basis = jnp.pad(basis, ((0, pad), (0, 0)))
        basis_chunks = basis.reshape(-1, chunk_size, output_size)

        def pullback_chunk(cotangents):
            return jax.vmap(lambda cotangent: pullback(cotangent)[0])(cotangents)

        rows = lax.map(pullback_chunk, basis_chunks)
        return rows.reshape(-1, params.shape[0])[:output_size]

    return jacobian


def benchmark(function, params: jax.Array, repeats: int):
    """Return one warm-up time and repeated post-compilation run times."""

    result, warmup_seconds = timed_call(function, params)
    run_seconds = []
    for _ in range(repeats):
        result, elapsed = timed_call(function, params)
        run_seconds.append(elapsed)
    return result, warmup_seconds, run_seconds


def trajectory(params: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return source positions for plotting (outside the timed region)."""

    t0, t_e, u0, _, _, alpha, _ = params
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
):
    """Save the magnification and seven parameter-sensitivity panels."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    _, _, _, q, s, _, rho = params
    positions = trajectory(params, times)
    critical_curves, caustic_curves = critical_and_caustic_curves(npts=500, nlenses=2, q=q, s=s)

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
        bbox_to_anchor=(0.06, 0.05, 0.9, 0.9),
    )
    for curve in caustic_curves:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="red", lw=0.7)
    for curve in critical_curves:
        inset.plot(np.asarray(curve.real), np.asarray(curve.imag), color="green", lw=0.7)
    circles = [plt.Circle((position.real, position.imag), radius=rho, fill=False) for position in positions]
    inset.add_collection(
        mpl.collections.PatchCollection(circles, match_original=True, alpha=0.08, edgecolor="blue", linewidth=0.5)
    )
    inset.plot(-q / (1.0 + q) * s, 0.0, "x", color="black", ms=3)
    inset.plot(1.0 / (1.0 + q) * s, 0.0, "x", color="black", ms=3)
    inset.set(
        xlabel=r"$\mathrm{Re}(w)$",
        ylabel=r"$\mathrm{Im}(w)$",
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
    )
    inset.set_aspect("equal")

    # jax.jacfwd and jax.jacrev both return (n_time, n_parameter) here.
    for index, label in enumerate(PARAMETER_LABELS):
        axes[index + 1].plot(times, jacobian[:, index])
        axes[index + 1].set_ylabel(label)
    axes[-1].set_xlabel("Time (day)")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_benchmark_plot(output_path: Path, medians: dict[str, float]):
    """Save a compact comparison of post-compilation AD execution times."""

    import matplotlib.pyplot as plt

    names = ("forward", "reverse")
    values = [medians[name] for name in names]
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    bars = axis.bar(names, values, color=("tab:blue", "tab:orange"))
    axis.set_ylabel("Median execution time (s)")
    axis.set_title("Binary-lens full Jacobian (JIT compiled)")
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4g} s",
            ha="center",
            va="bottom",
        )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a small CPU-friendly configuration for a smoke benchmark",
    )
    parser.add_argument("--n-points", type=int)
    parser.add_argument("--n-limb", type=int)
    parser.add_argument(
        "--reverse-chunk",
        type=int,
        default=48,
        help="number of output cotangents evaluated together in reverse mode",
    )
    parser.add_argument(
        "--with-reverse",
        action="store_true",
        help="also run the expensive reverse-mode diagnostic comparison",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def resolved_config(args) -> dict[str, int]:
    """Resolve CLI overrides against paper-like or quick defaults."""

    defaults = (
        {
            "n_points": 24,
            "n_limb": 80,
        }
        if args.quick
        else {
            "n_points": 500,
            "n_limb": 500,
        }
    )
    return {name: getattr(args, name) if getattr(args, name) is not None else value for name, value in defaults.items()}


def main():
    args = parse_args()
    config = resolved_config(args)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one")
    if any(value < 1 for value in config.values()):
        raise ValueError("point and limb arguments must be positive")
    if args.reverse_chunk < 1:
        raise ValueError("--reverse-chunk must be positive")

    devices = jax.devices()
    if not args.quick and not any(device.platform == "gpu" for device in devices):
        print(
            "[Warning] No GPU detected. The default configuration is expensive; " "use --quick for a CPU-friendly run."
        )

    t0, t_e, u0 = 0.0, 10.0, 0.1
    q, s, alpha, rho = 0.01, 1.0, np.deg2rad(50.0), 0.01
    params = jnp.asarray([t0, t_e, u0, q, s, alpha, rho])
    times = t0 + jnp.linspace(-0.5 * t_e, t_e, config["n_points"])
    model = make_model(
        times,
        n_limb=config["n_limb"],
    )

    value_function = jax.jit(model)
    forward_function = jax.jit(jax.jacfwd(model))
    reverse_chunk = min(args.reverse_chunk, config["n_points"])
    results = {}
    functions = [("value", value_function), ("forward", forward_function)]
    if args.with_reverse:
        functions.append(("reverse", jax.jit(chunked_jacrev(model, config["n_points"], reverse_chunk))))
    for name, function in functions:
        print(f"benchmarking {name}...", flush=True)
        result, warmup, runs = benchmark(function, params, args.repeats)
        results[name] = (result, warmup, runs)
        print(f"  warm-up={warmup:.3f} s, median={np.median(runs):.3f} s", flush=True)

    magnification_np = np.asarray(results["value"][0])
    forward_np = np.asarray(results["forward"][0])
    if not np.all(np.isfinite(magnification_np)):
        raise RuntimeError("magnification contains non-finite values")
    if not np.all(np.isfinite(forward_np)):
        raise RuntimeError("forward Jacobian contains non-finite values")
    reverse_np = None
    if args.with_reverse:
        reverse_np = np.asarray(results["reverse"][0])
        if not np.all(np.isfinite(reverse_np)):
            raise RuntimeError("reverse Jacobian contains non-finite values")
        np.testing.assert_allclose(forward_np, reverse_np, rtol=1e-8, atol=1e-9)

    medians = {name: float(np.median(results[name][2])) for name in results}
    maximum_difference = float(np.max(np.abs(forward_np - reverse_np))) if reverse_np is not None else None
    report = {
        "device": str(devices[0]),
        "device_kind": getattr(devices[0], "device_kind", "unknown"),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "mag_binary_implementation": ("retry-free best-effort; uniform G15/K31 fixed-1"),
        "parameter_names": list(PARAMETER_NAMES),
        "config": {
            **config,
            "with_reverse": args.with_reverse,
            "reverse_chunk": reverse_chunk,
            "repeats": args.repeats,
        },
        "reverse_implementation": ("chunked jax.vjp + lax.map" if args.with_reverse else None),
        "warmup_seconds": {name: results[name][1] for name in results},
        "run_seconds": {name: results[name][2] for name in results},
        "median_seconds": medians,
        "reverse_over_forward": (medians["reverse"] / medians["forward"] if args.with_reverse else None),
        "maximum_jacobian_absolute_difference": maximum_difference,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "magnification.csv",
        np.column_stack((np.asarray(times), magnification_np)),
        delimiter=",",
        header="time,magnification",
        comments="",
    )
    np.save(args.output_dir / "jacobian_forward.npy", forward_np)
    if reverse_np is not None:
        np.save(args.output_dir / "jacobian_reverse.npy", reverse_np)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        save_jacobian_plot(
            args.output_dir / "binary_jacobian.png",
            np.asarray(times),
            magnification_np,
            forward_np,
            np.asarray(params),
        )
        if args.with_reverse:
            save_benchmark_plot(args.output_dir / "ad_benchmark.png", medians)

    print(f"device: {devices[0]}")
    print(
        f"median of {args.repeats} compiled runs: "
        f"value={medians['value']:.3f} s, forward={medians['forward']:.3f} s"
    )
    if args.with_reverse:
        print(f"reverse={medians['reverse']:.3f} s")
        print(f"reverse / forward: {report['reverse_over_forward']:.3f}x")
        print(f"max |J_forward - J_reverse|: {maximum_difference:.3e}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
