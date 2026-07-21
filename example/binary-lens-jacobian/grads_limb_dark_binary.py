"""Limb-darkened binary-lens magnification Jacobian and AD benchmark.

This companion to ``grads_uniform_binary.py`` fixes a linear limb-darkening
coefficient ``u1`` and differentiates the resulting light curve with respect
to ``t0, tE, u0, q, s, alpha, rho``.  It compares forward mode with a
memory-bounded reverse-mode VJP and reports JIT warm-up separately from
steady-state execution.

``u1`` is intentionally not an eighth differentiated parameter: the public
``mag_binary`` API uses it as a static JIT argument to select the uniform or
linear limb-darkened kernel. Forward mode is the default and production path;
the expensive reverse-mode comparison is available only as an explicit
diagnostic.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from grads_uniform_binary import (
    PARAMETER_NAMES,
    benchmark,
    chunked_jacrev,
    make_model,
    resolved_config,
    save_benchmark_plot,
    save_jacobian_plot,
)

jax.config.update("jax_enable_x64", True)

JACOBIAN_COMPARISON_RTOL = 5e-7
JACOBIAN_COMPARISON_ATOL = 5e-8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a small CPU-friendly configuration for a smoke benchmark",
    )
    parser.add_argument("--u1", type=float, default=0.5)
    parser.add_argument("--n-points", type=int)
    parser.add_argument("--n-limb", type=int)
    parser.add_argument("--margin-r", type=float, default=0.5)
    parser.add_argument("--angular-atol", type=float, default=1e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--parallel-regions", action="store_true")
    parser.add_argument(
        "--reverse-chunk",
        type=int,
        default=8,
        help="output cotangents evaluated together in reverse mode",
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
        default=Path(__file__).resolve().parent / "limb_dark_outputs",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = resolved_config(args)
    if not 0.0 < args.u1 <= 1.0:
        raise ValueError("--u1 must lie in the interval (0, 1]")
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
    q, s, alpha, rho = 0.1, 1.1, np.deg2rad(50.0), 0.01
    params = jnp.asarray([t0, t_e, u0, q, s, alpha, rho])
    times = t0 + jnp.linspace(-0.5 * t_e, t_e, config["n_points"])
    model = make_model(
        times,
        u1=args.u1,
        n_limb=config["n_limb"],
        margin_r=args.margin_r,
        angular_atol=args.angular_atol,
        relative_tolerance=args.relative_tolerance,
        parallel_regions=args.parallel_regions,
    )

    value_function = jax.jit(model)
    forward_function = jax.jit(jax.jacfwd(model))
    reverse_chunk = min(args.reverse_chunk, config["n_points"])

    results = {}
    functions = [("value", value_function), ("forward", forward_function)]
    if args.with_reverse:
        functions.append(
            (
                "reverse",
                jax.jit(chunked_jacrev(model, config["n_points"], reverse_chunk)),
            )
        )
    for name, function in functions:
        print(f"benchmarking {name}...", flush=True)
        result, warmup, runs = benchmark(function, params, args.repeats)
        results[name] = (result, warmup, runs)
        print(
            f"  warm-up={warmup:.3f} s, median={np.median(runs):.3f} s",
            flush=True,
        )

    magnification = np.asarray(results["value"][0])
    forward = np.asarray(results["forward"][0])
    if not np.all(np.isfinite(magnification)):
        raise RuntimeError("magnification contains non-finite values")
    if not np.all(np.isfinite(forward)):
        raise RuntimeError("forward Jacobian contains non-finite values")
    reverse = None
    if args.with_reverse:
        reverse = np.asarray(results["reverse"][0])
        if not np.all(np.isfinite(reverse)):
            raise RuntimeError("reverse Jacobian contains non-finite values")
        np.testing.assert_allclose(
            forward,
            reverse,
            rtol=JACOBIAN_COMPARISON_RTOL,
            atol=JACOBIAN_COMPARISON_ATOL,
        )

    medians = {name: float(np.median(results[name][2])) for name in results}
    maximum_difference = float(np.max(np.abs(forward - reverse))) if reverse is not None else None
    report = {
        "device": str(devices[0]),
        "device_kind": getattr(devices[0], "device_kind", "unknown"),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "mag_binary_implementation": ("retry-free best-effort; limb-darkened G15/K31 fixed-1"),
        "limb_darkening_coefficient_u1": args.u1,
        "differentiated_parameter_names": list(PARAMETER_NAMES),
        "u1_is_static": True,
        "config": {
            **config,
            "margin_r": args.margin_r,
            "angular_atol": args.angular_atol,
            "relative_tolerance": args.relative_tolerance,
            "parallel_regions": args.parallel_regions,
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
        "jacobian_comparison_tolerances": (
            {
                "rtol": JACOBIAN_COMPARISON_RTOL,
                "atol": JACOBIAN_COMPARISON_ATOL,
            }
            if args.with_reverse
            else None
        ),
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
    if reverse is not None:
        np.save(args.output_dir / "jacobian_reverse.npy", reverse)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        save_jacobian_plot(
            args.output_dir / "binary_limb_dark_jacobian.png",
            np.asarray(times),
            magnification,
            forward,
            np.asarray(params),
        )
        if args.with_reverse:
            save_benchmark_plot(args.output_dir / "ad_benchmark.png", medians)

    print(f"device: {devices[0]}")
    print(f"u1: {args.u1:.6g} (static; not differentiated)")
    print(
        f"median of {args.repeats} compiled runs: "
        f"value={medians['value']:.3f} s, "
        f"forward={medians['forward']:.3f} s"
    )
    if args.with_reverse:
        print(f"reverse={medians['reverse']:.3f} s")
        print(f"reverse / forward: {report['reverse_over_forward']:.3f}x")
        print(f"max |J_forward - J_reverse|: {maximum_difference:.3e}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
