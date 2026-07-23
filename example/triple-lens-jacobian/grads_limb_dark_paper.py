"""Triple-lens limb-darkened magnification and forward Jacobian benchmark.

This companion to ``grads_uniform_paper.py`` uses the same triple-lens
trajectory and differentiates the light curve with respect to the same ten
trajectory, source-size, and lens parameters.  The linear limb-darkening
coefficient ``u1`` is deliberately static: it selects the compiled
limb-darkened integration kernel and is not an eleventh AD parameter.

Only forward-mode AD is benchmarked.  This matches the intended use of the
current triple-lens solver and avoids constructing the very expensive reverse
graph for the boundary and nested profile quadratures.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from grads_uniform_paper import PARAMETER_NAMES, benchmark, make_model, resolved_config, save_jacobian_plot

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use a small configuration for a smoke benchmark")
    parser.add_argument("--u1", type=float, default=0.5)
    parser.add_argument("--n-points", type=int)
    parser.add_argument("--n-limb", type=int)
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
        u1=args.u1,
        n_limb=config["n_limb"],
    )

    functions = {"value": jax.jit(model), "forward": jax.jit(jax.jacfwd(model))}
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
        "mag_triple_implementation": "retry-free best-effort; limb-darkened G15/K31 fixed-1; mixed charts",
        "forward_ad": "forward mode through boundary roots and nested limb-darkened profile quadrature",
        "limb_darkening_coefficient_u1": args.u1,
        "differentiated_parameter_names": list(PARAMETER_NAMES),
        "u1_is_static": True,
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
            args.output_dir / "triple_limb_dark_jacobian.png",
            np.asarray(times),
            magnification,
            forward,
            np.asarray(params),
            source_label=rf"Linear limb darkening: $u_1={args.u1:g}$ (static)",
        )

    print(f"device: {devices[0]}")
    print(f"u1: {args.u1:.6g} (static; not differentiated)")
    print(
        f"median of {args.repeats} compiled runs: "
        f"value={medians['value']:.3f} s, forward={medians['forward']:.3f} s"
    )
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
