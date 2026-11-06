"""Small LD-only CPU benchmark with frozen value references and Jacobian timing."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from microjax.inverse_ray import mag_binary, mag_triple


def timed(function, arguments, repeats):
    # The first invocation compiles; only synchronized warm calls are timed.
    value = jax.block_until_ready(function(*arguments))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = jax.block_until_ready(function(*arguments))
        samples.append(time.perf_counter() - start)
    return np.asarray(value), float(np.median(samples) * 1000)


def record_values(report, name, function, arguments, reference, repeats):
    values, milliseconds = timed(function, arguments, repeats)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Nonfinite {name} values")
    error = float(np.max(np.abs(values / np.asarray(reference) - 1)))
    report["values"][name] = {
        "points": values.size,
        "warm_ms": milliseconds,
        "max_relative_reference_difference": error,
    }
    print(f"{name}: {milliseconds:.3f} ms; max relative difference {error:.3g}", flush=True)


def record_jacobian(report, name, model, arguments, repeats):
    # Independent AD accuracy belongs to the focused tests, not this timer.
    forward, milliseconds = timed(jax.jit(jax.jacfwd(model)), arguments, repeats)
    if not np.isfinite(forward).all():
        raise RuntimeError(f"Nonfinite {name}")
    report[name] = {"warm_ms": milliseconds, "check": "finiteness_only"}
    print(f"{name}: {milliseconds:.3f} ms; finiteness only, no accuracy comparison", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lens", choices=("binary", "triple", "all"), default="all")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "platform": platform.platform(),
            "device": str(jax.devices("cpu")[0]),
            "x64": True,
            "u1": 0.5,
            "repeats": args.repeats,
        },
        "values": {},
    }
    with jax.default_device(jax.devices("cpu")[0]):
        if args.lens in ("binary", "all"):
            cases = json.loads(Path(__file__).with_name("cpu_benchmark_cases.json").read_text())["cases"]
            evaluate = jax.jit(
                lambda sources, rho, s, q, coefficient: mag_binary(
                    sources, rho, s=s, q=q, u1=coefficient, backend="cpu"
                )
            )
            for case in cases:
                sources = jnp.asarray([complex(*xy) for xy in case["sources"]])
                arguments = (sources, case["rho"], case["s"], case["q"], 0.5)
                record_values(report, f"limb_dark/{case['name']}", evaluate, arguments, case["limb_dark"], args.repeats)

            point = jnp.asarray(
                [
                    -0.3421551581838465,
                    0.19397867617641237,
                    0.0048408299514288945,
                    0.8917065758514715,
                    0.028320791269324543,
                ]
            )

            def binary_model(p):
                return mag_binary(jnp.asarray([p[0] + 1j * p[1]]), p[2], s=p[3], q=p[4], u1=0.5, backend="cpu")[0]

            record_jacobian(report, "forward_jacobian", binary_model, (point,), args.repeats)

        if args.lens in ("triple", "all"):
            references = json.loads(Path(__file__).with_name("cpu_triple_cases.json").read_text())
            evaluate = jax.jit(
                lambda sources, rho, s, q, q3, r3, psi: mag_triple(
                    sources, rho, s=s, q=q, q3=q3, r3=r3, psi=psi, u1=0.5, backend="cpu"
                )
            )
            for case in references["cases"]:
                arguments = (jnp.asarray([complex(*case["source"])]),) + tuple(
                    case[key] for key in ("rho", "s", "q", "q3", "r3", "psi")
                )
                record_values(
                    report,
                    f"triple/limb_dark/{case['name']}",
                    evaluate,
                    arguments,
                    [case["magnification"]],
                    args.repeats,
                )

            reference = references["forward_jacobian"]

            def triple_model(p, time):
                t0, t_e, u0, q, s, alpha, rho, q3, r3, psi = p
                tau = (time - t0) / t_e
                source = (-u0 * jnp.sin(alpha) + tau * jnp.cos(alpha)) + 1j * (
                    u0 * jnp.cos(alpha) + tau * jnp.sin(alpha)
                )
                return mag_triple(jnp.asarray([source]), rho, s=s, q=q, q3=q3, r3=r3, psi=psi, u1=0.5, backend="cpu")[0]

            record_jacobian(
                report,
                "triple_forward_jacobian",
                triple_model,
                (jnp.asarray(reference["parameters"]), reference["observation_time"]),
                args.repeats,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(row["max_relative_reference_difference"] > 1e-3 for row in report["values"].values()):
        raise SystemExit("CPU value reference difference exceeds 1e-3")


if __name__ == "__main__":
    main()
