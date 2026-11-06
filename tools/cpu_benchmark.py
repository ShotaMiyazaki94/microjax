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

from microjax.inverse_ray import mag_binary


def timed(function, arguments, repeats):
    # The first invocation compiles; only synchronized warm calls are timed.
    value = jax.block_until_ready(function(*arguments))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = jax.block_until_ready(function(*arguments))
        samples.append(time.perf_counter() - start)
    return np.asarray(value), float(np.median(samples) * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cases = json.loads(Path(__file__).with_name("cpu_benchmark_cases.json").read_text())["cases"]
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
        evaluate = jax.jit(
            lambda sources, rho, s, q, coefficient: mag_binary(sources, rho, s=s, q=q, u1=coefficient, backend="cpu")
        )
        for case in cases:
            sources = jnp.asarray([complex(*xy) for xy in case["sources"]])
            arguments = (sources, case["rho"], case["s"], case["q"], 0.5)
            values, milliseconds = timed(evaluate, arguments, args.repeats)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite limb_dark/{case['name']} values")
            error = float(np.max(np.abs(values / np.asarray(case["limb_dark"]) - 1)))
            report["values"][f"limb_dark/{case['name']}"] = {
                "points": len(sources),
                "warm_ms": milliseconds,
                "max_relative_reference_difference": error,
            }
            print(f"limb_dark/{case['name']}: {milliseconds:.3f} ms; max relative difference {error:.3g}", flush=True)

        # Time one LD full-solve Jacobian for all five physical inputs. This
        # benchmark checks only finiteness, not independent derivative accuracy;
        # numerical AD contracts remain in tests/unittests/inverse_ray/test_cpu.py.
        point = jnp.asarray(
            [-0.3421551581838465, 0.19397867617641237, 0.0048408299514288945, 0.8917065758514715, 0.028320791269324543]
        )

        def model(p):
            return mag_binary(jnp.asarray([p[0] + 1j * p[1]]), p[2], s=p[3], q=p[4], u1=0.5, backend="cpu")[0]

        forward, milliseconds = timed(jax.jit(jax.jacfwd(model)), (point,), args.repeats)
        if not np.isfinite(forward).all():
            raise RuntimeError("Nonfinite forward Jacobian")
        report["forward_jacobian"] = {"warm_ms": milliseconds, "check": "finiteness_only"}
        print(f"LD forward Jacobian: {milliseconds:.3f} ms; finiteness only, no accuracy comparison", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(row["max_relative_reference_difference"] > 1e-3 for row in report["values"].values()):
        raise SystemExit("CPU value reference difference exceeds 1e-3")


if __name__ == "__main__":
    main()
