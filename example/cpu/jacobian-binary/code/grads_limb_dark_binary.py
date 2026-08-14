"""Forward-mode Jacobian of a limb-darkened binary light curve on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from grads_uniform_binary import run_example


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use 24 points for a smoke run")
    parser.add_argument("--u1", type=float, default=0.5)
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "limb_dark",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.u1 <= 1.0:
        raise ValueError("--u1 must lie in the interval (0, 1]")
    params = jnp.asarray([0.0, 10.0, 0.05, 1e-3, 1.0, np.deg2rad(50.0), 0.005])
    run_example(
        u1=args.u1,
        params=params,
        output_dir=args.output_dir,
        plot_name="binary_limb_dark_jacobian.png",
        args=args,
    )


if __name__ == "__main__":
    main()
