#!/usr/bin/env python3
"""Benchmark the accelerator full solver on the CPU xdense parameter grid."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from microjax.inverse_ray import BinaryMagConfig  # noqa: E402
from microjax.inverse_ray.integrators.limb_dark import (  # noqa: E402
    mag_limb_dark_boundary,
)
from microjax.inverse_ray.integrators.uniform import (  # noqa: E402
    mag_uniform_boundary,
)
from microjax.inverse_ray.geometry.topology import (  # noqa: E402
    RADIAL_CAPACITY,
    RADIAL_INTERVAL_CAPACITY,
    RADIAL_TOPOLOGY,
)
from microjax.inverse_ray.roots.angular import (  # noqa: E402
    ANGULAR_CAPACITY,
    ANGULAR_DEGENERATE,
    ANGULAR_ROOT_FAILURE,
)


ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = ROOT.parent
OUTPUT_ROOT = ROOT / "outputs"
STRUCTURAL_STATUS = (
    ANGULAR_CAPACITY
    | ANGULAR_DEGENERATE
    | ANGULAR_ROOT_FAILURE
    | RADIAL_CAPACITY
    | RADIAL_TOPOLOGY
)
PUBLIC_BINARY_CONFIG = BinaryMagConfig()
FAST_RADIAL_INTERVAL_CAPACITY = PUBLIC_BINARY_CONFIG.radial_chunk_size


def _load_common_benchmark():
    """Load the local benchmark harness without importing its solver symbols."""

    path = REPOSITORY_ROOT / "benchmark_cpu" / "code" / "xdense_cpu.py"
    if not path.exists():
        raise FileNotFoundError(
            "benchmark_cpu/code/xdense_cpu.py is required for the shared grid "
            "and reporting harness"
        )
    spec = importlib.util.spec_from_file_location("microjax_xdense_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark harness from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMMON = _load_common_benchmark()


def parse_gpu_args(
    argv: Iterable[str] | None = None,
) -> tuple[argparse.Namespace, argparse.Namespace]:
    """Parse GPU-only options and delegate shared options to the CPU harness."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--n-limb", type=int, default=PUBLIC_BINARY_CONFIG.n_limb)
    parser.add_argument(
        "--radial-chunk-size",
        type=int,
        default=PUBLIC_BINARY_CONFIG.radial_chunk_size,
    )
    parser.add_argument("--radial-subdivisions", type=int, default=1)
    parser.add_argument(
        "--radial-order", type=int, choices=(19, 31, 47), default=19
    )
    parser.add_argument("--angular-profile-subdivisions", type=int, default=1)
    parser.add_argument("--robust-roots", action="store_true")
    parser.add_argument("--shallow-topology", action="store_true")
    parser.add_argument("--deep-topology", action="store_true")
    parser.add_argument(
        "--chart",
        choices=(
            "auto",
            "fast",
            "optimized",
            "global",
            "local",
            "cartesian",
            "hybrid",
            "cascade",
        ),
        default="fast",
        help=(
            "Image chart for binary full solves (fast uses the retry-free "
            "shallow polar route; optimized adds the low-q, low-rho "
            "polar/Cartesian cascade to auto)."
        ),
    )
    parser.add_argument("--configuration-batch-size", type=int, default=128)
    parser.add_argument("--cascade-configuration-batch-size", type=int, default=16)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--allow-non-gpu", action="store_true")
    parser.add_argument("--input-raw-status-zero", action="store_true")
    parser.add_argument("--input-q-max", type=float)
    parser.add_argument("--input-rho-max", type=float)
    parser.add_argument("--output-dir", type=Path)
    gpu_args, remaining = parser.parse_known_args(argv)
    shared_args = COMMON.parse_args(remaining)
    if gpu_args.n_limb < 64:
        parser.error("--n-limb must be at least 64")
    if gpu_args.radial_chunk_size <= 0:
        parser.error("--radial-chunk-size must be positive")
    if gpu_args.angular_profile_subdivisions <= 0:
        parser.error("--angular-profile-subdivisions must be positive")
    if not 1 <= gpu_args.radial_subdivisions <= 16:
        parser.error("--radial-subdivisions must be between 1 and 16")
    if gpu_args.configuration_batch_size <= 0:
        parser.error("--configuration-batch-size must be positive")
    if gpu_args.cascade_configuration_batch_size <= 0:
        parser.error("--cascade-configuration-batch-size must be positive")
    if gpu_args.device_index < 0:
        parser.error("--device-index must be non-negative")
    if gpu_args.shallow_topology and gpu_args.deep_topology:
        parser.error("--shallow-topology and --deep-topology are mutually exclusive")
    if gpu_args.chart == "fast":
        if gpu_args.deep_topology:
            parser.error("--chart fast cannot be combined with --deep-topology")
        # The dense audit found at most 37 active intervals. A 40-lane fast
        # buffer preserves one fully parallel radial launch without evaluating
        # the complete 64-slot general-purpose topology.
        gpu_args.shallow_topology = True
        gpu_args.radial_chunk_size = min(
            gpu_args.radial_chunk_size, FAST_RADIAL_INTERVAL_CAPACITY
        )
    preset_directory = "dense" if shared_args.preset == "xdense" else shared_args.preset
    shared_args.output_dir = (
        gpu_args.output_dir
        if gpu_args.output_dir is not None
        else OUTPUT_ROOT / preset_directory
    )
    if shared_args.refresh_timing_reference:
        parser.error(
            "refresh-timing-reference is unnecessary; reuse the CPU reference CSV"
        )
    return gpu_args, shared_args


def _boundary_kwargs(
    n_limb: int,
    radial_chunk_size: int,
    radial_subdivisions: int,
    robust_roots: bool,
    radial_order: int,
    deep_topology_sampling: bool,
    radial_interval_capacity: int,
) -> dict[str, object]:
    return {
        "Nlimb": n_limb,
        "margin_r": 0.5,
        "angular_atol": 1.0e-5,
        "relative_tolerance": 1.0e-4,
        "parallel_regions": False,
        "max_radial_subdivisions": radial_subdivisions,
        "robust_roots": robust_roots,
        "deep_topology_sampling": deep_topology_sampling,
        "certify_topology": False,
        "radial_strategy": "fixed",
        "radial_chunk_size": radial_chunk_size,
        "fixed_radial_order": radial_order,
        "return_info": True,
        "_radial_interval_capacity": radial_interval_capacity,
    }


def make_gpu_evaluator(
    *,
    n_limb: int,
    radial_chunk_size: int,
    radial_subdivisions: int,
    robust_roots: bool,
    radial_order: int,
    deep_topology_sampling: bool,
    angular_profile_subdivisions: int,
    u1: float,
    local_chart: bool,
    cartesian_chart: bool,
    hybrid_chart: bool,
    cascade_chart: bool,
    radial_interval_capacity: int,
) -> Callable:
    """Build one fixed-shape full-solve evaluator for a source batch."""

    common = _boundary_kwargs(
        n_limb,
        radial_chunk_size,
        radial_subdivisions,
        robust_roots,
        radial_order,
        deep_topology_sampling,
        radial_interval_capacity,
    )
    primary_common = dict(common, max_radial_subdivisions=1)

    if u1 == 0.0:
        if cartesian_chart or hybrid_chart or cascade_chart:
            raise ValueError("the Cartesian benchmark currently requires u1 > 0")

        def solve_polar(point, rho, s, q):
            return mag_uniform_boundary(
                point,
                rho,
                s=s,
                q=q,
                _planetary_local_chart=local_chart,
                **primary_common,
            )

    else:

        def solve_polar(point, rho, s, q):
            return mag_limb_dark_boundary(
                point,
                rho,
                s=s,
                q=q,
                u1=u1,
                angular_profile_subdivisions=angular_profile_subdivisions,
                _planetary_local_chart=local_chart,
                **primary_common,
            )

        def solve_polar_refined(point, rho, s, q):
            return mag_limb_dark_boundary(
                point,
                rho,
                s=s,
                q=q,
                u1=u1,
                angular_profile_subdivisions=angular_profile_subdivisions,
                _planetary_local_chart=local_chart,
                **common,
            )

        def solve_cartesian(point, rho, s, q):
            return mag_limb_dark_boundary(
                point,
                rho,
                s=s,
                q=q,
                u1=u1,
                angular_profile_subdivisions=angular_profile_subdivisions,
                _planetary_cartesian_chart=True,
                **primary_common,
            )

    def evaluate_configuration(points, rho, s, q):
        if cascade_chart:
            primary = jax.vmap(lambda point: solve_polar(point, rho, s, q))(points)
            fallback_capacity = min(16, points.shape[0])
            failed = primary.status != 0
            failed_count = jnp.sum(failed, dtype=jnp.int32)
            indices = jnp.nonzero(
                failed, size=fallback_capacity, fill_value=points.shape[0]
            )[0]
            safe_indices = jnp.minimum(indices, points.shape[0] - 1)
            selected_points = points[safe_indices]
            cartesian = jax.vmap(lambda point: solve_cartesian(point, rho, s, q))(
                selected_points
            )
            refined = jax.vmap(lambda point: solve_polar_refined(point, rho, s, q))(
                selected_points
            )
            chart_disagreement = jnp.abs(
                cartesian.magnification - refined.magnification
            ) / jnp.maximum(
                jnp.abs(refined.magnification),
                jnp.finfo(refined.magnification.dtype).tiny,
            )
            use_cartesian = (cartesian.status == 0) & (chart_disagreement <= 5.0e-3)
            rescued = jax.tree_util.tree_map(
                lambda cart, polar: jnp.where(use_cartesian, cart, polar),
                cartesian,
                refined,
            )
            active = jnp.arange(fallback_capacity, dtype=jnp.int32) < failed_count

            def scatter(primary_field, rescued_field, fill_value):
                extended = jnp.concatenate(
                    (
                        primary_field,
                        jnp.asarray([fill_value], dtype=primary_field.dtype),
                    )
                )
                replacements = jnp.where(active, rescued_field, fill_value)
                return extended.at[indices].set(replacements)[:-1]

            return primary._replace(
                magnification=scatter(
                    primary.magnification, rescued.magnification, 0.0
                ),
                estimated_error=scatter(
                    primary.estimated_error, rescued.estimated_error, jnp.nan
                ),
                status=scatter(primary.status, rescued.status, jnp.int32(0)),
            )
        if not cartesian_chart and not hybrid_chart:
            return jax.vmap(lambda point: solve_polar(point, rho, s, q))(points)
        cartesian = jax.vmap(lambda point: solve_cartesian(point, rho, s, q))(points)
        if not hybrid_chart:
            return cartesian

        # Sixteen lanes cover the observed low-q structural-failure tail while
        # configurations are batched independently.  With cb=16 this still
        # exposes 256 polar fallback points to the GPU in one regular launch.
        fallback_capacity = min(16, points.shape[0])
        failed = cartesian.status != 0
        failed_count = jnp.sum(failed, dtype=jnp.int32)
        indices = jnp.nonzero(
            failed, size=fallback_capacity, fill_value=points.shape[0]
        )[0]
        safe_indices = jnp.minimum(indices, points.shape[0] - 1)
        polar = jax.vmap(lambda point: solve_polar(point, rho, s, q))(
            points[safe_indices]
        )
        active = jnp.arange(fallback_capacity, dtype=jnp.int32) < failed_count

        def scatter(cartesian_field, polar_field, fill_value):
            extended = jnp.concatenate(
                (
                    cartesian_field,
                    jnp.asarray([fill_value], dtype=cartesian_field.dtype),
                )
            )
            replacements = jnp.where(active, polar_field, fill_value)
            return extended.at[indices].set(replacements)[:-1]

        magnification = scatter(cartesian.magnification, polar.magnification, 0.0)
        estimated_error = scatter(
            cartesian.estimated_error, polar.estimated_error, jnp.nan
        )
        status = scatter(cartesian.status, polar.status, jnp.int32(0))
        return cartesian._replace(
            magnification=magnification,
            estimated_error=estimated_error,
            status=status,
        )

    if cascade_chart:

        def evaluate_cascade_batch(points, rho, s, q):
            primary = jax.vmap(
                lambda config_points, config_rho, config_s, config_q: jax.vmap(
                    lambda point: solve_polar(point, config_rho, config_s, config_q)
                )(config_points)
            )(points, rho, s, q)
            batch_size, point_count = points.shape
            flat_count = batch_size * point_count
            fallback_capacity = min(256, flat_count)
            flat_failed = primary.status.reshape(-1) != 0
            failed_count = jnp.sum(flat_failed, dtype=jnp.int32)
            indices = jnp.nonzero(
                flat_failed, size=fallback_capacity, fill_value=flat_count
            )[0]
            safe_indices = jnp.minimum(indices, flat_count - 1)
            config_indices = safe_indices // point_count
            selected_points = points.reshape(-1)[safe_indices]
            cartesian = jax.vmap(solve_cartesian)(
                selected_points,
                rho[config_indices],
                s[config_indices],
                q[config_indices],
            )
            selected_primary = jax.tree_util.tree_map(
                lambda field: field.reshape(-1)[safe_indices], primary
            )
            chart_disagreement = jnp.abs(
                cartesian.magnification - selected_primary.magnification
            ) / jnp.maximum(
                jnp.abs(selected_primary.magnification),
                jnp.finfo(selected_primary.magnification.dtype).tiny,
            )
            use_cartesian = (cartesian.status == 0) & (chart_disagreement <= 5.0e-3)
            active = jnp.arange(fallback_capacity, dtype=jnp.int32) < failed_count
            needs_refined = active & ~use_cartesian
            # The dense low-q audit peaks at 143 rejected Cartesian lanes per
            # 16-configuration launch.  A 160-lane buffer retains headroom
            # without paying the refined polar cost for all 256 candidates.
            refined_capacity = min(160, fallback_capacity)
            refined_count = jnp.sum(needs_refined, dtype=jnp.int32)
            refined_indices = jnp.nonzero(
                needs_refined,
                size=refined_capacity,
                fill_value=fallback_capacity,
            )[0]
            safe_refined_indices = jnp.minimum(refined_indices, fallback_capacity - 1)
            refined_config_indices = config_indices[safe_refined_indices]
            refined = jax.vmap(solve_polar_refined)(
                selected_points[safe_refined_indices],
                rho[refined_config_indices],
                s[refined_config_indices],
                q[refined_config_indices],
            )
            refined_active = (
                jnp.arange(refined_capacity, dtype=jnp.int32) < refined_count
            )

            def scatter_refined(cartesian_field, refined_field, fill_value):
                extended = jnp.concatenate(
                    (
                        cartesian_field,
                        jnp.asarray([fill_value], dtype=cartesian_field.dtype),
                    )
                )
                replacements = jnp.where(refined_active, refined_field, fill_value)
                return extended.at[refined_indices].set(replacements)[:-1]

            rescued = cartesian._replace(
                magnification=scatter_refined(
                    cartesian.magnification, refined.magnification, 0.0
                ),
                estimated_error=scatter_refined(
                    cartesian.estimated_error, refined.estimated_error, jnp.nan
                ),
                status=scatter_refined(cartesian.status, refined.status, jnp.int32(0)),
            )

            def scatter(primary_field, rescued_field, fill_value):
                flat = primary_field.reshape(-1)
                extended = jnp.concatenate(
                    (flat, jnp.asarray([fill_value], dtype=flat.dtype))
                )
                replacements = jnp.where(active, rescued_field, fill_value)
                return (
                    extended.at[indices]
                    .set(replacements)[:-1]
                    .reshape(primary_field.shape)
                )

            return primary._replace(
                magnification=scatter(
                    primary.magnification, rescued.magnification, 0.0
                ),
                estimated_error=scatter(
                    primary.estimated_error, rescued.estimated_error, jnp.nan
                ),
                status=scatter(primary.status, rescued.status, jnp.int32(0)),
            )

        return jax.jit(evaluate_cascade_batch)
    return jax.jit(jax.vmap(evaluate_configuration))


def evaluate_configuration_batch(
    groups: list[list[dict[str, object]]],
    *,
    evaluator: Callable,
    repeats: int,
    warmup: bool,
    tier: int,
) -> list[dict[str, object]]:
    """Evaluate and time several fixed-lens configurations in one launch."""

    points = jnp.asarray(
        [
            [complex(float(row["x"]), float(row["y"])) for row in rows]
            for rows in groups
        ],
        dtype=jnp.complex128,
    )
    rho = jnp.asarray([float(rows[0]["rho"]) for rows in groups], dtype=jnp.float64)
    s = jnp.asarray([float(rows[0]["s"]) for rows in groups], dtype=jnp.float64)
    q = jnp.asarray([float(rows[0]["q"]) for rows in groups], dtype=jnp.float64)

    if warmup:
        jax.block_until_ready(evaluator(points, rho, s, q).magnification)
    durations: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = evaluator(points, rho, s, q)
        jax.block_until_ready(result.magnification)
        durations.append(time.perf_counter() - start)
    assert result is not None
    values = np.asarray(result.magnification)
    raw_statuses = np.asarray(result.status)
    statuses = np.bitwise_and(raw_statuses, STRUCTURAL_STATUS)
    seconds_per_point = statistics.median(durations) / points.size
    output = []
    for group_index, rows in enumerate(groups):
        for point_index, row in enumerate(rows):
            item = dict(row)
            value = float(values[group_index, point_index])
            item.update(
                microjax_magnification=value,
                microjax_tier=tier,
                microjax_status=int(statuses[group_index, point_index]),
                microjax_raw_status=int(raw_statuses[group_index, point_index]),
                microjax_seconds_per_point=seconds_per_point,
            )
            if "reference_magnification" in row:
                reference = float(row["reference_magnification"])
                item["relative_error"] = (
                    abs(value - reference) / abs(reference)
                    if np.isfinite(value)
                    and np.isfinite(reference)
                    and reference != 0.0
                    else np.nan
                )
            output.append(item)
    return output


def write_gpu_missed_configurations(
    rows: list[dict[str, object]],
    data_directory: Path,
) -> dict[str, object]:
    """Write the shared miss schema with GPU-specific filenames."""

    microjax_misses, vbm_misses = COMMON.classify_missed_points(rows)
    source_columns = list(rows[0]) if rows else []
    microjax_path = data_directory / "xdense_gpu_microjax_misses.csv"
    vbm_path = data_directory / "xdense_gpu_vbm_misses.csv"
    configurations_path = data_directory / "xdense_gpu_missed_configurations.json"
    COMMON._write_miss_csv(microjax_path, microjax_misses, source_columns, ())
    COMMON._write_miss_csv(
        vbm_path,
        vbm_misses,
        source_columns,
        ("vbm_cross_tolerance_relative_difference",),
    )
    microjax_configurations = COMMON._configuration_records(microjax_misses)
    vbm_configurations = COMMON._configuration_records(vbm_misses)
    payload = {
        "criteria": {
            "microjax": [
                "nonzero structural status",
                "non-finite or non-positive magnification",
                f"relative error above {COMMON.TARGET_RELATIVE_ERROR:.0e}",
            ],
            "vbm": [
                "strict or timing-reference timeout",
                "strict or timing-reference non-finite/non-positive magnification",
                "strict/timing-reference relative disagreement above "
                f"{COMMON.VBM_DISAGREEMENT_RELATIVE_ERROR:.0e}",
            ],
        },
        "microjax": microjax_configurations,
        "vbm": vbm_configurations,
    }
    configurations_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "microjax": {
            "points": len(microjax_misses),
            "configurations": len(microjax_configurations),
            "csv": microjax_path.name,
        },
        "vbm": {
            "points": len(vbm_misses),
            "configurations": len(vbm_configurations),
            "csv": vbm_path.name,
        },
        "configurations_json": configurations_path.name,
    }


def summarize_diagnostic_valid(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize accuracy where every boundary diagnostic accepted the value."""

    selected = [
        row
        for row in rows
        if int(row["microjax_raw_status"]) == 0
        and np.isfinite(float(row.get("relative_error", np.nan)))
    ]
    errors = np.asarray([float(row["relative_error"]) for row in selected], dtype=float)
    if errors.size == 0:
        return {"points": 0, "fraction": 0.0}
    return {
        "points": int(errors.size),
        "fraction": float(errors.size / len(rows)),
        "p50": float(np.quantile(errors, 0.50)),
        "p95": float(np.quantile(errors, 0.95)),
        "p99": float(np.quantile(errors, 0.99)),
        "max": float(np.max(errors)),
        "above_1e-3": int(np.sum(errors > 1.0e-3)),
        "above_1e-2": int(np.sum(errors > 1.0e-2)),
    }


def _existing_cpu_reference(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], Path] | None:
    """Load the matching CPU benchmark CSV as the immutable VBM reference."""

    preset_directory = "dense" if args.preset == "xdense" else args.preset
    data_directory = (
        REPOSITORY_ROOT / "benchmark_cpu" / "outputs" / preset_directory / "data"
    )
    csv_path = data_directory / "xdense_cpu.csv"
    json_path = data_directory / "xdense_cpu.json"
    if not csv_path.exists() or not json_path.exists():
        return None
    report = json.loads(json_path.read_text(encoding="utf-8"))
    expected = {
        "q_grid": args.q_grid,
        "s_grid": args.s_grid,
        "rho_grid": args.rho_grid,
        "points_per_configuration": args.points_per_configuration,
        "u1": args.u1,
    }
    if any(report.get(name) != value for name, value in expected.items()):
        return None
    with csv_path.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    required_reference = {
        "reference_magnification",
        "reference_seconds_per_point",
        "reference_timed_out",
        "timing_reference_magnification",
        "timing_reference_seconds_per_point",
        "timing_reference_timed_out",
    }
    if not source or not required_reference.issubset(source[0]):
        return None
    rows = []
    for source_row in source:
        row = {
            name: value
            for name, value in source_row.items()
            if not name.startswith("microjax_") and name != "relative_error"
        }
        row.update(
            config_id=int(source_row["config_id"]),
            point_id=int(source_row["point_id"]),
            q=float(source_row["q"]),
            s=float(source_row["s"]),
            rho=float(source_row["rho"]),
            x=float(source_row["x"]),
            y=float(source_row["y"]),
            reference_magnification=float(source_row["reference_magnification"]),
        )
        rows.append(row)
    return rows, csv_path


def _generate_rows(
    gpu_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], Path | None]:
    if args.input is not None:
        rows = COMMON.read_input(args.input, args.reference_column)
        with args.input.open(newline="", encoding="utf-8") as stream:
            source_rows = list(csv.DictReader(stream))
        selected_sources = source_rows
        if (
            args.input_error_above is not None
            or gpu_args.input_raw_status_zero
            or gpu_args.input_q_max is not None
            or gpu_args.input_rho_max is not None
        ):
            selected = []
            for source in source_rows:
                error_selected = args.input_error_above is None or (
                    source.get("relative_error", "") != ""
                    and float(source["relative_error"]) > args.input_error_above
                )
                status_selected = not gpu_args.input_raw_status_zero or (
                    source.get("microjax_raw_status", "") != ""
                    and int(source["microjax_raw_status"]) == 0
                )
                q_selected = (
                    gpu_args.input_q_max is None
                    or float(source["q"]) <= gpu_args.input_q_max
                )
                rho_selected = (
                    gpu_args.input_rho_max is None
                    or float(source["rho"]) <= gpu_args.input_rho_max
                )
                selected.append(
                    error_selected and status_selected and q_selected and rho_selected
                )
            rows = [row for row, keep in zip(rows, selected, strict=True) if keep]
            selected_sources = [
                row for row, keep in zip(source_rows, selected, strict=True) if keep
            ]
        for row, source in zip(rows, selected_sources, strict=True):
            for name, value in source.items():
                if (
                    name not in row
                    and not name.startswith("microjax_")
                    and name != "relative_error"
                ):
                    row[name] = value
        return rows, args.input
    if args.reference == "vbm":
        existing = _existing_cpu_reference(args)
        if existing is not None:
            return existing
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    caustic_cache = {}
    for config in COMMON.configurations(args):
        key = (config.q, config.s, args.caustic_points)
        if key not in caustic_cache:
            caustic_cache[key] = COMMON.caustic_segment_components(
                config.q,
                config.s,
                args.caustic_points,
            )
        rows.extend(
            COMMON.sample_points(
                config,
                count=args.points_per_configuration,
                maximum_distance=args.d_over_rho_max,
                caustic_points=args.caustic_points,
                caustic_sampling=args.caustic_sampling,
                rng=rng,
                components=caustic_cache[key],
            )
        )
    return rows, None


def main(argv: Iterable[str] | None = None) -> int:
    wall_start = time.perf_counter()
    gpu_args, args = parse_gpu_args(argv)
    devices = jax.devices()
    if gpu_args.device_index >= len(devices):
        raise ValueError(
            f"device index {gpu_args.device_index} is unavailable; devices={devices}"
        )
    device = devices[gpu_args.device_index]
    if device.platform != "gpu" and not gpu_args.allow_non_gpu:
        raise RuntimeError(
            f"CUDA GPU required, found {device}; pass --allow-non-gpu only for smoke tests"
        )

    if args.plot_only:
        data_directory = args.output_dir / "data"
        figure_directory = args.output_dir / "figures"
        csv_path = data_directory / "xdense_gpu.csv"
        json_path = data_directory / "xdense_gpu.json"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("benchmark output contains no points")
        report = (
            json.loads(json_path.read_text(encoding="utf-8"))
            if json_path.exists()
            else {}
        )
        # Replayed benchmarks intentionally replace the old microJAX columns,
        # but older output files did not retain the independent VBM timing
        # columns.  Recover them for plots from the recorded input CSV without
        # rerunning either numerical solver.
        reference_path_text = report.get("reference_csv")
        if (
            "reference_seconds_per_point" not in rows[0]
            and reference_path_text is not None
        ):
            reference_path = Path(reference_path_text)
            if reference_path.exists():
                with reference_path.open(newline="", encoding="utf-8") as stream:
                    reference_rows = list(csv.DictReader(stream))
                if len(reference_rows) == len(rows):
                    metadata_columns = (
                        "reference_seconds_per_point",
                        "reference_timed_out",
                        "timing_reference_magnification",
                        "timing_reference_seconds_per_point",
                        "timing_reference_timed_out",
                    )
                    for row, reference_row in zip(rows, reference_rows, strict=True):
                        same_point = all(
                            row[name] == reference_row[name]
                            for name in ("config_id", "point_id")
                        )
                        if not same_point:
                            raise ValueError(
                                "plot reference rows do not match output IDs"
                            )
                        for name in metadata_columns:
                            if name in reference_row:
                                row[name] = reference_row[name]
        figure_directory.mkdir(parents=True, exist_ok=True)
        COMMON.plot_results(
            rows,
            figure_directory / "xdense_gpu.png",
            solver_label="microJAX GPU fixed full solve",
        )
        COMMON.plot_speed_accuracy(
            rows,
            figure_directory / "xdense_gpu_parameter_maps.png",
            solver_label="GPU full-solve",
        )
        COMMON.plot_speed_accuracy(
            rows,
            figure_directory / "xdense_gpu_parameter_maps_vbm_reltol_1e-3.png",
            speed_reference_prefix="timing_reference",
            speed_reference_label=r"VBM $\mathrm{RelTol}=10^{-3}$",
            solver_label="GPU full-solve",
        )
        report.update(COMMON.summarize(rows))
        report["missed_configurations"] = write_gpu_missed_configurations(
            rows,
            data_directory,
        )
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"overview: {figure_directory / 'xdense_gpu.png'}")
        print(f"parameter maps: {figure_directory / 'xdense_gpu_parameter_maps.png'}")
        return 0

    rows, reference_csv = _generate_rows(gpu_args, args)
    if not rows:
        raise ValueError("benchmark contains no points")
    groups = COMMON.group_rows(rows)
    if args.reference == "vbm" and reference_csv is None:
        COMMON.evaluate_vbm_reference(
            groups,
            u1=args.u1,
            absolute_tolerance=args.reference_absolute_tolerance,
            relative_tolerance=args.reference_relative_tolerance,
            timeout=args.reference_timeout,
            workers=args.reference_workers,
            progress_label=(
                "VBM accuracy reference "
                f"(RelTol={args.reference_relative_tolerance:.0e})"
            ),
        )
        COMMON.evaluate_vbm_reference(
            groups,
            u1=args.u1,
            absolute_tolerance=args.reference_absolute_tolerance,
            relative_tolerance=args.timing_reference_relative_tolerance,
            timeout=args.reference_timeout,
            workers=args.reference_workers,
            column_prefix="timing_reference",
            progress_label=(
                "VBM timing reference "
                f"(RelTol={args.timing_reference_relative_tolerance:.0e})"
            ),
        )

    evaluators: dict[tuple[int, bool, bool, bool, bool], Callable] = {}
    evaluated: list[dict[str, object]] = []
    completed_configurations = 0
    steady_seconds = 0.0
    if gpu_args.chart == "fast":
        routed_groups = (
            (True, False, False, False, groups),
        )
    elif gpu_args.chart == "auto":
        routed_groups = (
            (
                True,
                False,
                False,
                False,
                [group for group in groups if float(group[0]["q"]) < 1.0e-2],
            ),
            (
                False,
                False,
                False,
                False,
                [group for group in groups if float(group[0]["q"]) >= 1.0e-2],
            ),
        )
    elif gpu_args.chart == "optimized":

        def low_q_low_rho(group):
            return float(group[0]["q"]) <= 1.0e-3 and float(group[0]["rho"]) <= 1.0e-3

        routed_groups = (
            (
                True,
                False,
                False,
                True,
                [group for group in groups if low_q_low_rho(group)],
            ),
            (
                True,
                False,
                False,
                False,
                [
                    group
                    for group in groups
                    if not low_q_low_rho(group) and float(group[0]["q"]) < 1.0e-2
                ],
            ),
            (
                False,
                False,
                False,
                False,
                [group for group in groups if float(group[0]["q"]) >= 1.0e-2],
            ),
        )
    else:
        routed_groups = (
            (
                gpu_args.chart in ("local", "hybrid", "cascade"),
                gpu_args.chart == "cartesian",
                gpu_args.chart == "hybrid",
                gpu_args.chart == "cascade",
                groups,
            ),
        )
    for (
        local_chart,
        cartesian_chart,
        hybrid_chart,
        cascade_chart,
        route_groups,
    ) in routed_groups:
        if not route_groups:
            continue
        route_batch_size = (
            gpu_args.cascade_configuration_batch_size
            if cascade_chart
            else gpu_args.configuration_batch_size
        )
        for start in range(0, len(route_groups), route_batch_size):
            active_groups = route_groups[start : start + route_batch_size]
            point_count = len(active_groups[0])
            if any(len(group) != point_count for group in active_groups):
                raise ValueError(
                    "configuration batching requires an equal point count per configuration"
                )
            padded_groups = active_groups + [active_groups[0]] * (
                route_batch_size - len(active_groups)
            )
            evaluator_key = (
                point_count,
                local_chart,
                cartesian_chart,
                hybrid_chart,
                cascade_chart,
            )
            if evaluator_key not in evaluators:
                route_radial_chunk_size = gpu_args.radial_chunk_size
                evaluators[evaluator_key] = make_gpu_evaluator(
                    n_limb=gpu_args.n_limb,
                    radial_chunk_size=route_radial_chunk_size,
                    radial_subdivisions=gpu_args.radial_subdivisions,
                    robust_roots=gpu_args.robust_roots,
                    radial_order=gpu_args.radial_order,
                    deep_topology_sampling=(
                        gpu_args.deep_topology
                        or (gpu_args.robust_roots and not gpu_args.shallow_topology)
                    ),
                    angular_profile_subdivisions=gpu_args.angular_profile_subdivisions,
                    u1=args.u1,
                    local_chart=local_chart,
                    cartesian_chart=cartesian_chart,
                    hybrid_chart=hybrid_chart,
                    cascade_chart=cascade_chart,
                    radial_interval_capacity=(
                        FAST_RADIAL_INTERVAL_CAPACITY
                        if gpu_args.chart == "fast"
                        else RADIAL_INTERVAL_CAPACITY
                    ),
                )
            batch_values = evaluate_configuration_batch(
                padded_groups,
                evaluator=evaluators[evaluator_key],
                repeats=args.timing_repeats,
                warmup=start == 0,
                tier=gpu_args.n_limb,
            )
            steady_seconds += (
                float(batch_values[0]["microjax_seconds_per_point"])
                * route_batch_size
                * point_count
            )
            evaluated.extend(batch_values[: sum(len(group) for group in active_groups)])
            completed_configurations += len(active_groups)
            if completed_configurations % 100 == 0 or completed_configurations == len(
                groups
            ):
                print(
                    f"[{completed_configurations}/{len(groups)}] configurations",
                    flush=True,
                )

    data_directory = args.output_dir / "data"
    figure_directory = args.output_dir / "figures"
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    csv_path = data_directory / "xdense_gpu.csv"
    json_path = data_directory / "xdense_gpu.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluated[0]))
        writer.writeheader()
        writer.writerows(evaluated)

    report = {
        "solver": "microjax GPU fixed full solve",
        "device": str(device),
        "device_kind": getattr(device, "device_kind", None),
        "preset": args.preset,
        "q_grid": args.q_grid,
        "s_grid": args.s_grid,
        "rho_grid": args.rho_grid,
        "points_per_configuration": args.points_per_configuration,
        "u1": args.u1,
        "n_limb": gpu_args.n_limb,
        "radial_chunk_size": gpu_args.radial_chunk_size,
        "radial_chunk_size_by_route": (
            {"local": gpu_args.radial_chunk_size}
            if gpu_args.chart == "fast"
            else None
        ),
        "radial_interval_capacity": (
            FAST_RADIAL_INTERVAL_CAPACITY
            if gpu_args.chart == "fast"
            else RADIAL_INTERVAL_CAPACITY
        ),
        "radial_subdivisions": gpu_args.radial_subdivisions,
        "radial_order": gpu_args.radial_order,
        "angular_profile_subdivisions": gpu_args.angular_profile_subdivisions,
        "robust_roots": gpu_args.robust_roots,
        "deep_topology_sampling": (
            gpu_args.deep_topology
            or (gpu_args.robust_roots and not gpu_args.shallow_topology)
        ),
        "chart": gpu_args.chart,
        "configuration_batch_size": gpu_args.configuration_batch_size,
        "cascade_configuration_batch_size": (gpu_args.cascade_configuration_batch_size),
        "timing_repeats": args.timing_repeats,
        "reference_absolute_tolerance": args.reference_absolute_tolerance,
        "reference_relative_tolerance": args.reference_relative_tolerance,
        "timing_reference_relative_tolerance": args.timing_reference_relative_tolerance,
        "jit_warmup_excluded": True,
        "reference_csv": str(reference_csv) if reference_csv is not None else None,
        **COMMON.summarize(evaluated),
        "aggregate_seconds_per_point": steady_seconds / len(evaluated),
        "throughput_points_per_second": len(evaluated) / steady_seconds,
    }
    report["missed_configurations"] = write_gpu_missed_configurations(
        evaluated,
        data_directory,
    )
    report["diagnostic_valid_relative_error"] = summarize_diagnostic_valid(evaluated)
    report["wall_seconds"] = time.perf_counter() - wall_start
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_plot:
        COMMON.plot_results(
            evaluated,
            figure_directory / "xdense_gpu.png",
            solver_label="microJAX GPU fixed full solve",
        )
        COMMON.plot_speed_accuracy(
            evaluated,
            figure_directory / "xdense_gpu_parameter_maps.png",
            solver_label="GPU full-solve",
        )
        COMMON.plot_speed_accuracy(
            evaluated,
            figure_directory / "xdense_gpu_parameter_maps_vbm_reltol_1e-3.png",
            speed_reference_prefix="timing_reference",
            speed_reference_label=r"VBM $\mathrm{RelTol}=10^{-3}$",
            solver_label="GPU full-solve",
        )
    report["wall_seconds"] = time.perf_counter() - wall_start
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
