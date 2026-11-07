"""Benchmark every saved LD parameter configuration within a runtime budget.

Choose one caustic-frame source per configuration, covering every available
q/rho/magnification cell before balancing remaining source selections. Accuracy
and paired CPU/VBM timings use the same single-position evaluations.

Requires a separately supplied 12,325-by-36 reference CSV and same-stem JSON
metadata (u1=0.5). These local numerical inputs and generated results are not
distributed with the code. Runtime measurement additionally needs VBMicrolensing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q_EDGES = np.geomspace(1e-6, 1.0, 6)
RHO_EDGES = np.geomspace(3e-5, 1.0, 6)
MAG_EDGES = np.geomspace(1.0, 40000.0, 9)
SEED = 20261106


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "src/microjax").rglob("*.py"))}


def bin_index(value, edges):
    if not edges[0] <= value <= edges[-1]:
        raise ValueError(f"Value {value} outside fixed bin edges {edges}")
    return min(int(np.searchsorted(edges, value, side="right") - 1), len(edges) - 2)


def load_cases(reference_data):
    """Load the entire frame; select timing positions independently of errors."""
    cases = defaultdict(list)
    valid_points = missing_points = rechecked_points = 0
    with reference_data.open(newline="") as stream:
        for row in csv.DictReader(stream):
            q, rho = float(row["q"]), float(row["rho"])
            reference = float(row["accuracy_reference_magnification"] or "nan")
            valid = bool(np.isfinite(reference) and reference > 0)
            q_bin, rho_bin = bin_index(q, Q_EDGES), bin_index(rho, RHO_EDGES)
            cases[int(row["config_id"])].append(
                {
                    "profile": "ld",
                    **{
                        key: int(row[key])
                        for key in (
                            "config_id",
                            "dense_row",
                            "point_id",
                            "caustic_component",
                            "caustic_component_count",
                        )
                    },
                    **{key: float(row[key]) for key in ("q", "s", "rho", "x", "y")},
                    "reference": reference,
                    "reference_state": "saved" if valid else "unavailable",
                    "reference_rechecked": row["accuracy_reference_rechecked"] == "True",
                    "q_bin": q_bin,
                    "rho_bin": rho_bin,
                    "stratum_id": rho_bin * 5 + q_bin,
                    "mag_bin": bin_index(reference, MAG_EDGES) if valid else -1,
                }
            )
            valid_points += valid
            missing_points += not valid
            rechecked_points += row["accuracy_reference_rechecked"] == "True"
    if len(cases) != 12325 or any(len(rows) != 36 for rows in cases.values()):
        raise ValueError("Saved dense frame is not the expected 12,325 x 36 census")
    rng = np.random.default_rng(SEED)
    timing = {}
    coverage = defaultdict(lambda: {"source_points": 0, "reference_points": 0, "config_ids": set(), "timing_groups": 0})
    stratum_counts = Counter()
    timing_configs = 0
    for config_id, rows in sorted(cases.items()):
        by_magnification = defaultdict(list)
        stratum_counts[rows[0]["stratum_id"]] += 1
        for index, row in enumerate(rows):
            if row["mag_bin"] >= 0:
                by_magnification[row["mag_bin"]].append(index)
            key = (row["q_bin"], row["rho_bin"], row["mag_bin"])
            cell = coverage[key]
            cell["source_points"] += 1
            cell["reference_points"] += row["mag_bin"] >= 0
            cell["config_ids"].add(config_id)
        timing[config_id] = []
        for mag_bin, indices in sorted(by_magnification.items()):
            selected = sorted(rng.choice(indices, min(2, len(indices)), replace=False).tolist())
            timing[config_id].append((mag_bin, indices, selected))
            coverage[(rows[0]["q_bin"], rows[0]["rho_bin"], mag_bin)]["timing_groups"] += 1
        timing_configs += bool(by_magnification)
    # Cover every attainable q/rho/magnification cell before balancing the rest.
    candidates = defaultdict(list)
    for config_id, groups in timing.items():
        first = cases[config_id][0]
        for mag_bin, _, _ in groups:
            candidates[(first["stratum_id"], mag_bin)].append(config_id)
    assigned = {}
    counts = Counter()
    for cell in sorted(candidates, key=lambda key: (len(candidates[key]), key)):
        available = [config_id for config_id in candidates[cell] if config_id not in assigned]
        if not available:
            raise ValueError(f"No unused configuration covers cell {cell}")
        config_id = int(rng.choice(available))
        assigned[config_id] = cell[1]
        counts[cell] += 1
    order = sorted(cases)
    rng.shuffle(order)
    for config_id in order:
        if config_id in assigned or not timing[config_id]:
            continue
        stratum_id = cases[config_id][0]["stratum_id"]
        bins = [group[0] for group in timing[config_id]]
        minimum = min(counts[(stratum_id, mag_bin)] for mag_bin in bins)
        chosen = int(rng.choice([mag_bin for mag_bin in bins if counts[(stratum_id, mag_bin)] == minimum]))
        assigned[config_id] = chosen
        counts[(stratum_id, chosen)] += 1
    selected_cases = {}
    selected_timing = {}
    for config_id, rows in cases.items():
        if config_id in assigned:
            mag_bin = assigned[config_id]
            indices = [index for index, row in enumerate(rows) if row["mag_bin"] == mag_bin]
            selected = int(rng.choice(indices))
            selected_timing[config_id] = [(mag_bin, indices, [0])]
        else:
            selected = int(rng.integers(len(rows)))
            selected_timing[config_id] = []
        selected_cases[config_id] = [rows[selected]]
    timing = selected_timing
    groups = len(assigned)
    selected_points = len(selected_cases)
    sampling = {
        "seed": SEED,
        "design": "One source per every configuration; first reserve each attainable q/rho/magnification cell, then balance available magnification bins within q/rho strata; seeded random source within assigned bin",
        "target": "Saved caustic-centered source-position frame; one selected caustic component per configuration, not all source-plane positions or an observational population",
        "q_edges": Q_EDGES.tolist(),
        "rho_edges": RHO_EDGES.tolist(),
        "magnification_edges": MAG_EDGES.tolist(),
        "population_configs": len(cases),
        "selected_configs": len(cases),
        "source_positions": valid_points + missing_points,
        "selected_positions": selected_points,
        "valid_reference_positions": valid_points,
        "unavailable_reference_positions": missing_points,
        "reference_rechecked_positions": rechecked_points,
        "timing_configurations": timing_configs,
        "configurations_without_timing_reference": len(cases) - timing_configs,
        "timing_groups": groups,
        "timing_unique_positions": groups,
        "timing_evaluated_positions": groups,
        "occupied_reference_cells": len(candidates),
        "selected_reference_cells": len(counts),
        "selected_valid_reference_positions": len(assigned),
        "selected_unavailable_reference_positions": len(cases) - len(assigned),
        "source_positions_per_configuration": 1,
        "strata": [
            {"stratum_id": key, "q_bin": key % 5, "rho_bin": key // 5, "population_configs": count}
            for key, count in sorted(stratum_counts.items())
        ],
        "coverage": [
            {
                "q_bin": key[0],
                "rho_bin": key[1],
                "mag_bin": key[2],
                "configurations": len(cell["config_ids"]),
                "selected_timing_configurations": counts.get((key[1] * 5 + key[0], key[2]), 0),
                **{k: v for k, v in cell.items() if k != "config_ids"},
            }
            for key, cell in sorted(coverage.items())
        ],
        "empty_cell_meaning": "No saved eligible reference positions in this source frame; not a claim of physical impossibility",
    }
    return selected_cases, timing, order, sampling


def _vbm_worker(connection):
    import VBMicrolensing

    connection.send("ready")
    while True:
        task = connection.recv()
        if task is None:
            return
        points, s, q, rho, repeats = task
        try:
            solver = VBMicrolensing.VBMicrolensing()
            solver.Tol = solver.RelTol = 1e-4
            solver.a1 = 0.5

            def evaluate():
                return [solver.BinaryMag2(s, q, x, y, rho) for x, y in points]

            evaluate()
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                values = evaluate()
                samples.append(time.perf_counter() - start)
            connection.send(("ok", values, float(np.median(samples))))
        except Exception as error:
            connection.send(("error", repr(error), None))


class VbmWorker:
    """Persistent worker with bounded C++ calls; IPC is outside timing."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.process = self.connection = None

    def start(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.process = context.Process(target=_vbm_worker, args=(child,), daemon=True)
        self.process.start()
        child.close()
        self.connection = parent
        if not parent.poll(30) or parent.recv() != "ready":
            self.close()
            raise RuntimeError("VBMicrolensing worker did not initialize")

    def evaluate(self, task):
        if self.process is None:
            self.start()
        try:
            self.connection.send(task)
            if self.connection.poll(self.timeout):
                return self.connection.recv()
        except (EOFError, BrokenPipeError, OSError):
            self.close()
            return "error", None, None
        self.close()
        return "timeout", None, None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
            self.process.close()
            self.connection.close()
            self.process = self.connection = None


def write_metadata(path, metadata):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)


def main():
    budget_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-data",
        type=Path,
        required=True,
        help="Local dense LD reference CSV; a same-stem JSON metadata file is required",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/cpu/figures-benchmark/data")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--time-budget", type=float, default=300.0)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0 or args.time_budget <= 0:
        parser.error("repeats, timeout and time budget must be positive")
    reference_metadata_path = args.reference_data.with_suffix(".json")
    if not args.reference_data.is_file() or not reference_metadata_path.is_file():
        parser.error("--reference-data requires an existing CSV and same-stem JSON metadata; inputs are not bundled")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_path, timing_path, metadata_path = (
        args.output_dir / name for name in ("cpu.csv", "timing.csv", "metadata.json")
    )
    if any(path.exists() for path in (point_path, timing_path, metadata_path)):
        parser.error("output files already exist; choose a new output directory")
    reference_metadata = json.loads(reference_metadata_path.read_text())
    if reference_metadata["u1"] != 0.5:
        raise ValueError("Saved accuracy references do not describe LD u1=0.5")
    cases, timing, order, sampling = load_cases(args.reference_data)
    print(
        f"All {len(cases)} configurations / {sampling['selected_positions']} selected LD positions; {sampling['selected_reference_cells']} covered reference cells; budget={args.time_budget}s",
        flush=True,
    )
    sys.path.insert(0, str(ROOT / "src"))
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from microjax.inverse_ray.cpu.lightcurve import mag_binary_cpu_lightcurve

    hashes = source_hashes()
    metadata = {
        "state": "running",
        "measured_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "scope": "LD u1=0.5; full CPU solve without multipole shortcut; all saved parameter configurations, one coverage-balanced source each",
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "platform": platform.platform(),
            "device": str(jax.devices("cpu")[0]),
            "x64": True,
            "VBMicrolensing": version("VBMicrolensing"),
        },
        "timing": {
            "repeats": args.repeats,
            "statistic": "Median synchronized single-position time",
            "batch_capacity": 1,
            "padding": "None; identical single CPU/VBM source position",
            "execution_order": "Seeded random configuration order; balanced randomized CPU-first / VBM-first across timing groups",
            "excluded": "Compilation, initial warmup, VBM group warmup, worker startup and IPC",
            "cpu_warmup": "Compile and warm capacity1 once before all configurations; accuracy uses the same timed evaluations",
            "vbm_Tol": 1e-4,
            "vbm_RelTol": 1e-4,
            "vbm_timeout_seconds_per_group": args.timeout,
            "speed_eligibility": "CPU/VBM positive finite and both agree with saved reference within relative1e-3",
        },
        "sampling": sampling,
        "accuracy_reference": "Saved audited LD references with targeted rechecks; not universally 1e-8; separate from fresh timing VBM values",
        "accuracy_reference_provenance": {
            "metadata": reference_metadata_path.name,
            "metadata_sha256": digest(reference_metadata_path),
            "u1": reference_metadata["u1"],
            "description": reference_metadata["accuracy_reference"],
            "separate_uniform_columns": reference_metadata["uniform_values"],
        },
        "input": args.reference_data.name,
        "input_sha256": digest(args.reference_data),
        "source_sha256": hashes,
        "profiles": {
            "ld": {"coefficient": 0.5, "selected_points": sampling["selected_positions"], "configurations": len(cases)}
        },
        "time_budget_seconds": args.time_budget,
    }
    write_metadata(metadata_path, metadata)
    point_columns = list(cases[order[0]][0]) + ["magnification", "relative_error", "status"]
    timing_columns = [
        "profile",
        "timing_group_id",
        "config_id",
        "q",
        "s",
        "rho",
        "q_bin",
        "rho_bin",
        "stratum_id",
        "mag_bin",
        "population_points",
        "selected_points",
        "evaluated_points",
        "selected_dense_rows",
        "reference_magnifications",
        "cpu_magnifications",
        "vbm_magnifications",
        "cpu_seconds_per_point",
        "vbm_seconds_per_point",
        "vbm_relative_error_max",
        "speed_eligible",
        "vbm_state",
        "cpu_status_max",
        "measurement_order",
    ]
    orders = np.arange(len(cases)) % 2 == 0
    np.random.default_rng(SEED + 1).shuffle(orders)
    worker = VbmWorker(args.timeout)
    states = Counter()
    counters = Counter()
    maximum_error = 0.0
    first = cases[order[0]]
    with jax.default_device(jax.devices("cpu")[0]):
        for capacity in (1,):
            points = jnp.asarray([complex(row["x"], row["y"]) for row in first[:capacity]])
            jax.block_until_ready(
                mag_binary_cpu_lightcurve(points, first[0]["rho"], s=first[0]["s"], q=first[0]["q"], u1=0.5)
            )
    print("Compiled and warmed capacity1", flush=True)
    started = time.perf_counter()
    try:
        with (
            point_path.open("w", newline="") as point_output,
            timing_path.open("w", newline="") as timing_output,
            jax.default_device(jax.devices("cpu")[0]),
        ):
            point_writer = csv.DictWriter(point_output, fieldnames=point_columns)
            timing_writer = csv.DictWriter(timing_output, fieldnames=timing_columns)
            point_writer.writeheader()
            timing_writer.writeheader()
            completed_configs = 0
            for config_index, config_id in enumerate(order, start=1):
                if time.perf_counter() - budget_start >= args.time_budget - max(5.0, args.timeout):
                    print("Runtime budget reached", flush=True)
                    break
                row = cases[config_id][0]
                rho, s, q = (row[key] for key in ("rho", "s", "q"))
                local_jax = jnp.asarray([complex(row["x"], row["y"])])

                def cpu():
                    samples = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        value = jax.block_until_ready(mag_binary_cpu_lightcurve(local_jax, rho, s=s, q=q, u1=0.5))
                        samples.append(time.perf_counter() - start)
                    return value, float(np.median(samples))

                def vbm():
                    return worker.evaluate(([(row["x"], row["y"])], s, q, rho, args.repeats))

                vbm_first = bool(orders[config_index - 1])
                available = row["mag_bin"] >= 0
                if available and vbm_first:
                    state, values, elapsed = vbm()
                    result, cpu_time = cpu()
                else:
                    result, cpu_time = cpu()
                    state, values, elapsed = vbm() if available else ("unavailable_reference", None, None)
                magnification = float(np.asarray(result.magnification)[0])
                status = int(np.asarray(result.status)[0])
                error = abs(magnification / row["reference"] - 1) if available else np.nan
                point_writer.writerow(
                    {**row, "magnification": magnification, "relative_error": error, "status": status}
                )
                counters["accuracy_points"] += 1
                counters["cpu_nonfinite"] += not np.isfinite(magnification)
                counters["cpu_nonpositive"] += bool(np.isfinite(magnification) and magnification <= 0)
                counters["error_above_1e3"] += bool(np.isfinite(error) and error > 1e-3)
                if np.isfinite(error):
                    maximum_error = max(maximum_error, error)
                if available:
                    vbm_value = float(values[0]) if state == "ok" else np.nan
                    valid_vbm = bool(np.isfinite(vbm_value) and vbm_value > 0)
                    if state == "ok" and not valid_vbm:
                        state = "invalid"
                    vbm_error = abs(vbm_value / row["reference"] - 1) if valid_vbm else np.nan
                    eligible = bool(
                        state == "ok"
                        and vbm_error <= 1e-3
                        and np.isfinite(magnification)
                        and magnification > 0
                        and error <= 1e-3
                    )
                    mag_bin, population_indices, _ = timing[config_id][0]
                    timing_writer.writerow(
                        {
                            "profile": "ld",
                            "timing_group_id": f"{config_id}:{mag_bin}",
                            **{
                                key: row[key]
                                for key in ("config_id", "q", "s", "rho", "q_bin", "rho_bin", "stratum_id")
                            },
                            "mag_bin": mag_bin,
                            "population_points": len(population_indices),
                            "selected_points": 1,
                            "evaluated_points": 1,
                            "selected_dense_rows": json.dumps([row["dense_row"]]),
                            "reference_magnifications": json.dumps([row["reference"]]),
                            "cpu_magnifications": json.dumps([magnification]),
                            "vbm_magnifications": json.dumps([vbm_value]),
                            "cpu_seconds_per_point": cpu_time,
                            "vbm_seconds_per_point": elapsed if state == "ok" else np.nan,
                            "vbm_relative_error_max": vbm_error,
                            "speed_eligible": eligible,
                            "vbm_state": state,
                            "cpu_status_max": status,
                            "measurement_order": "vbm_first" if vbm_first else "cpu_first",
                        }
                    )
                    counters["speed_eligible_groups"] += eligible
                states[state] += 1
                completed_configs = config_index
                if config_index % 1000 == 0 or config_index == len(order):
                    point_output.flush()
                    timing_output.flush()
                    print(
                        f"{config_index}/{len(order)} configs; elapsed={time.perf_counter()-started:.1f}s; VBM={dict(states)}; accuracy>1e-3={counters['error_above_1e3']}",
                        flush=True,
                    )
        metadata.update(
            {
                "state": "complete" if completed_configs == len(order) else "stopped_by_runtime_budget",
                "completed_configurations": completed_configs,
                "total_runtime_seconds": time.perf_counter() - budget_start,
                "completed_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
                "elapsed_seconds_excluding_compilation": time.perf_counter() - started,
                "vbm_states": dict(states),
                "quality": {
                    **counters,
                    "maximum_relative_error": maximum_error,
                    "accuracy_and_timing_same_evaluations": True,
                },
                "source_unchanged_during_measurement": hashes == source_hashes(),
                "outputs": {"cpu.csv": digest(point_path), "timing.csv": digest(timing_path)},
            }
        )
        write_metadata(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "state": metadata["state"],
                    "elapsed_seconds": metadata["elapsed_seconds_excluding_compilation"],
                    "quality": metadata["quality"],
                    "vbm_states": metadata["vbm_states"],
                }
            ),
            flush=True,
        )
    finally:
        worker.close()


if __name__ == "__main__":
    main()
