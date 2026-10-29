"""Run the compact CPU value checks without regenerating dense references.

The frozen cases are a regression screen, not a population error estimate or
an accuracy certificate. Unresolved references are reported separately.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/data/cpu_accuracy"
PROFILES = ("uniform", "ld")


def reference_number(value):
    # Archived CSVs encode an absent reference as an empty field.
    return float(value) if str(value).strip() else float("nan")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    source = ROOT / "src/microjax"
    return {str(path.relative_to(source)): digest(path) for path in sorted(source.rglob("*.py"))}


def read_cases(path):
    with Path(path).open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    required = {"dense_row", "config_id", "cohort", "x", "y", "rho", "s", "q"}
    for profile in PROFILES:
        required.update((f"baseline_{profile}", f"reference_{profile}", f"{profile}_reference_state"))
    if not cases or not required.issubset(cases[0]):
        raise ValueError(f"Invalid case table: {path}")
    validate_cases(cases)
    return cases


def validate_cases(cases):
    identifiers = [int(row["dense_row"]) for row in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Case table contains duplicate dense_row identifiers")
    configurations = {}
    for row in cases:
        physical = {name: float(row[name]) for name in ("x", "y", "rho", "s", "q")}
        if not all(np.isfinite(value) for value in physical.values()):
            raise ValueError(f"Nonfinite physical input at row {row['dense_row']}")
        if any(physical[name] <= 0 for name in ("rho", "s", "q")):
            raise ValueError(f"Nonpositive physical scale at row {row['dense_row']}")
        parameters = tuple(physical[name] for name in ("rho", "s", "q"))
        previous = configurations.setdefault(row["config_id"], parameters)
        if previous != parameters:
            raise ValueError(f"Inconsistent lens parameters within configuration {row['config_id']}")
        for profile in PROFILES:
            baseline = float(row[f"baseline_{profile}"])
            reference = reference_number(row[f"reference_{profile}"])
            state = row[f"{profile}_reference_state"]
            if not np.isfinite(baseline) or baseline <= 0:
                raise ValueError(f"Invalid saved baseline at row {row['dense_row']}")
            if state not in ("saved", "checked", "unresolved", "missing"):
                raise ValueError(f"Unknown reference state {state!r} at row {row['dense_row']}")
            valid_reference = np.isfinite(reference) and reference > 0
            if (state == "missing") == valid_reference:
                raise ValueError(f"Reference state disagrees with value at row {row['dense_row']}")


def assess(cases, values, profile, tolerance=1e-3, baseline_rtol=1e-10):
    """Separate reference regressions from changes needing reference review."""
    if len(cases) != len(values):
        raise ValueError("Solver result count does not match case count")
    records = []
    for row, value in zip(cases, values):
        actual, tier, status, nodes = map(float, value)
        baseline = float(row[f"baseline_{profile}"])
        reference = reference_number(row[f"reference_{profile}"])
        state = row[f"{profile}_reference_state"]
        valid_reference = np.isfinite(reference) and reference > 0
        comparable = valid_reference and state in ("saved", "checked")
        finite = np.isfinite(actual) and actual > 0
        error = abs(actual / reference - 1) if valid_reference and finite else float("inf")
        old_error = abs(baseline / reference - 1) if valid_reference else float("inf")
        change = abs(actual / baseline - 1) if finite else float("inf")
        raw_miss = valid_reference and error > tolerance
        new_miss = comparable and old_error <= tolerance and error > tolerance
        # A previously bad point must not silently get worse. Ignore changes
        # below 1e-8 in relative reference error, well below the 1e-3 threshold.
        worsened = comparable and old_error > tolerance and error > old_error + 1e-8
        review = not comparable and change > baseline_rtol
        records.append(
            dict(
                dense_row=int(row["dense_row"]),
                cohort=row["cohort"],
                profile=profile,
                magnification=actual,
                baseline=baseline,
                reference=reference,
                reference_state=state,
                comparable=bool(comparable),
                relative_reference_difference=error if valid_reference else "",
                baseline_reference_difference=old_error if valid_reference else "",
                relative_baseline_change=change,
                tier=int(tier),
                status=int(status),
                nodes=int(nodes),
                invalid_value=not bool(finite),
                raw_reference_miss=bool(raw_miss),
                new_reference_miss=bool(new_miss),
                known_miss_worsened=bool(worsened),
                unresolved_change_needs_review=bool(review),
            )
        )
    comparable = [row for row in records if row["comparable"]]
    raw = [row for row in records if row["relative_reference_difference"] != ""]

    def maximum(rows, key):
        numbers = [row[key] for row in rows if np.isfinite(row[key])]
        return max(numbers) if numbers else None

    summary = dict(
        points=len(records),
        comparable_saved_references=len(comparable),
        unresolved_references=sum(row["reference_state"] == "unresolved" for row in records),
        missing_references=sum(row["reference_state"] == "missing" for row in records),
        invalid_values=sum(row["invalid_value"] for row in records),
        raw_reference_misses=sum(row["raw_reference_miss"] for row in raw),
        comparable_reference_misses=sum(row["raw_reference_miss"] for row in comparable),
        new_reference_misses=sum(row["new_reference_miss"] for row in records),
        known_misses_worsened=sum(row["known_miss_worsened"] for row in records),
        unresolved_changes_needing_review=sum(row["unresolved_change_needs_review"] for row in records),
        maximum_comparable_reference_difference=maximum(comparable, "relative_reference_difference"),
        maximum_raw_reference_difference=maximum(raw, "relative_reference_difference"),
        maximum_relative_baseline_change=maximum(records, "relative_baseline_change"),
        nonzero_status=sum(row["status"] != 0 for row in records),
    )
    summary["regression_gate_passed"] = not any(
        summary[key]
        for key in (
            "invalid_values",
            "new_reference_misses",
            "known_misses_worsened",
            "unresolved_changes_needing_review",
        )
    )
    return summary, records


def replay_history(manifest, cases):
    """Replay saved rejected candidates; this does not execute an old solver."""
    by_id = {int(row["dense_row"]): row for row in cases}
    included = set(by_id)
    reports = []
    for block in manifest["historical_regressions"]:
        records = block["records"]
        crossings = [row for row in records if row["candidate_error"] > 1e-3 and row["baseline_error"] <= 1e-3]
        caught = [row for row in crossings if int(row["dense_row"]) in included]
        detected = 0
        for record in caught:
            row = by_id[int(record["dense_row"])]
            value = [record["candidate_value"], row["baseline_tier"], row["baseline_status"], row["baseline_nodes"]]
            summary, _ = assess([row], [value], block["profile"])
            detected += not summary["regression_gate_passed"]
        reports.append(
            dict(
                candidate=block["candidate"],
                profile=block["profile"],
                path=block["path"],
                historical_crossings=len(crossings),
                included_crossings=len(caught),
                all_crossings_included=len(caught) == len(crossings),
                detected_crossings=detected,
                all_crossings_rejected=detected == len(crossings),
            )
        )
    missing = sorted(
        {int(row["dense_row"]) for block in manifest["historical_regressions"] for row in block["records"]} - included
    )
    crossing_count = sum(row["historical_crossings"] for row in reports)
    return dict(
        mode="saved historical outcomes; no numerical solver execution",
        scope="Only historical candidates and failures recorded in the manifest; not unknown-failure coverage",
        candidate_reports=reports,
        historical_crossings=crossing_count,
        missing_dense_rows=missing,
        detected_crossings=sum(row["detected_crossings"] for row in reports),
        regression_gate_passed=crossing_count > 0
        and not missing
        and all(row["all_crossings_rejected"] for row in reports),
    )


def evaluate(cases, batch_size):
    # Delay importing JAX until after --replay/--help, which need no compiler.
    sys.path.insert(0, str(ROOT / "src"))
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from microjax.inverse_ray.cpu.lightcurve import mag_binary_cpu_one_shot_lightcurve

    groups = defaultdict(list)
    for index, row in enumerate(cases):
        groups[row["config_id"]].append(index)
    evaluated_points = sum((len(indices) + batch_size - 1) // batch_size * batch_size for indices in groups.values())
    values = {}
    with jax.default_device(jax.devices("cpu")[0]):
        for profile, u1 in (("uniform", 0.0), ("ld", 0.5)):
            output = np.empty((len(cases), 4))
            done = 0
            for group_index, indices in enumerate(groups.values()):
                row = cases[indices[0]]
                # Keep shared rho/s/q outside the map, as in the original
                # measurement. A size-one scan can compile differently; use
                # at least two positions and discard padded outputs.
                for start in range(0, len(indices), batch_size):
                    selected = indices[start : start + batch_size]
                    points = np.asarray([complex(float(cases[i]["x"]), float(cases[i]["y"])) for i in selected])
                    points = np.pad(points, (0, batch_size - len(points)), mode="edge")
                    result = jax.device_get(
                        mag_binary_cpu_one_shot_lightcurve(
                            jnp.asarray(points), float(row["rho"]), s=float(row["s"]), q=float(row["q"]), u1=u1
                        )
                    )
                    output[selected] = np.column_stack(
                        (result.magnification, result.tier, result.status, result.n_radial_nodes)
                    )[: len(selected)]
                    done += len(selected)
                if group_index == 0 or (group_index + 1) % 128 == 0 or done == len(cases):
                    print(f"{profile}: {done}/{len(cases)}", flush=True)
            values[profile] = output
    return values, dict(
        jax=jax.__version__,
        jaxlib=jax.lib.__version__,
        device=str(jax.devices("cpu")[0]),
        configuration_groups=len(groups),
        batch_capacity=batch_size,
        evaluated_positions_per_profile_including_padding=evaluated_points,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("difficult", "survey", "compact", "holdout", "full"), default="compact")
    parser.add_argument("--cases", type=Path, default=FIXTURES / "cases.csv")
    parser.add_argument("--manifest", type=Path, default=FIXTURES / "manifest.json")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / "benchmark_cpu/outputs",
        help="Local archived evidence used only by --suite full",
    )
    parser.add_argument("--replay", action="store_true", help="Check compact coverage of saved historical failures")
    parser.add_argument("--batch", type=int, default=2, help="Positions per same-configuration call; at least two")
    parser.add_argument("--max-points", type=int, help="Partial smoke run; does not validate the complete suite")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; existing results are never overwritten"
    )
    args = parser.parse_args()
    if args.batch < 2 or (args.max_points is not None and args.max_points < 1):
        parser.error("batch must be at least two and max-points must be positive")
    if args.replay and args.suite != "compact":
        parser.error("--replay uses --suite compact")
    input_paths = [args.cases, args.manifest]
    if args.suite == "full":
        input_paths.extend(
            args.evidence_root / "unified_profile_20261014" / name
            for name in ("joint112/values.csv", "uniform_reference_crosscheck112.csv")
        )
    input_hashes = {str(path.resolve()): digest(path) for path in input_paths}
    manifest = json.loads(args.manifest.read_text())
    if digest(args.cases) != manifest["cases_sha256"]:
        parser.error("Case table hash differs from its frozen manifest; regenerate both together")
    cases = read_cases(args.cases)
    if args.suite == "full":
        from cpu_accuracy_cases import load_full_table

        cases = load_full_table(ROOT, evidence_root=args.evidence_root)
        validate_cases(cases)
    else:
        cohorts = ("difficult", "survey") if args.suite == "compact" else (args.suite,)
        cases = [row for row in cases if row["cohort"] in cohorts]
    complete_count = len(cases)
    if args.max_points:
        cases = cases[: args.max_points]
    if not cases:
        parser.error("No cases selected")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.replay:
        report = replay_history(manifest, cases)
    else:
        initial = source_hashes()
        start = time.perf_counter()
        values, environment = evaluate(cases, args.batch)
        all_records = []
        summaries = {}
        for profile in PROFILES:
            summaries[profile], records = assess(cases, values[profile], profile)
            all_records.extend(records)
        with (args.output_dir / "points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
            writer.writeheader()
            writer.writerows(all_records)
        changed = source_hashes() != initial
        report = dict(
            profiles=summaries,
            environment=environment,
            source_sha256=initial,
            source_changed_during_run=changed,
            elapsed_seconds=time.perf_counter() - start,
            scope="CPU full-solve value regression screen at u=0 and u=0.5; no multipole shortcut or derivative test",
            interpretation="Saved reference differences, not certified errors or a population failure rate; unresolved/missing references are not passes",
            tolerance=1e-3,
            unresolved_baseline_rtol=1e-10,
            known_miss_error_slack=1e-8,
            regression_gate_passed=not changed and all(part["regression_gate_passed"] for part in summaries.values()),
        )
        # These are overlapping views of the same results, never pooled rates.
        report["cohorts"] = {}
        for cohort in sorted({row["cohort"] for row in cases}):
            indices = [i for i, row in enumerate(cases) if row["cohort"] == cohort]
            report["cohorts"][cohort] = {
                profile: assess([cases[i] for i in indices], values[profile][indices], profile)[0]
                for profile in PROFILES
            }
    inputs_changed = any(digest(path) != expected for path, expected in input_hashes.items())
    report["regression_gate_passed"] = report["regression_gate_passed"] and not inputs_changed
    report.update(
        suite=args.suite,
        points=len(cases),
        complete_suite_points=complete_count,
        partial_run=len(cases) != complete_count,
        measured_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        cases_sha256=digest(args.cases),
        manifest_sha256=digest(args.manifest),
        driver_sha256=digest(__file__),
        input_sha256=input_hashes,
        inputs_changed_during_run=inputs_changed,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("source_sha256", "cohorts", "candidate_reports")
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if report["regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
