#!/usr/bin/env python3
"""Freeze a small CPU accuracy fixture from the locally saved benchmark evidence.

Generation reads the ignored benchmark_cpu directory; validation of the generated
fixture does not. This script runs no numerical solver and needs only Python's
standard library. Saved references retain their existing limitations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 1e-3
SURVEY_POINTS = 384
HOLDOUT_CONFIGS = 32
HOLDOUT_POINTS_PER_CONFIG = 16
SEED = "microjax-cpu-accuracy-v1-20261029"
FIELDS = [
    "dense_row",
    "config_id",
    "point_id",
    "x",
    "y",
    "rho",
    "s",
    "q",
    "baseline_uniform",
    "baseline_ld",
    "reference_uniform",
    "reference_ld",
    "baseline_tier",
    "baseline_status",
    "baseline_nodes",
    "cohort",
    "reasons",
    "uniform_reference_state",
    "ld_reference_state",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def evidence_path(root: Path, evidence_root: Path | None) -> Path:
    return evidence_root if evidence_root is not None else root / "benchmark_cpu/outputs"


def load_evidence(root: Path, evidence_root: Path | None = None):
    base = evidence_path(root, evidence_root)
    checkpoint = base / "unified_profile_20261014"
    rows = read_csv(checkpoint / "joint112/values.csv")
    audit = {int(row["dense_row"]): row for row in read_csv(checkpoint / "uniform_reference_crosscheck112.csv")}
    return rows, audit


def normalized_row(row, audit, cohort="full", reasons="saved_dense_grid"):
    row_id = int(row["dense_row"])
    checked = audit.get(row_id)
    uniform_reference = row["uniform_reference"]
    uniform_state = "saved"
    if checked:
        uniform_state = "checked" if checked["reference_checked"] == "True" else "unresolved"
        if uniform_state == "checked":
            # The original 1e-7 reference at row 317044 is known to be wrong.
            # Only this independent, cross-engine/stability-checked value replaces it.
            uniform_reference = checked["vbbl_1e9"]
    ld_reference = row["accuracy_reference_magnification"]
    return {
        **{name: row[name] for name in ("dense_row", "config_id", "point_id", "x", "y", "rho", "s", "q")},
        "baseline_uniform": row["uniform"],
        "baseline_ld": row["ld"],
        "reference_uniform": uniform_reference,
        "reference_ld": ld_reference,
        "baseline_tier": row["tier"],
        "baseline_status": row["status"],
        "baseline_nodes": row["nodes"],
        "cohort": cohort,
        "reasons": reasons,
        "uniform_reference_state": uniform_state,
        "ld_reference_state": "saved" if positive_finite(ld_reference) else "missing",
    }


def load_full_table(root: Path, evidence_root: Path | None = None) -> list[dict[str, str]]:
    """Return normalized saved dense rows, including the audited reference flags.

    This is for an explicitly requested full audit. The compact checker should
    read its tracked cases.csv directly instead of importing historical evidence.
    Numeric CSV strings are preserved without a float parse/write round trip.
    """
    rows, audit = load_evidence(root, evidence_root)
    return [normalized_row(row, audit) for row in rows]


def positive_finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def relative_error(value, reference) -> float:
    if not positive_finite(reference):
        return math.nan
    return abs(float(value) / float(reference) - 1)


def finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def historical_record(row, candidate_value, reference, baseline, candidate_error=None, baseline_error=None):
    candidate_error = relative_error(candidate_value, reference) if candidate_error is None else float(candidate_error)
    baseline_error = relative_error(baseline, reference) if baseline_error is None else float(baseline_error)
    if not (math.isfinite(candidate_error) and math.isfinite(baseline_error)):
        return None
    if candidate_error <= THRESHOLD or baseline_error > THRESHOLD:
        return None
    return {
        "dense_row": int(row["dense_row"]),
        "candidate_value": float(candidate_value),
        "reference": float(reference),
        "baseline": float(baseline),
        "candidate_error": candidate_error,
        "baseline_error": baseline_error,
    }


def collect_history(root: Path, base: Path):
    """Collect saved newly failing cases without rerunning rejected candidates."""
    blocks = []
    input_paths = set()

    def add(path, candidate, profile, rows, extract):
        input_paths.add(path)
        records = []
        for row in rows:
            record = historical_record(row, *extract(row))
            if record is not None:
                records.append(record)
        blocks.append(
            {
                "path": display_path(path, root),
                "sha256": sha256(path),
                "candidate": candidate,
                "profile": profile,
                "evaluated_points": len(rows),
                "records": sorted(records, key=lambda item: item["dense_row"]),
            }
        )

    # Every available regression CSV is inventoried, including empty outcomes.
    for path in sorted(base.rglob("*regressions.csv")):
        rows = read_csv(path)
        candidate = str(path.relative_to(base).with_suffix(""))
        if path.parent.name == "public_groups":
            label = path.name.removesuffix("_regressions.csv")
            add(
                path,
                candidate,
                "ld",
                rows,
                lambda row, label=label: (
                    row[f"{label}_magnification"],
                    row["accuracy_reference_magnification"],
                    row["current_magnification"],
                    row[f"{label}_relative_error"],
                    row["current_relative_error"],
                ),
            )
        elif "uniform" in str(path) or path.name == "screen_regressions.csv":
            if path.name == "screen_regressions.csv":
                add(
                    path,
                    candidate,
                    "uniform",
                    rows,
                    lambda row: (
                        row["unified_magnification"],
                        row["reference_magnification"],
                        row["current_magnification"],
                        row["unified_error"],
                        row["current_error"],
                    ),
                )
            elif path.parent.name == "joint112":
                add(
                    path,
                    candidate,
                    "uniform",
                    rows,
                    lambda row: (
                        row["uniform"],
                        row["uniform_reference"],
                        row["uniform_baseline"],
                        row["error"],
                        row["baseline_error"],
                    ),
                )
            else:
                add(
                    path,
                    candidate,
                    "uniform",
                    rows,
                    lambda row: (
                        row["magnification"],
                        row["reference"],
                        row["baseline"],
                        row["error"],
                        row["baseline_error"],
                    ),
                )
        elif path.parent.name == "joint112":
            add(
                path,
                candidate,
                "ld",
                rows,
                lambda row: (
                    row["ld"],
                    row["accuracy_reference_magnification"],
                    row["microjax_magnification"],
                    row["error"],
                    row["baseline_error"],
                ),
            )
        else:
            add(
                path,
                candidate,
                "ld",
                rows,
                lambda row: (
                    row["candidate_magnification"],
                    row["accuracy_reference_magnification"],
                    row["microjax_magnification"],
                    row["candidate_error"],
                    row["baseline_error"],
                ),
            )

    checkpoint = base / "unified_profile_20261014"
    path = base / "chart_reduction_20261014/uniform_screen/values.csv"
    rows = read_csv(path)
    for label in ("two_chart", "theta_only"):
        add(
            path,
            f"uniform_screen/{label}",
            "uniform",
            rows,
            lambda row, label=label: (
                row[f"{label}_magnification"],
                row["reference_magnification"],
                row["current_magnification"],
                row[f"{label}_error"],
                row["current_error"],
            ),
        )

    path = checkpoint / "trace_values.csv"
    by_variant = defaultdict(list)
    for row in read_csv(path):
        # These two LD outputs share a table whose reference column is uniform.
        # Comparing them to that column would manufacture thousands of failures.
        if row["solver"].startswith("unified") or row["solver"] == "uniform_old56":
            by_variant[row["solver"]].append(row)
    screen = {int(row["dense_row"]): row for row in read_csv(checkpoint / "screen.csv")}
    input_paths.add(checkpoint / "screen.csv")
    for label, rows in sorted(by_variant.items()):
        add(
            path,
            f"trace/{label}",
            "uniform",
            rows,
            lambda row: (
                row["value"],
                row["reference_magnification"],
                screen[int(row["dense_row"])]["current_magnification"],
                None,
                row["current_error"],
            ),
        )

    for filename in ("newton12_values.csv", "support_values.csv", "trace64_newton12_screen.csv"):
        path = checkpoint / filename
        by_variant = defaultdict(list)
        for row in read_csv(path):
            by_variant[row.get("variant", "trace64_newton12")].append(row)
        for label, rows in sorted(by_variant.items()):
            add(
                path,
                f"support_trace/{label}",
                "uniform",
                rows,
                lambda row: (
                    row["value"],
                    row["reference_magnification"],
                    row["current_magnification"],
                    row["error"],
                    row["current_error"],
                ),
            )

    path = base / "simplify_cpu_20261014/screen/values.csv"
    by_variant = defaultdict(list)
    for row in read_csv(path):
        by_variant[(row["variant"], row["profile"])].append(row)
    for (label, profile), rows in sorted(by_variant.items()):
        add(path, f"cache_screen/{label}", profile, rows, lambda row: (row["value"], row["reference"], row["baseline"]))
    return blocks, input_paths


def rank(purpose, value):
    return hashlib.sha256(f"{SEED}:{purpose}:{value}".encode()).digest()


def select_rows(rows, blocks):
    by_id = {int(row["dense_row"]): row for row in rows}
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["config_id"])].append(row)
    historical_ids = {record["dense_row"] for block in blocks for record in block["records"]}
    reasons = defaultdict(set)
    for row_id in historical_ids:
        reasons[row_id].add("historical_regression")
    for row in rows:
        row_id = int(row["dense_row"])
        for profile, value, reference in (
            ("uniform", row["uniform"], row["uniform_reference"]),
            ("ld", row["ld"], row["accuracy_reference_magnification"]),
        ):
            if relative_error(value, reference) > THRESHOLD:
                reasons[row_id].add(f"current_{profile}_raw_discrepancy")
    hard_ids = set(reasons)
    for row_id in sorted(hard_ids):
        row = by_id[row_id]
        neighbors = sorted(
            (
                math.hypot(float(other["x"]) - float(row["x"]), float(other["y"]) - float(row["y"])),
                int(other["dense_row"]),
            )
            for other in groups[int(row["config_id"])]
            if int(other["dense_row"]) != row_id
        )[:2]
        for _, neighbor_id in neighbors:
            reasons[neighbor_id].add("near_historical_or_current_discrepancy")
    difficult_ids = set(reasons)
    difficult_configs = {int(by_id[row_id]["config_id"]) for row_id in difficult_ids}

    # Reserve entire configurations before choosing the survey. Neither this
    # ranking nor the survey ranking uses candidate/reference errors or values.
    holdout_configs = sorted(set(groups) - difficult_configs, key=lambda value: rank("holdout_config", value))[
        :HOLDOUT_CONFIGS
    ]
    holdout_ids = set()
    for config in holdout_configs:
        ordered = sorted(groups[config], key=lambda row: rank("holdout_point", row["dense_row"]))
        holdout_ids.update(int(row["dense_row"]) for row in ordered[:HOLDOUT_POINTS_PER_CONFIG])

    remaining = [row for row in rows if int(row["config_id"]) not in difficult_configs | set(holdout_configs)]
    levels = {name: sorted({row[name] for row in rows}, key=float) for name in ("q", "s", "rho")}
    bins = {
        name: {value: index * 5 // len(values) for index, value in enumerate(values)} for name, values in levels.items()
    }

    def tokens(row):
        coarse = tuple(bins[name][row[name]] for name in ("q", "s", "rho")) + (row["tier"],)
        return {("coarse", coarse), ("q", row["q"]), ("s", row["s"]), ("rho", row["rho"]), ("route", row["tier"])}

    ordered = sorted(remaining, key=lambda row: rank("survey", row["dense_row"]))
    all_tokens = set().union(*(tokens(row) for row in remaining))
    seen, survey_ids, survey_configs = set(), set(), set()
    for row in ordered:
        config = int(row["config_id"])
        row_tokens = tokens(row)
        if config not in survey_configs and not row_tokens <= seen:
            survey_ids.add(int(row["dense_row"]))
            survey_configs.add(config)
            seen.update(row_tokens)
        if seen == all_tokens:
            break
    if seen != all_tokens or len(survey_ids) > SURVEY_POINTS:
        raise ValueError("Survey budget cannot cover the requested saved-grid strata")
    for row in ordered:
        if len(survey_ids) == SURVEY_POINTS:
            break
        config = int(row["config_id"])
        if config not in survey_configs:
            survey_ids.add(int(row["dense_row"]))
            survey_configs.add(config)
    if len(survey_ids) != SURVEY_POINTS or len(holdout_ids) != HOLDOUT_CONFIGS * HOLDOUT_POINTS_PER_CONFIG:
        raise ValueError("Insufficient disjoint configurations for survey/holdout")
    cohorts = {row_id: "difficult" for row_id in difficult_ids}
    cohorts.update({row_id: "survey" for row_id in survey_ids})
    cohorts.update({row_id: "holdout" for row_id in holdout_ids})
    for row_id in survey_ids:
        reasons[row_id].add("stratified_saved_grid_coverage")
    for row_id in holdout_ids:
        reasons[row_id].add("fixed_disjoint_config_holdout")
    assert historical_ids <= set(cohorts)
    assert not (set(holdout_configs) & (difficult_configs | survey_configs))
    return cohorts, reasons, historical_ids, hard_ids, holdout_configs


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path, output: Path, evidence_root: Path | None = None):
    base = evidence_path(root, evidence_root)
    rows, audit = load_evidence(root, evidence_root)
    blocks, input_paths = collect_history(root, base)
    cohorts, reasons, historical_ids, hard_ids, holdout_configs = select_rows(rows, blocks)
    compact = [
        normalized_row(row, audit, cohorts[int(row["dense_row"])], ";".join(sorted(reasons[int(row["dense_row"])])))
        for row in rows
        if int(row["dense_row"]) in cohorts
    ]
    checkpoint = base / "simplify_cpu_20261014/evidence_index.json"
    source_hashes = json.loads(checkpoint.read_text())["production_source_sha256"]
    changed = [name for name, expected in source_hashes.items() if sha256(root / "src/microjax" / name) != expected]
    if changed:
        raise ValueError(f"Saved baseline source differs from current source: {changed}")
    input_paths.update(
        {
            checkpoint,
            base / "unified_profile_20261014/joint112/values.csv",
            base / "unified_profile_20261014/uniform_reference_crosscheck112.csv",
        }
    )
    write_csv(output / "cases.csv", compact)
    selected_ids = {int(row["dense_row"]) for row in compact}
    records = [record for block in blocks for record in block["records"]]
    summaries = {}
    for cohort in ("difficult", "survey", "holdout"):
        selected = [row for row in compact if row["cohort"] == cohort]
        summaries[cohort] = {
            "rows": len(selected),
            "configurations": len({row["config_id"] for row in selected}),
            "routes": dict(sorted(Counter(row["baseline_tier"] for row in selected).items())),
            "q_levels": len({row["q"] for row in selected}),
            "s_levels": len({row["s"] for row in selected}),
            "rho_levels": len({row["rho"] for row in selected}),
            "uniform_reference_states": dict(
                sorted(Counter(row["uniform_reference_state"] for row in selected).items())
            ),
            "ld_reference_states": dict(sorted(Counter(row["ld_reference_state"] for row in selected).items())),
        }
    reference_audit = [
        {
            "dense_row": row_id,
            "state": "checked" if row["reference_checked"] == "True" else "unresolved",
            "raw_reference": finite_or_none(row["uniform_reference"]),
            "vbm_1e9": finite_or_none(row["vbm_1e9"]),
            "vbbl_1e9": finite_or_none(row["vbbl_1e9"]),
            "vbbl_1e7": finite_or_none(row["vbbl_1e7"]),
            "reference_agreement": finite_or_none(row["reference_agreement"]),
            "reference_stability": finite_or_none(row["reference_stability"]),
        }
        for row_id, row in sorted(audit.items())
    ]
    manifest = {
        "format_version": 1,
        "threshold": THRESHOLD,
        "columns": FIELDS,
        "cases_sha256": sha256(output / "cases.csv"),
        "generator_sha256": sha256(Path(__file__)),
        "rows": len(compact),
        "full_grid_rows": len(rows),
        "cohorts": summaries,
        "selection": {
            "seed": SEED,
            "historical_and_current_hard_rows": len(hard_ids),
            "neighbors": "Two nearest saved x/y positions within the same configuration per hard row; distance ties use dense_row.",
            "survey": "384 rows from distinct other configurations; covers every available exact q/s/rho level and 5x5x5-by-route strata, filled by stable hash ranking.",
            "holdout": "32 configurations reserved before survey selection; 16 hash-selected positions each; no configuration overlaps with difficult or survey.",
            "holdout_config_ids": sorted(holdout_configs),
            "scope": "Frozen coverage samples of the existing caustic-centered grid; not a population error estimate or a new physical domain.",
        },
        "input_sha256": {display_path(path, root): sha256(path) for path in sorted(input_paths)},
        "source_sha256": source_hashes,
        "historical_regressions": blocks,
        "historical_regression_rows": sorted(historical_ids),
        "replay": {
            "records": len(records),
            "unique_rows": len(historical_ids),
            "included_records": sum(record["dense_row"] in selected_ids for record in records),
            "included_unique_rows": len(historical_ids & selected_ids),
            "missing_dense_rows": sorted(historical_ids - selected_ids),
            "method": "Saved candidate_error > 0.001 and baseline_error <= 0.001; no numerical solver executed.",
            "excluded_comparisons": "trace_values.csv LD labels use uniform reference columns, so LD-vs-uniform comparisons are excluded.",
        },
        "uniform_reference_audit": reference_audit,
        "reference_policy": {
            "saved": "Existing saved independent reference, with no new convergence certification.",
            "unresolved": "114 uniform discrepancies fail the existing strict cross-engine/stability criterion; report separately, never as passes.",
            "checked": "One independent cross-engine/stability-checked reference replaces its known-bad raw value; original and replacement are recorded in uniform_reference_audit.",
            "missing": "Absent/nonpositive/nonfinite LD reference; report separately, never as a pass.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evidence-root", type=Path, help="Directory containing saved benchmark output checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/data/cpu_accuracy")
    args = parser.parse_args()
    manifest = build(args.root.resolve(), args.output, args.evidence_root)
    print(
        json.dumps({"rows": manifest["rows"], "cohorts": manifest["cohorts"], "replay": manifest["replay"]}, indent=2)
    )


if __name__ == "__main__":
    main()
