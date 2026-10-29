"""Guard the numerical screen's decisions, especially uncertain references."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cpu_accuracy_tools", ROOT / "tools/cpu_accuracy.py")
accuracy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accuracy)


def case(identifier=0, reference=10.0, baseline=10.0, state="saved"):
    row = dict(
        dense_row=str(identifier), cohort="difficult", config_id="0", x="0.1", y="0.2", rho="0.001", s="1", q="0.1"
    )
    for profile in accuracy.PROFILES:
        row.update(
            {
                f"reference_{profile}": str(reference),
                f"baseline_{profile}": str(baseline),
                f"{profile}_reference_state": state,
            }
        )
    return row


def test_uncertain_and_missing_references_are_not_accuracy_passes():
    rows = [case(reference=5.0, state="unresolved"), case(1, reference="", state="missing")]
    summary, records = accuracy.assess(rows, [[10.0, 5, 0, 20], [10.0, 8, 0, 20]], "uniform")
    assert summary["comparable_saved_references"] == 0
    assert summary["unresolved_references"] == summary["missing_references"] == 1
    assert summary["raw_reference_misses"] == 1
    assert summary["regression_gate_passed"]  # baseline retained; no physical precision claim
    assert not any(row["comparable"] for row in records)
    changed, _ = accuracy.assess(rows, [[10.1, 5, 0, 20], [10.0, 8, 0, 20]], "uniform")
    assert changed["unresolved_changes_needing_review"] == 1
    assert not changed["regression_gate_passed"]


def test_new_reference_failure_and_worsening_known_failure_are_distinct():
    rows = [case(), case(1, baseline=10.1)]
    summary, _ = accuracy.assess(rows, [[10.02, 5, 0, 20], [10.2, 8, 0, 20]], "ld")
    assert summary["new_reference_misses"] == 1
    assert summary["known_misses_worsened"] == 1
    assert not summary["regression_gate_passed"]


def test_nonfinite_solver_value_fails_even_without_reference():
    summary, _ = accuracy.assess([case(reference=np.nan, state="missing")], [[np.nan, 5, 0, 20]], "ld")
    assert summary["invalid_values"] == 1
    assert not summary["regression_gate_passed"]
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize(
    "row",
    [
        case(baseline=np.nan),
        case(baseline=0.0),
        case(state="unknown"),
        case(reference=np.nan),
        case(state="missing"),
    ],
)
def test_malformed_baseline_or_reference_cannot_silently_pass(row):
    with pytest.raises(ValueError):
        accuracy.validate_cases([row])


def test_empty_history_is_not_a_successful_replay():
    assert not accuracy.replay_history({"historical_regressions": []}, [case()])["regression_gate_passed"]


def test_history_replay_checks_values_rather_than_only_row_membership():
    row = case()
    row.update(baseline_tier="5", baseline_status="0", baseline_nodes="20")
    history = {
        "historical_regressions": [
            dict(
                candidate="example",
                profile="uniform",
                path="example.csv",
                records=[dict(dense_row=0, candidate_value=10.0, candidate_error=0.01, baseline_error=0.0)],
            )
        ]
    }
    report = accuracy.replay_history(history, [row])
    assert report["historical_crossings"] == 1
    assert report["missing_dense_rows"] == []
    assert report["detected_crossings"] == 0
    assert not report["regression_gate_passed"]


def test_grouped_calls_reject_inconsistent_lens_parameters():
    rows = [case(), case(1)]
    rows[1]["rho"] = "0.002"
    with pytest.raises(ValueError, match="Inconsistent lens parameters"):
        accuracy.validate_cases(rows)


def test_frozen_cases_cover_recorded_failures_and_keep_holdout_configs_separate():
    fixtures = ROOT / "tests/data/cpu_accuracy"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    rows = accuracy.read_cases(fixtures / "cases.csv")
    assert accuracy.digest(fixtures / "cases.csv") == manifest["cases_sha256"]
    compact = [row for row in rows if row["cohort"] in ("difficult", "survey")]
    replay = accuracy.replay_history(manifest, compact)
    assert replay["historical_crossings"] > 0
    assert replay["detected_crossings"] == replay["historical_crossings"]
    assert replay["regression_gate_passed"]
    assert {200150, 317044} <= {int(row["dense_row"]) for row in compact}
    holdout = [row for row in rows if row["cohort"] == "holdout"]
    assert holdout and any(row["cohort"] == "survey" for row in compact)
    assert {row["config_id"] for row in compact}.isdisjoint({row["config_id"] for row in holdout})
    assert {int(row["baseline_tier"]) for row in compact} == {5, 8}
    corrected = next(row for row in rows if int(row["dense_row"]) == 317044)
    assert corrected["uniform_reference_state"] == "checked"
    assert abs(float(corrected["reference_uniform"]) / 31.880995191627925 - 1) < 1e-14
