#!/usr/bin/env python3
"""Check frozen primary, ratio-transfer, and external-score ledgers.

This script does not refit models or regenerate external forecasts. For a
complete primary/ratio-transfer refit use compare_development_v2.py.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"13", "29", "47", "71", "101"}
PRIMARY = {"Wu_2021": (41, 33, 8), "Chen_2024": (18, 18, 0),
           "Matusu_2026": (133, 115, 18)}
EXPECTED_PRIMARY = {"C0": 1.27714, "C1": 1.27299,
                    "S0": 1.27449, "S1": 1.27330}
EXPECTED_ROMANO = {"C0": 5.40162, "C1": 3.27237,
                   "S0": 3.24821, "S1": 3.26747}
EXPECTED_EXTERNAL = {
    "Beretta": ("results/external_beretta_2022_2026-10-08",
                "overall_32", 32, 25, 7,
                {"C0": 1.55455, "C1": 1.48670, "S0": 1.48490, "S1": 1.48624}),
    "Al-Zuhairi": ("results/external_al_zuhairi_30um_2026-10-07",
                   "overall_23", 23, 23, 0,
                   {"C0": 1.73003, "C1": 1.61551, "S0": 1.60375, "S1": 1.61315}),
}

def rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

def approx(actual: float, expected: float, label: str, tol: float = 6e-5) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tol):
        raise AssertionError(f"{label}: {actual:.9f} differs from {expected:.5f}")

def check_cohorts() -> None:
    cohort = rows("data/cohorts/development_v2_199_2026-10-06.csv")
    folds = rows("data/validation/development_v2_grouped_fold_assignments_2026-10-06.csv")
    ids = [r["record_id"] for r in cohort]
    assert len(cohort) == len(folds) == len(set(ids)) == 199
    assert {r["record_id"] for r in folds} == set(ids)
    by_group = defaultdict(list)
    for record in cohort:
        by_group[record["publication_group_id"]].append(record)
    assert set(by_group) == set(PRIMARY) | {"Romano_2018"}
    for group, expected in {**PRIMARY, "Romano_2018": (7, 6, 1)}.items():
        items = by_group[group]
        n_failure = sum(int(x["failure_event_1"]) for x in items)
        assert (len(items), n_failure, len(items) - n_failure) == expected, group
    assert sum(int(r["failure_event_1"]) for r in cohort) == 172
    assert sum(r["stress_ratio_R"] == "0.1" for r in cohort) == 192
    assert all(r["stress_ratio_R"] == "0.1" for g in PRIMARY for r in by_group[g])
    assert all(float(r["stress_ratio_R"]) == -1 for r in by_group["Romano_2018"])
    for row in folds:
        if row["publication_group_id"] in PRIMARY:
            assert row["primary_holdout_fold"] == "holdout_" + row["publication_group_id"]
    print("PASS: 199 independent first exposures, 192 primary R=0.1, grouped source holdouts.")

def macro_by_arm(path: str, expected: dict[str, float]) -> None:
    metrics = rows(path)
    for arm in ("C0", "C1", "S0", "S1"):
        sub = [r for r in metrics if r["arm"] == arm]
        seeds = {r["seed"] for r in sub}
        if arm in ("C0", "C1"):
            assert seeds == {""} and len(sub) == (3 if "romano" not in path else 1)
        else:
            assert seeds == SEEDS and len(sub) == (15 if "romano" not in path else 5)
        by_group = defaultdict(list)
        for r in sub:
            by_group[r["fold"]].append(float(r["mean_nll"]))
        if "romano" not in path:
            assert set(by_group) == set(PRIMARY)
            val = sum(sum(v) / len(v) for v in by_group.values()) / 3
        else:
            assert set(by_group) == {"Romano_2018_transfer"}
            v = by_group["Romano_2018_transfer"]
            val = sum(v) / len(v)
        approx(val, expected[arm], path + " " + arm)
        print(f"  {arm}: {val:.5f}")

def check_external(label: str) -> None:
    folder, group, n, failures, runouts, expected = EXPECTED_EXTERNAL[label]
    summary = rows(folder + "/summary.csv")
    records = rows(folder + "/scores_per_record.csv")
    condition_col = "stratum" if label == "Beretta" else "condition"
    unique_id = "specimen_key" if label == "Beretta" else "original_specimen_id"
    nll_col = "nll" if label == "Beretta" else "nll_log10_life_density"
    assert len(records) == 12 * n, label
    baseline = [r for r in records if r["arm"] == "C1"]
    assert len(baseline) == len({r[unique_id] for r in baseline}) == n
    assert sum(int(r["event_failure"]) for r in baseline) == failures
    assert n - failures == runouts
    for arm in ("C0", "C1", "S0", "S1"):
        fit_sets = defaultdict(list)
        for r in records:
            if r["arm"] == arm:
                fit_sets[r["seed"]].append(r)
        expected_seeds = {""} if arm in ("C0", "C1") else SEEDS
        assert set(fit_sets) == expected_seeds, (label, arm)
        assert all(len(v) == n for v in fit_sets.values()), (label, arm)
        val = sum(sum(float(r[nll_col]) for r in v) / n
                  for v in fit_sets.values()) / len(fit_sets)
        summary_row = [r for r in summary if r[condition_col] == group and r["arm"] == arm]
        assert len(summary_row) == 1
        approx(val, float(summary_row[0]["mean_nll"]), label + " archive " + arm, 1e-10)
        approx(val, expected[arm], label + " published " + arm)
        print(f"  {arm}: {val:.5f}")
    print(f"PASS: {label}, {n} first exposures ({failures} failures / {runouts} runouts).")

def main() -> None:
    check_cohorts()
    print("Primary equal-publication means:")
    macro_by_arm("results/development_v2/fold_metrics.csv", EXPECTED_PRIMARY)
    print("Romano secondary means (not primary external validation):")
    macro_by_arm("results/development_v2/romano_transfer_metrics.csv", EXPECTED_ROMANO)
    for name in EXPECTED_EXTERNAL:
        print(name + " fixed-model external archive:")
        check_external(name)
    print("PASS: all archived comparison scores match the frozen report values.")
    print("Scope: independently audits archived scores; full external reforecasting is not included.")

if __name__ == "__main__":
    main()
