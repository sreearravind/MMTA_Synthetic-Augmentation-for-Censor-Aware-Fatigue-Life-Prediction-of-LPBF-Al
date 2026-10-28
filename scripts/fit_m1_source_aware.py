#!/usr/bin/env python3
"""Locked M1 source-aware censored Weibull fit and external comparison.

See docs/m1_source_aware_model_and_external_evaluation_lock_2026-10-06.md.
Run --stage validate before --stage external. The latter refuses to run
unless all training-source and quadrature checks passed.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp

from fit_weibull_exact_baseline import fit as fit_e0

LN10 = math.log(10)
EXPECTED_EXACT = "29aff38de63880dbfbaf65d3b876b81bd5afabfaaaac1615e2e503fbfa661a30"
EXPECTED_MANIFEST = "0df2fdec2399cae1c76c202a323691431bdc96bb9eb31fa278f6b307ac84eb91"
EXPECTED_EXTERNAL_BLOB = "840b7855fdc0d51c988dc6d7816f9b54cb6e2276"
FOLDS = ("E0_Wu_2021", "E0_Romano_2018", "E0_Chen_2024")
TAUS = (0.25, 0.10, 0.50)  # primary first; the other two are sensitivities
FIT_NODES = 161
CHECK_NODES = 321


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_sha(path):
    body = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    if not records:
        raise ValueError("Cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def quadrature(nodes):
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    return np.sqrt(2) * points, np.log(weights) - 0.5 * math.log(math.pi)


def conditional_logterms(params, stress, cycles, event, offsets):
    alpha, beta, log_k = params
    k = math.exp(log_k)
    x = np.log2(np.asarray(stress) / 100.0)
    z = np.log10(np.asarray(cycles))
    mu = alpha + beta * x[:, None] + offsets[None, :]
    h = LN10 * k * (z[:, None] - mu)
    if np.any(h > 600):
        return None
    tail = np.exp(h)
    return np.asarray(event)[:, None] * (log_k + math.log(LN10) + h) - tail


def study_marginal_objective(params, grouped, tau, nodes):
    points, logweights = quadrature(nodes)
    offsets = tau * points
    total = 0.0
    for group in grouped:
        terms = conditional_logterms(params, *group, offsets)
        if terms is None:
            return 1e300
        total -= float(logsumexp(logweights + np.sum(terms, axis=0)))
    return total if math.isfinite(total) else 1e300


def groups_for(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_id"]].append(r)
    return [
        (
            np.array([float(r["stress_amplitude_MPa"]) for r in group]),
            np.array([float(r["cycles_failure_or_bound"]) for r in group]),
            np.array([int(r["failure_event_1"]) for r in group]),
        )
        for _, group in sorted(groups.items())
    ]


def fit_m1(rows, tau):
    grouped = groups_for(rows)
    e0, _ = fit_e0(rows)
    alpha0 = float(e0.x[0])
    starts = (np.asarray(e0.x), np.array([alpha0, -1.0, -0.5]),
              np.array([alpha0, -3.0, 0.5]))
    bounds = ((0., 12.), (-12., 0.), (-4., 3.))
    solutions = [
        minimize(study_marginal_objective, start,
                 args=(grouped, tau, FIT_NODES), method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 3000, "ftol": 1e-12, "maxls": 50})
        for start in starts
    ]
    converged = [s for s in solutions if s.success and math.isfinite(s.fun)
                 and s.fun < 1e299]
    if not converged:
        raise RuntimeError(f"No converged M1 fit at tau={tau}: "
                           + "; ".join(s.message for s in solutions))
    result = min(converged, key=lambda s: s.fun)
    obj_check = study_marginal_objective(result.x, grouped, tau, CHECK_NODES)
    delta = abs(result.fun - obj_check)
    if delta > 1e-5:
        raise RuntimeError(f"Quadrature objective instability: {delta:g}")
    return result, e0, delta


def predictive_logterms(params, tau, stress, cycles, event, nodes=FIT_NODES):
    points, logweights = quadrature(nodes)
    terms = conditional_logterms(params, np.asarray(stress),
                                 np.asarray(cycles), np.asarray(event),
                                 tau * points)
    if terms is None:
        raise FloatingPointError("Nonfinite predictive likelihood")
    return logsumexp(terms + logweights[None, :], axis=1)


def e0_logterms(params, stress, cycles, event):
    terms = conditional_logterms(params, stress, cycles, event, np.array([0.]))
    if terms is None:
        raise FloatingPointError("Nonfinite E0 predictive likelihood")
    return terms[:, 0]


def prediction_checks(params, tau, stress, cycles, event):
    a = predictive_logterms(params, tau, stress, cycles, event, FIT_NODES)
    b = predictive_logterms(params, tau, stress, cycles, event, CHECK_NODES)
    delta = float(np.max(np.abs(a - b)))
    if delta > 1e-5:
        raise RuntimeError(f"Predictive quadrature instability: {delta:g}")
    return a, delta


def median_m1(params, tau, stress):
    # S_Z(z)=0.5. Search beyond plausible log10 fatigue-life support.
    f = lambda z: predictive_logterms(params, tau, [stress], [10.0 ** z], [0])[0] + math.log(2)
    return brentq(f, 0.0, 12.0, xtol=1e-10)


def median_e0(params, stress):
    alpha, beta, log_k = params
    return alpha + beta * math.log2(stress / 100) + math.log10(math.log(2)) / math.exp(log_k)


def validate_inputs(exact_path, manifest_path, external_path):
    assert sha256(exact_path) == EXPECTED_EXACT, "Exact input hash differs from lock"
    assert sha256(manifest_path) == EXPECTED_MANIFEST, "Split manifest hash differs from lock"
    assert blob_sha(external_path) == EXPECTED_EXTERNAL_BLOB, "External ledger blob differs from lock"
    exact = read_csv(exact_path)
    assert len(exact) == 66 and len({r["record_id"] for r in exact}) == 66
    counts = Counter((r["study_id"], r["failure_event_1"]) for r in exact)
    assert counts == Counter({("Wu_2021", "1"): 33, ("Wu_2021", "0"): 8,
                             ("Romano_2018", "1"): 6, ("Romano_2018", "0"): 1,
                             ("Chen_2024", "1"): 18})
    assert all(float(r["stress_amplitude_MPa"]) > 0
               and float(r["cycles_failure_or_bound"]) > 0 for r in exact)
    manifest = [r for r in read_csv(manifest_path)
                if r["analysis"] == "primary_exact_LOSO"]
    assert len(manifest) == 198
    by_id = {r["record_id"]: r for r in exact}
    folds = {}
    for fold in FOLDS:
        members = [r for r in manifest if r["fold_id"] == fold]
        assert len(members) == 66
        train = [by_id[m["record_key"].removeprefix("EXACT:")]
                 for m in members if m["role"] == "train"]
        test = [by_id[m["record_key"].removeprefix("EXACT:")]
                for m in members if m["role"] == "test"]
        assert len(train) + len(test) == 66
        assert not ({r["study_id"] for r in train} & {r["study_id"] for r in test})
        assert {r["study_id"] for r in test} == {fold.removeprefix("E0_")}
        folds[fold] = (train, test)
    external = json.loads(external_path.read_text(encoding="utf-8"))
    assert len(external) == 49 and len({r["record_id"] for r in external}) == 49
    included = [r for r in external if r["primary_holdout"] is True]
    held = [r for r in external if r["primary_holdout"] is False]
    assert len(included) == 46 and len(held) == 3
    assert {r["record_id"] for r in held} == {
        "HN2019_VF_15", "HN2019_SB_05", "HN2019_SB_10"}
    counts = Counter((r["finish"], r["event"]) for r in included)
    assert counts == Counter({("vibrofinished", "failure"): 14,
                             ("vibrofinished", "runout"): 2,
                             ("sandblasted", "failure"): 13,
                             ("sandblasted", "runout"): 2,
                             ("machined and polished", "failure"): 13,
                             ("machined and polished", "runout"): 2})
    for r in included:
        assert r["source_doi"] == "10.3390/met9101063"
        assert r["stress_ratio_R"] == 0.1
        for level in ("low", "center", "high"):
            assert abs(r[f"sigma_amplitude_mpa_{level}"] -
                       .45 * r[f"sigma_max_mpa_{level}"]) < .011
        if r["event"] == "runout":
            assert r["cycles_center_or_stop"] == 5_000_000
        else:
            assert r["cycles_glyph_low"] < r["cycles_center_or_stop"] < r["cycles_glyph_high"]
    return exact, folds, included


def params_record(label, tau, rows, m1, e0, obj_delta, pred_delta):
    return dict(fit=label, tau=tau, train_n=len(rows),
                train_study_count=len({r["study_id"] for r in rows}),
                train_fail=sum(int(r["failure_event_1"]) for r in rows),
                train_runout=sum(1-int(r["failure_event_1"]) for r in rows),
                alpha=float(m1.x[0]), beta=float(m1.x[1]),
                weibull_shape_k=math.exp(m1.x[2]),
                marginal_training_nll=float(m1.fun),
                e0_alpha=float(e0.x[0]), e0_beta=float(e0.x[1]),
                e0_shape_k=math.exp(e0.x[2]),
                quadrature_objective_difference_161_321=obj_delta,
                quadrature_max_predictive_log_difference_161_321=pred_delta,
                beta_at_constraint=bool(m1.x[1] <= -11.999999 or m1.x[1] >= -1e-6),
                shape_at_constraint=bool(m1.x[2] <= -3.999999 or m1.x[2] >= 2.999999))


def source_folds(exact, folds):
    fits, metrics = [], []
    for fold, (train, test) in folds.items():
        stress = np.array([float(r["stress_amplitude_MPa"]) for r in test])
        cycles = np.array([float(r["cycles_failure_or_bound"]) for r in test])
        event = np.array([int(r["failure_event_1"]) for r in test])
        for tau in TAUS:
            m1, e0, obj_delta = fit_m1(train, tau)
            pred, pred_delta = prediction_checks(m1.x, tau, stress, cycles, event)
            e0_pred = e0_logterms(e0.x, stress, cycles, event)
            fits.append(params_record(fold, tau, train, m1, e0, obj_delta, pred_delta))
            metrics.append(dict(fold=fold, tau=tau, test_n=len(test),
                                fail_n=int(event.sum()), runout_n=int(len(event)-event.sum()),
                                m1_mean_nll=float(-pred.mean()),
                                e0_mean_nll=float(-e0_pred.mean()),
                                m1_minus_e0=float(np.mean(e0_pred-pred)),
                                m1_failure_nll_sum=float(-pred[event==1].sum()),
                                m1_runout_nll_sum=float(-pred[event==0].sum())))
    return fits, metrics


def external_array(external, stress_level, life_level):
    stress = np.array([float(r[f"sigma_amplitude_mpa_{stress_level}"]) for r in external])
    cycles = np.array([float(
        r[f"cycles_glyph_{life_level}"] if r["event"] == "failure" and life_level != "center"
        else r["cycles_center_or_stop"]) for r in external])
    event = np.array([int(r["event"] == "failure") for r in external])
    return stress, cycles, event


def external_scores(exact, external):
    params, scores, predictions = [], [], []
    strata = {"overall": external,
              "VF": [r for r in external if r["finish"] == "vibrofinished"],
              "SB": [r for r in external if r["finish"] == "sandblasted"],
              "MP": [r for r in external if r["finish"] == "machined and polished"]}
    for tau in TAUS:
        m1, e0, obj_delta = fit_m1(exact, tau)
        s0, c0, v0 = external_array(external, "center", "center")
        _, pred_delta = prediction_checks(m1.x, tau, s0, c0, v0)
        params.append(params_record("full_exact_66", tau, exact, m1, e0, obj_delta, pred_delta))
        # E0 is identical across tau values; retain repeated rows only for a paired table.
        for stress_level in ("low", "center", "high"):
            for life_level in ("low", "center", "high"):
                scenario = stress_level + "_" + life_level
                stress, cycles, event = external_array(external, stress_level, life_level)
                log_m1, _ = prediction_checks(m1.x, tau, stress, cycles, event)
                log_e0 = e0_logterms(e0.x, stress, cycles, event)
                assert len(log_m1) == 46 and np.isfinite(log_m1).all()
                for group, rows in strata.items():
                    ix = [external.index(r) for r in rows]
                    e = event[ix]
                    a, b = log_m1[ix], log_e0[ix]
                    scores.append(dict(tau=tau, scenario=scenario, stratum=group,
                                       n=len(rows), failures=int(e.sum()),
                                       runouts=int(len(e)-e.sum()),
                                       m1_mean_nll=float(-a.mean()),
                                       e0_mean_nll=float(-b.mean()),
                                       m1_minus_e0=float(np.mean(b-a)),
                                       m1_failure_nll_sum=float(-a[e==1].sum()),
                                       m1_runout_nll_sum=float(-a[e==0].sum()),
                                       e0_failure_nll_sum=float(-b[e==1].sum()),
                                       e0_runout_nll_sum=float(-b[e==0].sum())))
                if scenario == "center_center":
                    for i, r in enumerate(external):
                        predictions.append(dict(record_id=r["record_id"], finish=r["finish"],
                                                event=r["event"], tau=tau,
                                                stress_amplitude_MPa=stress[i],
                                                cycles_or_bound=cycles[i],
                                                m1_log_contribution=float(log_m1[i]),
                                                e0_log_contribution=float(log_e0[i]),
                                                m1_minus_e0_nll=float(log_e0[i]-log_m1[i]),
                                                m1_median_log10=median_m1(m1.x, tau, stress[i]),
                                                e0_median_log10=median_e0(e0.x, stress[i]),
                                                failure_log10_absolute_error_m1=(
                                                    abs(median_m1(m1.x, tau, stress[i])-math.log10(cycles[i]))
                                                    if event[i] else ""),
                                                failure_log10_absolute_error_e0=(
                                                    abs(median_e0(e0.x, stress[i])-math.log10(cycles[i]))
                                                    if event[i] else "")))
    return params, scores, predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validate", "external"], required=True)
    ap.add_argument("--exact", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--external", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    exact, folds, external = validate_inputs(args.exact, args.manifest, args.external)
    args.output.mkdir(parents=True, exist_ok=True)
    validation_path = args.output / "validation_passed.json"
    if args.stage == "validate":
        fits, metrics = source_folds(exact, folds)
        write_csv(args.output / "development_fits.csv", fits)
        write_csv(args.output / "development_scores_exploratory.csv", metrics)
        status = dict(status="passed", exact_sha256=sha256(args.exact),
                      manifest_sha256=sha256(args.manifest),
                      external_blob_sha=blob_sha(args.external),
                      source_folds=list(folds), primary_tau=TAUS[0],
                      fit_count=len(fits), quadrature="161 vs 321 nodes, tolerance 1e-5",
                      note="Development fold scores previously inspected; exploratory")
        validation_path.write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2))
    else:
        if not validation_path.exists():
            raise RuntimeError("Run the validation stage before external scoring")
        status = json.loads(validation_path.read_text())
        assert status["status"] == "passed"
        assert status["exact_sha256"] == sha256(args.exact)
        assert status["manifest_sha256"] == sha256(args.manifest)
        assert status["external_blob_sha"] == blob_sha(args.external)
        params, scores, predictions = external_scores(exact, external)
        write_csv(args.output / "external_fits.csv", params)
        write_csv(args.output / "external_scores_9_scenarios.csv", scores)
        write_csv(args.output / "external_center_predictions_46.csv", predictions)
        run = dict(status="provisional graph-derived validation",
                   external_source="Hamidi Nasab 2019, Figure 10",
                   note="Two computational methods, one analyst; second human reader pending",
                   software=dict(python=platform.python_version(),
                                 numpy=np.__version__, scipy=scipy.__version__),
                   input_hashes=dict(exact_sha256=sha256(args.exact),
                                     manifest_sha256=sha256(args.manifest),
                                     external_blob_sha=blob_sha(args.external)),
                   primary_tau=TAUS[0], sensitivity_taus=TAUS[1:],
                   scenarios=9, external_records=len(external))
        (args.output / "run_metadata.json").write_text(json.dumps(run, indent=2) + "\n")
        print(json.dumps([r for r in scores if r["tau"] == .25
                          and r["scenario"] == "center_center"], indent=2))


if __name__ == "__main__":
    main()
