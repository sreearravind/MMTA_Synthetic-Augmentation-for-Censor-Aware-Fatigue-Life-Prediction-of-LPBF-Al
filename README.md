# Supplementary code and data for anonymous peer review

This repository contains only the data, computational scripts, precomparison protocol, and archived score evidence needed to assess the fatigue-life modelling results reported in the accompanying manuscript. It is a review package, not a mirror of a manuscript-development repository.

## Evaluation scope

The original experimental inventory contains **199 distinct first exposures**: Wu (41), Chen (18), Matušů (133), and Romano (7). The **primary evaluation** comprises 192 tests at stress ratio **R = 0.1** (166 failures and 26 runouts), grouped into three whole-publication holdouts. Romano's seven R = −1 records (six failures and one runout) form a separate, secondary transfer challenge **within** the development inventory. Beretta (32 first exposures, 25 failures and seven runouts) and Al-Zuhairi (23 failure-only exposures) are distinct external score checks, not additional primary folds.

C0 is the pooled experimental-only Weibull model; C1 is the source-marginal experimental-only model. S0 and S1 are teacher-generated, training-only augmentation arms; S1 uses a fixed training-source runout-support filter. The five generation seeds (13, 29, 47, 71, 101) are not independent experiments. The primary success criterion was not met.

## Contents

- `data/cohorts/` — the fixed 199-row first-exposure inventory, including observed failure/runout indicators and recorded stopping bounds.
- `data/validation/` — original publication-held-out assignments and prespecified fold feasibility summary.
- `scripts/fit_weibull_exact_baseline.py`, `scripts/fit_m1_source_aware.py`, `scripts/compare_development_v2.py` — the original C0/C1 and four-arm primary/secondary model-fitting implementation, with input-blob integrity checks.
- `scripts/verify_results.py` — a dependency-free audit of source membership and archived NLL results.
- `docs/development_v2_precomparison_protocol_2026-10-06.md` — the prespecified training-only augmentation and validation criteria, including the scoring rule.
- `results/development_v2/` — frozen parameter, publication-fold, Romano-transfer, and run metadata; the modest-sized Romano per-record prediction ledger is provided for inspecting the secondary challenge.
- `results/external_beretta_2022_2026-10-08/` and `results/external_al_zuhairi_30um_2026-10-07/` — fixed external forecast tables, forecast manifests, per-record scores, and aggregate score tables for inspection and independent aggregation.

There are no manuscript drafts, cover letters, personal records, unpublished exploratory candidates, source workbooks, figures, author details, or unrelated experiments in this repository.

## Reproduce and verify

Python **3.12** was used for the frozen model runs with the library versions in `requirements.txt`. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/verify_results.py
```

The second command checks the frozen 199/192/7 source counts, three publication-held-out macro means, Romano scores, and both external cohorts' per-record and aggregate negative log likelihoods.

To **refit the primary four-arm models and the secondary Romano challenge** using the original code and locked inputs, run:

```bash
python scripts/compare_development_v2.py --root . --out reproduced/development_v2
```

The fit can be substantially slower than the score audit. The `reproduced/` directory is a local output and is not part of the archival review package. Numeric output may vary at negligible floating-point tolerance with optimizer/platform changes.

### Frozen results (lower censored NLL is better)

| Cohort | C0 | C1 | S0 (five-seed mean) | S1 (five-seed mean) |
| :-- | --: | --: | --: | --: |
| Primary, equal publication macro | 1.27714 | 1.27299 | 1.27449 | 1.27330 |
| Romano, separate seven-exposure challenge | 5.40162 | 3.27237 | 3.24821 | 3.26747 |
| Beretta, one external campaign | 1.55455 | 1.48670 | 1.48490 | 1.48624 |
| Al-Zuhairi, one failure-only external preprint | 1.73003 | 1.61551 | 1.60375 | 1.61315 |

**Important scope boundary:** The two external folders permit auditing archived forecasts, per-record scores, and their aggregates. The source-specific, third-party outcome intake workbooks and external forecast/score generation scripts are intentionally not redistributed in this minimal reviewer package. Accordingly, the external forecast generation and outcome joining cannot be rerun completely from these files alone. The original primary and Romano comparison *can* be rerun from the included inputs and code. No external cohort was added to the primary macro, and no external outcomes were used to refit model parameters.

## Data provenance and reuse

Record-level fatigue observations were transcribed or extracted from published experimental sources. Source DOI, test location, specimen/protocol metadata and event indicators retained in the 199-record inventory support attribution and auditing. Reviewers should refer to the respective original publications and supplemental-data terms for third-party reuse. This supplementary package does not grant new rights to the source publications or their supplemental materials.

The archived outcomes and scores are research records, not independent prospective replications. Runout events are right-censored at the documented specimen-specific lower bound, not treated as failures at a common nominal life.

## Reviewer use

Use the **anonymized repository mirror URL** associated with the manuscript, rather than the source GitHub repository URL. The repository is intentionally author-neutral. To retain review anonymity, avoid distributing URLs pointing to personal accounts, unpublished full manuscripts, or unrelated development histories.
