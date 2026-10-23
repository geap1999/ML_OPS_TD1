# Telco Churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predict whether a telecom customer will churn (`Churn`: `Yes`/`No`) from a
single customer snapshot (IBM Telco Customer Churn dataset). This repository
currently contains the **TP01** foundation: a reproducible project structure,
a pinned environment and a data-inspection program. Model training and
preprocessing pipelines arrive in **TP02**.

## Generated structure (TP01, Option A)

Generated with **Cookiecutter Data Science (CCDS) v2.3.0** via
`uvx --from cookiecutter-data-science ccds` with these options:

| Option | Value |
|---|---|
| Project name | `Telco Churn` |
| Repository name | `telco-churn` |
| Module name | `telco_churn` |
| Python version | `3.12` |
| Dataset storage | `none` (default; local CSV only) |
| Environment manager | `uv` |
| Dependency file | `pyproject.toml` (+ committed `uv.lock`) |
| Pydata packages | `none` (added explicitly: `pandas`) |
| Testing framework | `none` |
| Linting/formatting | `ruff` |
| License / docs / scaffold | none / none / none (empty module) |

CCDS v2 keeps the importable package at the repository root (`telco_churn/`,
no `src/` layout) — this is kept as generated.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and Git.

```sh
git clone <your-repo-url> telco-churn
cd telco-churn
uv sync --locked          # recreate the exact environment from uv.lock
```

## Source data

The raw customer snapshot is **not** in Git, and `data/raw/` does not exist
in a fresh clone. Create it, download the CSV and verify its hash:

```sh
mkdir -p data/raw
curl -fsSL -o data/raw/telco.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv
sha256sum data/raw/telco.csv
# expected: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

Provenance and field notes: [`references/data_source.md`](references/data_source.md).

## Run the inspection

```sh
uv run python scripts/inspect_data.py
```

One documented command; writes `reports/data_overview.json` (machine-readable)
and prints a summary. The report contains: dimensions and field types, target
counts and proportions, duplicate `customerID` count, missing-value counts
(both `isna()` and an explicit empty/whitespace-only string check), the source
SHA-256, and the interpreter/pandas versions of the run.

Latest committed result (`reports/data_overview.json`): 7 043 rows × 21
columns; `Churn`: 5 174 `No` (73.5%) vs 1 869 `Yes` (26.5%); 0 duplicate
customer IDs; 0 `NaN`; **11 whitespace-only `TotalCharges` values** — the
reason `TotalCharges` is read as text.

## Environment recreation evidence

`evidence/tp01/` keeps the run-1 report, the run-1 dependency tree/list, the
run-2 report produced after `uv sync --locked` into a fresh `.venv`, and
`recreation_comparison.md`: all report content is identical across the two
runs except the run timestamp.

## Where things go (project policy)

| Item | Location | Why |
|---|---|---|
| Original CSV (immutable) | `data/raw/` | original input, excluded from Git by policy; pinned URL + SHA-256 instead |
| Intermediate / prepared datasets | `data/interim/`, `data/processed/` | later, only if a transformation is worth persisting |
| Reusable inspection/training code | `telco_churn/` (package) | importable, versioned with the code |
| One-off entry points | `scripts/` | thin CLIs calling package functions |
| Fitted pipelines (joblib) | `models/` | serialized outputs, never code |
| Plots / generated reports | `reports/` (+ `reports/figures/`) | deliberately tracked when needed for assessment |
| Hyperparameters / run config | `configs/` (TP02) | declarative, outside source code |
| Notebooks | `notebooks/` | exploration only; not the runnable workflow |
| Behavior checks | `tests/` | e.g. predictions survive save/reload (TP02) |

Note: folder names communicate intent; the project does not implement
Bronze/Silver/Gold storage, data versioning or access control by itself.

## TP02 — training pipeline, comparison and reusable artifact

### Train (terminal)

```sh
uv run python scripts/train.py --milestone1   # first two-feature model + majority baseline
uv run python scripts/train.py                # full: 3 candidates + baseline, selection, artifact
```

Source: `data/raw/telco.csv` (SHA-256-checked above). Configuration:
`configs/train.json` — seed 42, stratified 60/20/20 split, six input
features (`customerID` and `Churn` never inputs), parsing and preprocessing
policies, candidate list, primary metric (validation **average precision**
for class `Yes`, chosen before inspecting results), 0.5 threshold policy.

### Predict (fresh process, inference only — no fit)

```sh
uv run python scripts/predict.py
```

Loads `models/selected_pipeline.joblib` (fitted preprocessing **+**
classifier together), predicts the batch in `models/inference_batch.csv`
and writes `reports/reload_check.json` proving agreement with the
training-time predictions (classes exact, scores within 1e-9).

### Checks

```sh
uv run pytest
```

13 focused checks (TP02 section 5): disjoint customer IDs across
train/validation/test, imputer statistics equal training-only medians (and
*not* the full-dataset value), prediction leaves statistics unchanged,
blank-numeric and unseen-category policies, loud failure on a missing
feature, save/reload agreement, no `fit` in the inference module.

### Results and decisions

- Comparison (`reports/model_comparison.json`, validation only):
  logistic AP 0.647 / decision tree 0.603 / random forest 0.604 /
  majority baseline 0.265. Selected: **logistic regression** — highest
  validation average precision, the metric fixed in advance; simplest
  candidate and fastest to fit.
- The reserved test set was evaluated exactly once, after selection
  (`metadata.json` → `final_test_evaluation`, guarded by a counter).
- Milestone 1 (`reports/first_model_metrics.json`): the two-feature model
  (accuracy 0.783, AP 0.599) beats the always-"No" baseline (accuracy
  0.735 but recall 0.000) — and the six-feature jump to AP 0.647 reflects
  the *feature set*, not the algorithm; no such claim is made.

### Artifact metadata

`models/metadata.json` retains: source hash + pinned URL, seed and
split sizes, customer allocations (`models/split_allocation.json`),
feature roles and parsing policy, preprocessing settings, classifier
parameters, validation metrics and fit durations, positive-class
definition, threshold policy, library versions, and the artifact hash.

### Limitations

- Unseen categories are encoded as all-zeros (`handle_unknown="ignore"`):
  predictions *run*, but their reliability for such records is not proven.
- 0.5 is the default threshold; a cost-driven threshold policy is a
  possible extension (selection must still happen on validation only).
- Fit durations are descriptive timings on one machine, not benchmarks.
- Score calibration (is 0.8 an 80% risk?) has not been verified.

## Roadmap (TP03)

Experiment tracking and model versioning, building on these candidates and
artifacts.

## Handover notes

See `evidence/tp01/handover.md` for peer-execution feedback, the one issue
found and its resolution, and the current limitations of this handover.
Written answers to every checkpoint question of the TP01 brief are kept in
`evidence/tp01/questions_answers.md`.
