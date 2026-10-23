# Telco Churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Predict whether a telecom customer will churn (`Churn`: `Yes`/`No`) from a
single customer snapshot (IBM Telco Customer Churn dataset).

- **TP01**: reproducible project structure, pinned environment and a
  data-inspection program.
- **TP02**: a configurable training workflow. A scikit-learn `Pipeline`
  (preprocessing + classifier) is compared across three learned models and a
  baseline on one fixed split, and the selected *fitted* pipeline is saved
  with metadata so it can be reloaded for inference without retraining.

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

Prerequisites: [uv](https://docs.astral.sh/uv/) and Git. Every command in
this README works unchanged on Linux, macOS and Windows (PowerShell or Git
Bash); nothing depends on `make`, `curl` or `sha256sum`.

```sh
git clone <your-repo-url> telco-churn
cd telco-churn
uv sync --locked          # recreate the exact environment from uv.lock
```

The interpreter is pinned to **Python 3.12.11** by the committed
`.python-version` file; `uv.lock` alone only constrains it to `3.12.*`, so
without the pin uv would reuse any 3.12 already installed. uv downloads
3.12.11 automatically if it is missing.

> Windows note: if `uv run` or an import fails with *"An Application Control
> policy has blocked this file"*, Windows Smart App Control is blocking an
> unsigned binary: the `.venv\Scripts\python.exe` launcher or compiled
> modules such as scikit-learn's. That is a local machine policy, not a
> project issue. Use a machine without that policy, or WSL.

## Source data

The raw customer snapshot is **not** in Git, and `data/raw/` does not exist
in a fresh clone. One command creates it, downloads the commit-pinned CSV and
verifies its hash (pure Python, so it works the same in Bash, zsh, PowerShell
or cmd):

```sh
uv run python scripts/download_data.py
# SHA-256 verified: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

It stops with an error, and writes nothing, if the downloaded bytes do not
match. It also refuses to overwrite an existing `data/raw/telco.csv` whose
hash differs.

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
| Hyperparameters / run config | `configs/` | declarative, outside source code |
| Notebooks | `notebooks/` | exploration only; not the runnable workflow |
| Behavior checks | `tests/` | e.g. predictions survive save/reload |

Note: folder names communicate intent; the project does not implement
Bronze/Silver/Gold storage, data versioning or access control by itself.

## TP02: train, compare, save and reuse the model

All settings live in [`configs/train.json`](configs/train.json): split seed
and fractions, feature roles, preprocessing, candidates, the selection rule
(fixed **before** looking at results) and output paths. Run from the project
root, after downloading the data:

```sh
uv run python scripts/first_model.py      # milestone 1: 2 features + baseline
uv run python scripts/train.py            # compare candidates, select, save artifact
uv run python scripts/verify_artifact.py  # fresh-process reload check (no fit, no labels)
uv run python scripts/predict.py --input evidence/tp02/example_new_customers.csv
uv run pytest                             # the reliability checks
```

### Prediction task and split

- Positive class: `Churn = Yes`, encoded as `1`. `customerID` is kept for
  split checks and passed through at inference, but is **never** an input.
  `Churn` is never an input either.
- One stratified 60/20/20 split, seed `42`: 4 225 train / 1 409 validation /
  1 409 test customers. The allocation is saved per customer in
  `models/churn_pipeline/split_allocation.csv`.
- Train fits everything, validation compares and selects, and **test is
  reserved**: no script evaluates on it.

### Input contract (stateless parsing before the pipeline)

`FeatureSchema` in [`telco_churn/features.py`](telco_churn/features.py)
applies the same rules at training and inference time. It learns nothing from
data.

| Inputs | Accepted | Missing value | Error |
|---|---|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | number or numeric text | blank / whitespace-only → `NaN` → median imputed | other non-numeric text → `ValueError` (never silently `NaN`) |
| `Contract`, `InternetService`, `PaymentMethod` | text (whitespace stripped) | blank → `NaN` → most-frequent imputed | unseen category → all-zero one-hot (still scored; see limitations) |
| any of the six | | | column absent → `KeyError` |

### Pipeline

`ChurnPipelineBuilder` in [`telco_churn/pipeline.py`](telco_churn/pipeline.py)
returns a new, **unfitted** pipeline for each candidate:

```text
Pipeline
├── preprocessor: ColumnTransformer (other columns dropped)
│   ├── num: SimpleImputer(median) → StandardScaler   [tenure, MonthlyCharges, TotalCharges]
│   └── cat: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
│                                                      [Contract, InternetService, PaymentMethod]
└── classifier
```

A single `pipeline.fit(X_train, y_train)` learns every statistic (medians,
means/scales, categories) from the training partition only. Validation and
new records reuse them through `predict_proba`.

### Comparison and selection (validation only)

Selection metric chosen in advance: **average precision** on positive-class
probabilities. Churners are about 26.5% of customers, and the team needs to
rank who is at risk, which accuracy does not measure. Precision, recall and
F1 at threshold 0.5, accuracy and confusion matrices are reported as
supporting evidence. Rule: highest validation average precision; candidates
within 0.01 count as tied, and ties go to the simpler model.

| Model | AP | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| majority baseline | 0.265 | 0.000 | 0.000 | 0.000 | 0.735 |
| logistic regression | 0.648 | 0.664 | 0.529 | 0.589 | **0.804** |
| decision tree (depth 5) | 0.617 | 0.636 | **0.602** | **0.618** | 0.803 |
| **random forest** (300 trees, depth 8) | **0.664** | 0.653 | 0.484 | 0.556 | 0.795 |

Full table with confusion matrices, parameters and fit times:
[`reports/model_comparison.json`](reports/model_comparison.json) (also as
`.csv`). Selection note: [`evidence/tp02/selection_note.md`](evidence/tp02/selection_note.md).

### Saved artifact (`models/churn_pipeline/`)

| File | Content |
|---|---|
| `pipeline.joblib` | fitted preprocessing **and** classifier, as one object |
| `metadata.json` | data SHA-256, seed, split sizes, feature roles and input contract, preprocessing settings and learned statistics, model parameters, validation metrics, positive class, threshold, selection decision, config hash, Git commit and per-file source hashes, Python and library versions, and the SHA-256 of every file here |
| `split_allocation.csv` | customer → train / validation / test |
| `reference_inputs.csv`, `reference_predictions.csv` | 10 raw validation customers (no labels) and the predictions made before saving |

`pipeline.joblib` is a pickle, which can run code when loaded. Load only
artifacts you trust. `ArtifactStore.load` refuses a file whose SHA-256 differs
from `metadata.json`, and warns if installed library versions differ from
the training ones.

### Reliability evidence

- `reports/reload_check.json`: a separate process loads the artifact with
  `fit` disabled, predicts the reference batch and matches the saved
  probabilities (max |difference| = 0, tolerance 1e-12). Learned statistics
  are unchanged afterwards.
- [`tests/test_reliability.py`](tests/test_reliability.py) (16 tests;
  output in `evidence/tp02/test_output.txt`) covers:
  - customer IDs are disjoint across partitions and the split is reproducible;
  - imputer medians equal *training* medians (they differ from full-data
    medians, so the check can detect leakage);
  - `predict` does not change fitted state;
  - a blank `TotalCharges` is imputed with the training median;
  - an unseen category encodes as all zeros;
  - malformed numeric text and missing columns fail visibly;
  - a tampered artifact is refused;
  - fresh-process reload matches.

### Limitations

- Verified on Linux (Ubuntu 24.04, Python 3.12.11, scikit-learn 1.9.1). The
  code has nothing OS-specific, but a native Windows run was not possible on
  the test machine because of Smart App Control (see the Windows note above).
- Handling an unseen category only means the pipeline does not crash.
  Predictions for such customers are not shown to be reliable.
- Probabilities are not calibrated: a score of 0.8 is a ranking signal, not
  a guaranteed 80% churn risk. Threshold 0.5 is a convention, not a business
  decision.
- No hyperparameter search or cross-validation; one split, modest settings.
  Fit times are descriptive, measured on one machine.
- The test set has not been evaluated yet. Do it once, for the frozen model,
  after selection is final.

## Handover notes

See `evidence/tp01/handover.md` for peer-execution feedback, the issues
found and their resolutions, and the current limitations of this handover.
Written answers to every checkpoint question of the TP01 brief are kept in
`evidence/tp01/questions_answers.md`. TP02 peer review:
`evidence/tp02/peer_review.md`.
