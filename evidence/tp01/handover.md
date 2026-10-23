# TP01 handover note

## What a teammate can reproduce today

From a clean clone, following the README alone (no oral instructions):

1. Recreate the exact environment with `uv sync --locked` (Python 3.12.11,
   pandas 3.1.0, per `uv.lock`).
2. Download the original CSV from the commit-pinned URL and confirm its
   SHA-256 (`16320c9c…e91`, full value in `references/data_source.md`).
3. Run `uv run python scripts/inspect_data.py` and obtain the same
   machine-readable report as `reports/data_overview.json`
   (7 043 × 21, target 73.5% `No` / 26.5% `Yes`, 11 whitespace-only
   `TotalCharges`, 0 duplicate IDs).
4. Check `evidence/tp01/recreation_comparison.md` showing that a destroyed and
   recreated `.venv` produces identical statistics.

## Peer execution record

Peers executed the README steps in clean directories without oral help.
Two issues were found and fixed:

1. **Ordering issue:** the first inspection attempt failed with a
   `FileNotFoundError` before the data-download section was reached.
   *Resolution:* the README now places the download + hash check *before*
   "Run the inspection", and the script raises an explicit, actionable
   error pointing at the README.
2. **Missing directory in a fresh clone (found by clean-room testing):**
   `data/` is kept out of Git by policy, so `data/raw/` does not exist at
   all in a new clone and the download command
   `curl -o data/raw/telco.csv` fails before writing anything.
   *Resolution:* the README download step now starts with
   `mkdir -p data/raw`.

**Full clean-room walkthrough verified end to end:** fresh
`git clone` → `uv sync --locked` (Python 3.12.11, pandas 3.1.0) →
`mkdir -p data/raw` + download → `sha256sum` = `16320c9c…e91` (matches
`references/data_source.md`) → `uv run python scripts/inspect_data.py`
reproduces the committed report exactly (7 043 × 21; 5 174 `No` /
1 869 `Yes`; 11 whitespace-only `TotalCharges`; 0 duplicate IDs).

## Peer review record (documentation branch)

- Branch: `docs/project-onboarding` (README onboarding rewrite) → **PR #1,
  opened, reviewed and merged**.
- Review question from peer, posted on PR #1:
  *"`TotalCharges` is read as text because of the 11 blank values found by
  the inspection report. Should TP02 impute them inside the preprocessing
  pipeline, or drop the 11 rows? I'd also like to understand why you kept
  the raw CSV out of Git with only a hash as proof of identity."*
- Response posted as a reply on PR #1: **impute inside the pipeline, never
  edit the raw file.** Dropping rows discards 11 real customers (~0.16%)
  and bakes a data decision into a place (`data/raw/`) that must stay
  immutable; an imputer inside the TP02 pipeline keeps the decision
  versioned, configurable (`configs/train.json`) and applied identically at
  inference time. On the CSV: a filename does not identify content — two
  files named `telco.csv` can differ — so the commit-pinned URL + SHA-256
  (`references/data_source.md`) is the reproduction contract, verified in
  the clean-room test.
- Discussion visible at: https://github.com/geap1999/ML_OPS_TD1/pull/1

## Current limitations (what is NOT yet reproducible)

- A **trained model** is not reproducible yet: that needs TP02's pipeline
  code, the training script, the recorded configuration, package versions and
  the random seed — plus a saved fitted pipeline in `models/` with metadata
  linking it to a report, data hash and commit.
- No automated tests yet (`tests/` planned in TP02).
- The TP02 peers should each write their own individual answers
  (`questions_answers.md` is one student's wording).

## Git workflow status

- Remote: https://github.com/geap1999/ML_OPS_TD1
- History rewritten from the initial single push into logical commits on
  `master` (scaffold → environment → provenance → inspection → evidence).
- `docs/project-onboarding` branched from `master`, reviewed (peer question
  + response posted on the PR), and merged as PR #1 (merge commit
  `ab97a58`). Final TP01 state is on `master`.
- Remaining for TP02: keep the branch → PR → review workflow for training
  code, configs and tests.
