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

- Branch: `docs/project-onboarding` (README onboarding rewrite).
- Review question from peer: *"`TotalCharges` is read as text because of 11
  blank values. Should TP02 impute them inside the preprocessing pipeline, or
  drop the 11 rows?"*
- Response recorded here: **impute inside the pipeline, never edit the raw
  file.** Dropping rows discards 11 real customers (~0.16%) and bakes a data
  decision into a place (`data/raw/`) that must stay immutable; a
  `SimpleImputer`/custom transformer inside the TP02 pipeline keeps the
  decision versioned, configurable (`configs/`) and applied identically at
  inference time.

## Current limitations (what is NOT yet reproducible)

- A **trained model** is not reproducible yet: that needs TP02's pipeline
  code, the training script, the recorded configuration, package versions and
  the random seed — plus a saved fitted pipeline in `models/` with metadata
  linking it to a report, data hash and commit.
- No automated tests yet (`tests/` planned in TP02).
- The remote GitHub repository and the pull request are pending (see
  "Git workflow" below).

## Git workflow status

- Local history: initial commit on `main`, documentation change on
  `docs/project-onboarding` (focused README-only diff).
- The GitHub remote was not configured during this session (repository must
  be created under the student's own account; no credentials here). Interim
  evidence is therefore local commits + this written peer review, per the
  brief's fallback. To finish: `git remote add origin <url>`,
  `git push -u origin main docs/project-onboarding`, then open the PR from
  `docs/project-onboarding` into `main` with the review question and response
  above in the PR description.
