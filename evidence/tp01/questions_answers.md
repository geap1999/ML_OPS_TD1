# TP01 checkpoint questions — written answers

Answers to every "Questions:" block of the TP01 brief, kept as individual
explanation evidence. Written after implementing and testing the project.

## 1. Project and environment

**How do `pyproject.toml`, `uv.lock` and `.venv/` differ?**

- `pyproject.toml` is the *declaration*: project metadata (name, version,
  Python requirement) and the abstract dependencies we chose (`pandas`,
  `ruff`, ...). Human-authored, versioned in Git.
- `uv.lock` is the *resolution*: the exact versions selected for every
  direct and transitive dependency, so everyone gets byte-identical
  installs. Machine-generated but deliberately committed.
- `.venv/` is the *installation*: an actual folder with the interpreter and
  installed packages on one machine. Regenerable at any time from the two
  files above (`uv sync --locked`), never committed, never copied between
  projects or machines.

Rule of thumb: declare in `pyproject.toml`, pin in `uv.lock`, throw away
and rebuild `.venv/` whenever useful.

**Why separate notebooks from reusable source code?**

A notebook is a linear, stateful experiment log: its cells depend on
execution order, it is hard to import, hard to test and painful to diff.
The package (`telco_churn/`) holds the logic in importable, tested,
reusable functions; the thin script (`scripts/inspect_data.py`) becomes the
one documented entry point. A notebook may *call* the package for
exploration, but nothing in the project depends on notebook state — so the
runnable workflow survives even if the notebook is deleted or half-run.

**Where should the fitted model go, and where should the code that builds
it go?**

A fitted pipeline is data produced by a run → `models/`, saved with joblib
together with metadata (data hash, config, versions). The code that builds
and fits it is source → the importable package (`telco_churn/pipeline.py`,
`training.py` in TP02). A `modeling/` package directory contains Python
code, never serialized outputs — mixing the two makes artifacts unfindable
and code untestable.

## 2. Identifying the data

**Can two files with the same name contain different data?**

Yes — a filename is a label, not an identity. Someone can re-save the CSV
from Excel, reorder rows, or rename a different export to `telco.csv`, and
every tool that trusts the name silently uses different bytes. That is why
we record and verify a SHA-256 hash: it identifies the *content*, not the
name.

**Does a matching hash prove that the business data are correct?**

No. The hash proves only that our bytes equal the pinned upstream bytes —
integrity and identity, not truth. If the snapshot upstream was incomplete
or biased, we reproduce it faithfully. Correctness of the business data is
a question for the data producer and for sanity checks (row counts, label
balance, field plausibility), not for a hash.

**Why should the target and customer ID not become ordinary model inputs
later?**

`Churn` is what we predict: training with it as a feature is label leakage
— the model would learn to copy the answer. `customerID` is a unique
identifier with no business meaning: the model would memorize per-customer
patterns (or, worse, churn correlated with ID assignment order) that cannot
generalize to new customers. The ID is useful for joining and for
deduplication checks, then excluded from features.

## 3. Inspection program

**Why might a charge column be read as text?**

`TotalCharges` contains 11 fields that are blank/whitespace instead of
numbers. pandas cannot make a numeric dtype out of a column with non-numeric
strings, so the whole column is read as string dtype. A numeric spread across
all rows proves the column is text. This is exactly what our report shows
(`TotalCharges: str`, 11 empty-after-strip values) and what TP02's pipeline
will have to handle explicitly.

**Why should the script save a report rather than only print output?**

A print vanishes with the terminal; a JSON report is durable evidence: it
can be diffed between runs (our recreation test), compared before/after a
change, committed for assessment, and re-read by code without retyping.
It also records the environment it ran in, so the numbers are interpretable
later. Printing a summary is still useful for the human — the file is for
everything else.

**Would always predicting the majority class be a meaningful performance
baseline in TP02?**

Yes — as a *floor*, not a goal. With 73.5% `No`, the constant "always No"
model scores 73.5% accuracy, so any candidate must beat that to justify its
complexity. But it is meaningless as a *business* answer: it never flags a
single churner (recall on `Yes` = 0), and accuracy hides it because the
classes are imbalanced. This is the argument for also using precision /
recall / F1 on the `Yes` class in TP02.

## 5. Git workflow

**What does a commit identify?**

An immutable snapshot of the tracked tree plus its parent, author, message
and timestamp, identified by its SHA — a point you can return to or diff
against forever. It records *what changed and why*, not the time spent.

**What is still unrecorded if your working tree has edits?**

Everything not staged and committed: the edits exist only on your disk.
`git status`/`git diff` show them, but no other machine, no collaborator
and no recovery point has them — an un-pushed, un-committed change is one
`rm -rf` away from gone.

**What information belongs in a helpful PR description?**

What changed and why (intent, not just the diff), how it was verified (the
commands run and their results), what to look at closely, and open
questions for reviewers. Ours states the README/handover intent, cites the
clean-room reproduction test, and asks the reviewer about the blank
`TotalCharges` handling.

## 6. Handover

**What can another engineer now reproduce, and what evidence would still be
needed to reproduce a trained model?**

Reproducible today, from the README alone: the exact environment
(`uv sync --locked` → Python 3.12.11, pandas 3.1.0), the exact input data
(pinned URL + SHA-256 check), and the exact inspection report (one command;
before/after evidence in `evidence/tp01/`).

Not yet reproducible: a *trained model*. That would additionally need the
TP02 pipeline code, the training entry point, the recorded configuration
(`configs/train.json`), a pinned random seed, and a saved fitted pipeline
in `models/` with metadata linking it to a commit, a data hash and a run
report. A model artifact alone, without that provenance chain, is not
reproducible.
