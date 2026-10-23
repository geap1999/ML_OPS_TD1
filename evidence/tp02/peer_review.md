# TP02 peer review record

**Status: not done yet.** The brief requires a peer to reproduce training,
or load the trusted artifact, using only the README, and to record one issue
and its resolution. This file is filled in after that exchange. The steps
below say what the peer should run.

## What the peer runs (README only, no oral help)

1. Fresh clone, then `uv sync --locked`.
2. `uv run python scripts/download_data.py`.
3. Either reproduce training with `uv run python scripts/train.py`, or load
   the committed artifact with `uv run python scripts/verify_artifact.py` and
   `uv run python scripts/predict.py --input evidence/tp02/example_new_customers.csv`.
4. `uv run pytest`.

## Record

- Peer:
- Date:
- Path taken (retrain / load artifact):
- Result:
- Issue found:
- Resolution:
