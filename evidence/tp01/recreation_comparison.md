# TP01 environment recreation evidence

Procedure:
1. Run 1: `uv run python scripts/inspect_data.py` in the original environment
   (report kept as `data_overview_run1_before_recreation.json`).
2. Set the environment aside (`mv .venv venv.bak`), no process was using it.
3. Recreate from the declared, locked dependencies: `uv sync --locked`.
4. Run 2: same command, same code, same original CSV
   (`reports/data_overview.json`, copied as `data_overview_run2_after_recreation.json`).

Comparison: all report content identical, except the run timestamp (expected).

- Interpreter: 3.12.11
- pandas: 3.1.0
- Source SHA-256 identical in both runs: True
- Data statistics (shape, target counts, duplicates, missing/whitespace):
  identical in both runs.
- Only the run timestamp differs, which is expected: it is not part of the
  reproducible content of the report.
