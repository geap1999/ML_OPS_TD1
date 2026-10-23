"""Reading and inspection helpers for the Telco churn customer snapshot (TP01).

This module only *inspects* data: it never cleans, splits, imputes or trains.
The raw CSV under ``data/raw/`` is treated as immutable input.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys

import pandas as pd

# Commit-pinned download URL of the original snapshot (see references/data_source.md).
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "telco.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "data_overview.json"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"


def sha256sum(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_raw(source_path: Path = DEFAULT_SOURCE_PATH) -> pd.DataFrame:
    """Read the original CSV unchanged; never mutate the file on disk."""
    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {source_path}. "
            "Download the commit-pinned CSV into data/raw/ (see README.md)."
        )
    return pd.read_csv(source_path)


def _string_columns(df: pd.DataFrame) -> list[str]:
    """String-typed columns for both pandas 2 (object) and pandas 3 (str dtype)."""
    return [
        name for name in df.columns if pd.api.types.is_string_dtype(df[name])
    ]


def check_empty_strings(df: pd.DataFrame) -> dict[str, int]:
    """Count values that are empty or whitespace-only, per column.

    This is deliberately different from ``isna()``: pandas only turns truly
    empty fields into NaN, so a field containing a single space stays a
    non-null string. Both views are reported side by side.
    """
    counts: dict[str, int] = {}
    for name in _string_columns(df):
        stripped = df[name].astype("string").str.strip()
        counts[name] = int(stripped.eq("").sum())
    return counts


def check_missing(df: pd.DataFrame) -> dict[str, int]:
    """Count pandas NaN values, per column."""
    counts = df.isna().sum()
    return {name: int(n) for name, n in counts.items() if n > 0}


def check_target(df: pd.DataFrame) -> dict:
    """Raw counts and proportions of the churn labels (incl. any missing)."""
    counts = df[TARGET_COLUMN].value_counts(dropna=False)
    proportions = df[TARGET_COLUMN].value_counts(dropna=False, normalize=True)
    return {
        "counts": {str(label): int(n) for label, n in counts.items()},
        "proportions": {
            str(label): float(p) for label, p in proportions.items()
        },
    }


def environment_info() -> dict:
    """Interpreter and library versions needed to interpret a report run."""
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": importlib.metadata.version("pandas"),
        "telco_churn_version": importlib.metadata.version("telco_churn"),
    }


def build_report(source_path: Path = DEFAULT_SOURCE_PATH) -> dict:
    """Compute every TP01 inspection check for one run over the raw CSV."""
    df = read_raw(source_path)

    return {
        "run": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "command": "uv run python scripts/inspect_data.py",
        },
        "source": {
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "sha256": sha256sum(source_path),
            "origin_url": SOURCE_URL,
            "note": "Original snapshot; never modified locally.",
        },
        "environment": environment_info(),
        "data": {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "dtypes": {name: str(dtype) for name, dtype in df.dtypes.items()},
            "target": check_target(df),
            "duplicate_customer_ids": int(df[ID_COLUMN].duplicated().sum()),
            "missing_values_isna": check_missing(df),
            "empty_or_whitespace_only_values": check_empty_strings(df),
        },
    }


def write_report(
    report: dict,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Save the report as machine-readable JSON (create parents as needed)."""
    import json

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_path
