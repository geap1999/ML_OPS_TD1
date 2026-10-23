"""Command-line entry point for the TP01 data inspection.

Usage (from the project root):

    uv run python scripts/inspect_data.py

Writes ``reports/data_overview.json`` from the immutable CSV in
``data/raw/telco.csv``. All logic lives in ``telco_churn.dataset``.
"""

from pathlib import Path

from telco_churn.dataset import (
    DEFAULT_REPORT_PATH,
    DEFAULT_SOURCE_PATH,
    build_report,
    write_report,
)


def main(
    source_path: Path = DEFAULT_SOURCE_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    report = build_report(source_path)
    written = write_report(report, report_path)
    data = report["data"]
    print(f"Source: {report['source']['path']}")
    print(f"SHA-256: {report['source']['sha256']}")
    print(f"Shape: {data['shape']['rows']} rows x {data['shape']['columns']} columns")
    print(f"Target counts: {data['target']['counts']}")
    print(f"Duplicate customer IDs: {data['duplicate_customer_ids']}")
    print(f"NaN values: {data['missing_values_isna'] or 'none'}")
    print(f"Empty/whitespace-only values: {data['empty_or_whitespace_only_values']}")
    print(f"Report written to: {written}")
    return written


if __name__ == "__main__":
    main()
