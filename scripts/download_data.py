from pathlib import Path

from telco_churn.dataset import DEFAULT_SOURCE_PATH, EXPECTED_SHA256, download_raw


def main(destination: Path = DEFAULT_SOURCE_PATH) -> Path:
    """Fetches the original CSV and verifies its SHA-256, on any OS.

    Usage (from the project root, on Linux, macOS or Windows)::

        uv run python scripts/download_data.py

    Creates ``data/raw/`` if needed. All logic lives in
    ``telco_churn.dataset.download_raw``.

    Args:
        destination: Where to store the raw CSV.

    Returns:
        Path of the verified raw CSV.
    """
    written = download_raw(destination)
    print(f"Raw dataset: {written}")
    print(f"SHA-256 verified: {EXPECTED_SHA256}")
    return written


if __name__ == "__main__":
    main()
