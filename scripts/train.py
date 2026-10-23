"""Terminal entry point for TP02 training.

Usage (from the project root):

    uv run python scripts/train.py --milestone1   # first model + baseline
    uv run python scripts/train.py                # full comparison + artifact
"""

import argparse
import json
from pathlib import Path

from telco_churn.training import (
    DEFAULT_SOURCE,
    train_and_select,
    train_first_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--milestone1",
        action="store_true",
        help="Run only the first two-feature model + majority baseline.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the raw CSV (must match the documented SHA-256).",
    )
    args = parser.parse_args()

    if args.milestone1:
        results = train_first_model(source_path=args.source)
        for name, metrics in results["models"].items():
            cm = metrics["confusion_matrix"]
            print(
                f"{name}: accuracy={metrics['accuracy']:.4f} "
                f"AP={metrics['average_precision']:.4f} "
                f"recall@0.5={metrics['recall_at_threshold']:.4f} "
                f"confusion[[TN,FP],[FN,TP]]={cm}"
            )
        print("Full report: reports/first_model_metrics.json")
    else:
        summary = train_and_select(source_path=args.source)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
