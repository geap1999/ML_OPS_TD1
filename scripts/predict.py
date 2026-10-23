import argparse
from pathlib import Path

import pandas as pd

from telco_churn.config import DEFAULT_CONFIG_PATH, TrainingConfig
from telco_churn.prediction import ChurnPredictor


def main(input_csv: Path, output_csv: Path | None, artifact_dir: Path) -> pd.DataFrame:
    """Predicts churn for new customers with the saved pipeline (no training).

    Usage (from the project root)::

        uv run python scripts/predict.py --input new_customers.csv [--output preds.csv]

    The input CSV needs the six documented input columns; ``customerID`` is
    passed through when present and ``Churn`` is not required. Values are read
    as text and parsed by the documented input contract.

    Args:
        input_csv: Customers to score.
        output_csv: Where to write predictions; printed only when ``None``.
        artifact_dir: Trusted artifact directory.

    Returns:
        Predictions per customer.
    """
    records = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    predictions = ChurnPredictor.from_artifact(artifact_dir).predict(records)
    if output_csv is None:
        print(predictions.to_string(index=False))
    else:
        predictions.to_csv(output_csv, index=False, lineterminator="\n")
        print(f"{len(predictions)} predictions written to: {output_csv}")
    return predictions


if __name__ == "__main__":
    default_artifact = TrainingConfig.from_json(DEFAULT_CONFIG_PATH).outputs.artifact_dir
    parser = argparse.ArgumentParser(description="Predict churn with the saved pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="CSV of customers to score")
    parser.add_argument("--output", type=Path, default=None, help="CSV to write predictions to")
    parser.add_argument("--artifact", type=Path, default=default_artifact)
    arguments = parser.parse_args()
    main(arguments.input, arguments.output, arguments.artifact)
