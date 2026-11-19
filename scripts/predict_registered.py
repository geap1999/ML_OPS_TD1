import argparse
from pathlib import Path

import pandas as pd

from telco_churn.config import DEFAULT_CONFIG_PATH, TrainingConfig
from telco_churn.registry import RegisteredModelLoader


def main(
    input_csv: Path, output_csv: Path | None, tracking_uri: str, model_uri: str
) -> pd.DataFrame:
    """Predicts churn with the registered pipeline, loaded through its alias.

    Usage (from the project root, with the tracking server running)::

        uv run python scripts/predict_registered.py --input new_customers.csv

    Inference only: nothing here imports training code or calls ``fit``. The
    input CSV needs the six documented input columns; ``customerID`` is
    passed through when present and ``Churn`` is not required. Values are
    read as text and parsed by the documented input contract.

    Args:
        input_csv: Customers to score.
        output_csv: Where to write predictions; printed only when ``None``.
        tracking_uri: Address of the tracking server.
        model_uri: Registry URI of the model to load.

    Returns:
        Predictions per customer.
    """
    model = RegisteredModelLoader(tracking_uri).load(model_uri)
    version = model.version
    print(f"Loaded {model_uri} -> {version.name} version {version.version}")
    print(f"Source run: {version.run_id}; logged model: {version.source}")
    records = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    predictions = model.predictor().predict(records)
    if output_csv is None:
        print(predictions.to_string(index=False))
    else:
        predictions.to_csv(output_csv, index=False, lineterminator="\n")
        print(f"{len(predictions)} predictions written to: {output_csv}")
    return predictions


if __name__ == "__main__":
    tracking = TrainingConfig.from_json(DEFAULT_CONFIG_PATH).tracking
    parser = argparse.ArgumentParser(description="Predict churn with the registered pipeline.")
    parser.add_argument("--input", type=Path, required=True, help="CSV of customers to score")
    parser.add_argument("--output", type=Path, default=None, help="CSV to write predictions to")
    parser.add_argument("--tracking-uri", default=tracking.tracking_uri)
    parser.add_argument("--model-uri", default=tracking.alias_uri)
    arguments = parser.parse_args()
    main(arguments.input, arguments.output, arguments.tracking_uri, arguments.model_uri)
