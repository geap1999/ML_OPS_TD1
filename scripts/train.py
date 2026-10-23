import argparse
from pathlib import Path

from telco_churn.config import DEFAULT_CONFIG_PATH, TrainingConfig
from telco_churn.training import TrainingWorkflow


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Trains every candidate, selects one on validation and saves it.

    Usage (from the project root)::

        uv run python scripts/train.py [--config configs/train.json]

    Writes ``reports/model_comparison.{json,csv}`` and the artifact directory
    ``models/churn_pipeline/``. The test partition is never evaluated.

    Args:
        config_path: Training configuration file.

    Returns:
        Summary with comparison rows, decision and metadata path.
    """
    config = TrainingConfig.from_json(config_path)
    summary = TrainingWorkflow(config).run()
    metric = config.selection.metric
    print(f"Validation comparison (selection metric chosen in advance: {metric})")
    for row in summary["comparison"]["rows"]:
        print(
            f"  {row['model']:20s} AP={row['average_precision']:.4f} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} "
            f"acc={row['accuracy']:.4f} fit={row['fit_seconds']:.3f}s"
        )
    decision = summary["decision"]
    print(f"Selected: {decision.selected}. {decision.rationale}")
    print(f"Comparison: {config.outputs.comparison_json}")
    print(f"Artifact metadata: {summary['metadata_path']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP02: train, compare and save the pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    main(parser.parse_args().config)
