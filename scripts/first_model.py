import argparse
from pathlib import Path

from telco_churn.config import DEFAULT_CONFIG_PATH, TrainingConfig
from telco_churn.training import FirstModelExperiment


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Runs milestone 1: a two-feature model and a majority baseline.

    Usage (from the project root)::

        uv run python scripts/first_model.py [--config configs/train.json]

    Args:
        config_path: Training configuration file.

    Returns:
        The report written to ``reports/first_model.json``.
    """
    config = TrainingConfig.from_json(config_path)
    report = FirstModelExperiment(config).run()
    print(f"Features: {report['features']}  (fitted on train, evaluated on validation)")
    for name, metrics in report["models"].items():
        matrix = metrics["confusion_matrix"]
        print(
            f"{name:20s} accuracy={metrics['accuracy']:.4f} recall={metrics['recall']:.4f} "
            f"confusion [labels 0=No,1=Yes; rows=true] {matrix['matrix']}"
        )
    print(f"Report written to: {config.outputs.first_model_report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP02 milestone 1: first model + baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    main(parser.parse_args().config)
