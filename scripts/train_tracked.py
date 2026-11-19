import argparse
from pathlib import Path

from telco_churn.config import DEFAULT_CONFIG_PATH, TrainingConfig
from telco_churn.tracking import TrackedTrainingWorkflow


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Trains every candidate in its own MLflow run, then registers the selected one.

    Usage (from the project root, with the tracking server running)::

        uv run python scripts/train_tracked.py [--config configs/train.json]

    Each candidate gets one run with its parameters, validation metrics,
    tags, artifacts, dataset inputs and complete fitted pipeline. The
    selected pipeline is registered and given the configured alias. The test
    partition is never evaluated.

    Args:
        config_path: Training configuration file with a ``tracking`` section.

    Returns:
        The record written to ``tracking.run_report``.
    """
    config = TrainingConfig.from_json(config_path)
    record = TrackedTrainingWorkflow(config).run()
    print(f"Experiment: {record['experiment']['name']} (id {record['experiment']['id']})")
    print(f"Git commit: {record['git_commit']} (uncommitted changes: {record['git_dirty']})")
    for run in record["runs"]:
        print(
            f"  {run['run_name']:24s} run {run['run_id']} "
            f"AP={run['val_average_precision']:.4f} P={run['val_precision']:.4f} "
            f"R={run['val_recall']:.4f} F1={run['val_f1']:.4f}"
        )
    selected = record["selected"]
    print(f"Selected: {selected['run_name']}. {record['selection']['decision']['rationale']}")
    print(f"Run ID: {selected['run_id']}")
    print(f"Model URI: {selected['model_uri']}")
    print(
        f"Registered: {selected['registered_model']} version {selected['version']}, "
        f"alias '{selected['alias']}' ({selected['alias_uri']})"
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP03: tracked training and registration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    main(parser.parse_args().config)
