import argparse
from pathlib import Path
import sys

from telco_churn.config import DEFAULT_CONFIG_PATH, TrainingConfig
from telco_churn.registry import RegisteredModelLoader, RegistryVerifier
from telco_churn.reporting import ReportWriter


def main(tracking_uri: str, model_uri: str, report_path: Path) -> int:
    """Reloads the registered pipeline in this fresh process and checks its predictions.

    Usage (from the project root, with the tracking server running)::

        uv run python scripts/verify_registered.py

    No training code runs and no labels are read: ``fit`` is disabled on the
    loaded pipeline before it predicts the reference batch logged by its
    source run. The result is compared with the predictions the in-memory
    pipeline made before it was logged.

    Args:
        tracking_uri: Address of the tracking server.
        model_uri: Registry URI of the model to load.
        report_path: Where to write the evidence JSON.

    Returns:
        Process exit code: ``0`` if every check passed, ``1`` otherwise.
    """
    record = RegistryVerifier(RegisteredModelLoader(tracking_uri)).verify(model_uri)
    ReportWriter.write_json(report_path, record)
    print(
        f"Loaded {record['requested_uri']} -> {record['registered_model']} "
        f"version {record['version']}"
    )
    print(f"Source run: {record['source_run_id']}; logged model: {record['logged_model_uri']}")
    for name, passed in record["checks"].items():
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    print(
        f"Max |probability difference| = {record['max_abs_probability_difference']:.3g} "
        f"(tolerance {record['tolerance_abs_probability']:g}) over {record['n_records']} records"
    )
    print(f"Evidence written to: {report_path}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    tracking = TrainingConfig.from_json(DEFAULT_CONFIG_PATH).tracking
    parser = argparse.ArgumentParser(description="Fresh-process reload check via the registry.")
    parser.add_argument("--tracking-uri", default=tracking.tracking_uri)
    parser.add_argument("--model-uri", default=tracking.alias_uri)
    parser.add_argument("--report", type=Path, default=tracking.reload_report)
    arguments = parser.parse_args()
    sys.exit(main(arguments.tracking_uri, arguments.model_uri, arguments.report))
