import argparse
from pathlib import Path
import sys

from telco_churn.config import DEFAULT_CONFIG_PATH, TrainingConfig
from telco_churn.dataset import PROJECT_ROOT
from telco_churn.reporting import ReportWriter
from telco_churn.verification import ArtifactVerifier

DEFAULT_REPORT = PROJECT_ROOT / "reports" / "reload_check.json"


def main(artifact_dir: Path, report_path: Path = DEFAULT_REPORT) -> int:
    """Reloads the saved pipeline in this fresh process and checks its predictions.

    Usage (from the project root, after ``scripts/train.py``)::

        uv run python scripts/verify_artifact.py

    No training code runs and no labels are read: ``fit`` is disabled on the
    loaded pipeline before predicting the saved reference batch.

    Args:
        artifact_dir: Trusted artifact directory.
        report_path: Where to write the evidence JSON.

    Returns:
        Process exit code: ``0`` if every check passed, ``1`` otherwise.
    """
    record = ArtifactVerifier(artifact_dir).verify()
    ReportWriter.write_json(report_path, record)
    for name, passed in record["checks"].items():
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    print(
        f"Max |probability difference| = {record['max_abs_probability_difference']:.3g} "
        f"(tolerance {record['tolerance_abs_probability']:g}) over {record['n_records']} records"
    )
    print(f"Evidence written to: {report_path}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    default_artifact = TrainingConfig.from_json(DEFAULT_CONFIG_PATH).outputs.artifact_dir
    parser = argparse.ArgumentParser(description="Fresh-process reload check of the artifact.")
    parser.add_argument("--artifact", type=Path, default=default_artifact)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    sys.exit(main(arguments.artifact, arguments.report))
