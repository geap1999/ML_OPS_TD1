"""Terminal entry point for inference-only prediction (TP02 section 4).

Usage (from the project root, after training):

    uv run python scripts/predict.py

Loads ``models/selected_pipeline.joblib`` in this fresh process (no fit,
no labels, no training execution), predicts the small batch in
``models/inference_batch.csv`` and verifies agreement with the
predictions recorded at training time, writing ``reports/reload_check.json``.
"""

from pathlib import Path

from telco_churn.prediction import run_reload_check


def main(models_dir: Path | None = None) -> None:
    models_dir = models_dir or Path("models")
    check = run_reload_check(models_dir)
    status = "PASSED" if check["passed"] else "FAILED"
    print(
        f"Reload check {status}: {check['batch_size']} batch customers, "
        f"classes agree={check['class_predictions_agree']}, "
        f"scores agree within {check['score_tolerance']}={check['scores_agree']}"
    )
    print(f"Artifact SHA-256: {check['artifact_sha256']}")
    print("Evidence written to: reports/reload_check.json")
    if not check["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
