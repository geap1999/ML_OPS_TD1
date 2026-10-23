"""Inference-only loading and prediction (TP02 section 4).

This module never fits anything. It loads the trusted fitted artifact
(preprocessing + classifier together), consumes the documented six-column
schema through the same stateless boundary used at training time, and
proves on a small batch that the saved artifact reproduces the original
in-process predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from telco_churn.pipeline import FEATURES, prepare_features

BATCH_COLUMNS = FEATURES  # documented schema: six feature columns, no Churn


def load_pipeline(path: Path):
    """Load a trusted fitted pipeline. Trained artifacts are code+data; only
    load files you produced or trust. Returns the fitted sklearn Pipeline."""
    return joblib.load(path)


def predict_customers(pipeline, customers: pd.DataFrame) -> dict:
    """Predict from raw, unencoded rows.

    ``customers`` must contain the six documented feature columns (original
    input types: numbers as text or numbers, categories as strings; blank
    numerics are allowed and follow the imputation policy). Validation of
    the schema and numeric parsing happens in ``prepare_features``.
    """
    features = prepare_features(customers[BATCH_COLUMNS])
    predictions = pipeline.predict(features)
    result = {"class_predictions": [int(p) for p in predictions]}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(features)
        class_index = list(pipeline.classes_).index(1)
        result["positive_scores"] = [float(s) for s in proba[:, class_index]]
    return result


def run_reload_check(
    models_dir: Path,
    report_path: Path | None = None,
) -> dict:
    """Fresh-process evidence: loaded artifact == original predictions."""
    artifact_path = models_dir / "selected_pipeline.joblib"
    batch_path = models_dir / "inference_batch.csv"
    expected_path = models_dir / "inference_expected.json"

    pipeline = load_pipeline(artifact_path)  # no fit anywhere in this module
    batch = pd.read_csv(batch_path)
    actual = predict_customers(pipeline, batch)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    classes_agree = actual["class_predictions"] == expected["class_predictions"]
    scores_agree = all(
        abs(a - e) <= 1e-9
        for a, e in zip(actual["positive_scores"], expected["positive_scores"])
    )
    check = {
        "description": (
            "Fresh-process reload check: predictions from the loaded "
            "artifact agree with the in-process predictions recorded at "
            "training time; no fit call occurs in this process."
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "batch_size": len(batch),
        "class_predictions_agree": classes_agree,
        "score_tolerance": 1e-9,
        "scores_agree": scores_agree,
        "passed": bool(classes_agree and scores_agree),
        "actual": actual,
    }
    if report_path is None:
        report_path = Path("reports") / "reload_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    return check
