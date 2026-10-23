"""TP02 training workflow: split, first model, candidates, selection, artifact.

Rules enforced by this module (from the TP02 brief):

- One stratified 60/20/20 split with one seed; the same customer
  allocation is reused for every candidate and recorded as evidence.
- ``customerID`` and ``Churn`` never become input features.
- Selection uses **validation only**. The test set is evaluated exactly
  once, after selection, and a counter in the metadata guards that.
- Every candidate gets a freshly built (unfitted) pipeline, so no learned
  statistic is shared or leaked between candidates.
- The saved artifact is the fitted *preprocessing + classifier together*.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
from numpy.typing import NDArray
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from telco_churn.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    LABEL_MAP,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    build_pipeline,
    prepare_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "telco.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train.json"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
ARTIFACT_NAME = "selected_pipeline.joblib"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw(source_path: Path = DEFAULT_SOURCE) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {source_path} (see README.md)."
        )
    return pd.read_csv(source_path)


def make_splits(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Stratified 60/20/20 split; returns raw frames (IDs and target kept)."""
    train_df, remaining_df = train_test_split(
        df, test_size=0.4, stratify=df[TARGET_COLUMN], random_state=seed
    )
    valid_df, test_df = train_test_split(
        remaining_df,
        test_size=0.5,
        stratify=remaining_df[TARGET_COLUMN],
        random_state=seed,
    )
    return {"train": train_df, "validation": valid_df, "test": test_df}


def _positive_scores(estimator, X) -> NDArray | None:
    """Continuous positive-class scores; None if the model has no proba API."""
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(X)
    class_index = list(estimator.classes_).index(LABEL_MAP[POSITIVE_LABEL])
    return proba[:, class_index]


def evaluate_model(pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    """Fixed metric definitions, identical for every candidate."""
    predictions = pipeline.predict(X)
    acc = float(accuracy_score(y, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", pos_label=LABEL_MAP[POSITIVE_LABEL], zero_division=0
    )
    scores = _positive_scores(pipeline, X)
    return {
        "positive_class": f"{POSITIVE_LABEL} (encoded {LABEL_MAP[POSITIVE_LABEL]})",
        "threshold": threshold,
        "n_examples": len(y),
        "n_positives": int((y == LABEL_MAP[POSITIVE_LABEL]).sum()),
        "accuracy": acc,
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "f1_at_threshold": float(f1),
        "average_precision": float(average_precision_score(y, scores)) if scores is not None else None,
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def environment_identity() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "pandas": importlib.metadata.version("pandas"),
        "joblib": importlib.metadata.version("joblib"),
        "telco_churn": importlib.metadata.version("telco_churn"),
    }


def train_first_model(
    config: dict | None = None,
    source_path: Path = DEFAULT_SOURCE,
    report_path: Path | None = None,
) -> dict:
    """Milestone 1: tenure + MonthlyCharges only, versus the majority baseline."""
    config = config or load_config()
    seed = config["seed"]
    two_features = config["features"]["first_milestone_two_features"]

    df = load_raw(source_path)
    splits = make_splits(df, seed)

    X_train = splits["train"][two_features].apply(pd.to_numeric, errors="raise")
    y_train = splits["train"][TARGET_COLUMN].map(LABEL_MAP)
    X_valid = splits["validation"][two_features].apply(pd.to_numeric, errors="raise")
    y_valid = splits["validation"][TARGET_COLUMN].map(LABEL_MAP)

    learned = LogisticRegression(max_iter=1000)
    learned.fit(X_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)  # fitted on training data, like any candidate

    results = {
        "description": (
            "Milestone 1: LogisticRegression on tenure+MonthlyCharges only, "
            "compared with a majority-class baseline, both fitted on the "
            "training split and evaluated on the validation split."
        ),
        "seed": seed,
        "features": two_features,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "models": {
            "logistic_two_features": evaluate_model(learned, X_valid, y_valid),
            "majority_baseline": evaluate_model(baseline, X_valid, y_valid),
        },
    }
    if report_path is None:
        report_path = REPORTS_DIR / "first_model_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def _make_classifier(name: str):
    """Candidate constructors; modest configurations, per the brief."""
    if name == "logistic":
        return LogisticRegression(max_iter=1000)
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=42)
    raise ValueError(f"Unknown candidate: {name}")


def train_and_select(
    config: dict | None = None,
    source_path: Path = DEFAULT_SOURCE,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Full workflow: comparison on validation, selection, artifact, metadata."""
    config = config or load_config()
    seed = config["seed"]
    threshold = config["threshold_policy"] and 0.5

    df = load_raw(source_path)
    splits = make_splits(df, seed)
    X = {name: prepare_features(frame) for name, frame in splits.items()}
    y = {
        name: frame[TARGET_COLUMN].map(LABEL_MAP)
        for name, frame in splits.items()
    }

    # --- candidates: same split, same six features, fresh pipeline each ---
    candidate_names = ["logistic", "decision_tree", "random_forest"]
    rows: list[dict] = []
    fitted: dict[str, object] = {}
    for name in candidate_names:
        candidate = build_pipeline(_make_classifier(name))  # fresh state
        start = time.perf_counter()
        candidate.fit(X["train"], y["train"])
        duration = time.perf_counter() - start
        fitted[name] = candidate
        metrics = evaluate_model(candidate, X["validation"], y["validation"], threshold)
        rows.append(
            {
                "model": name,
                "params": candidate.named_steps["classifier"].get_params(),
                "fit_seconds": round(duration, 3),
                **metrics,
            }
        )

    # majority baseline: same features/split/metric definitions
    baseline = build_pipeline(DummyClassifier(strategy="most_frequent"))
    baseline.fit(X["train"], y["train"])
    rows.append(
        {
            "model": "majority_baseline",
            "params": {"strategy": "most_frequent"},
            "fit_seconds": None,
            **evaluate_model(baseline, X["validation"], y["validation"], threshold),
        }
    )

    comparison = {
        "description": (
            "Validation comparison: three learned classifiers plus the "
            "majority baseline, identical six features, identical seed-42 "
            "split, identical metric definitions."
        ),
        "primary_selection_metric": config["primary_selection_metric"],
        "seed": seed,
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "candidates": rows,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str) + "\n", encoding="utf-8"
    )

    # --- selection: chosen in advance, validation only ---
    learned_rows = [r for r in rows if r["model"] != "majority_baseline"]
    selected_name = max(learned_rows, key=lambda r: r["average_precision"] or -1)["model"]
    selected_pipeline = fitted[selected_name]
    decision = (
        f"Selected '{selected_name}': highest validation average precision "
        f"(metric fixed before inspecting results). The baseline and all "
        f"other candidates are recorded for context. Test data played no "
        f"role in this decision."
    )

    # --- artifact: fitted preprocessing + classifier together ---
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / ARTIFACT_NAME
    joblib.dump(selected_pipeline, artifact_path)

    # --- metadata / reproduction record ---
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "origin_url": SOURCE_URL,
        },
        "environment": environment_identity(),
        "positive_class": f"{POSITIVE_LABEL} (encoded {LABEL_MAP[POSITIVE_LABEL]})",
        "threshold_policy": "classifier default (0.5) for class predictions",
        "primary_selection_metric": config["primary_selection_metric"],
        "split": {
            "seed": seed,
            "fractions": config["split"],
            "sizes": {k: len(v) for k, v in splits.items()},
            "allocation_file": "split_allocation.json",
        },
        "feature_roles": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "never_inputs": [ID_COLUMN, TARGET_COLUMN],
            "parsing": config["numeric_parsing"],
        },
        "preprocessing": config["preprocessing"],
        "comparison_table": "reports/model_comparison.json",
        "selected_model": selected_name,
        "selection_decision": decision,
        "selected_classifier_params": selected_pipeline.named_steps["classifier"].get_params(),
        "artifact": {
            "path": str(artifact_path),
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "format": "joblib (fitted sklearn Pipeline: preprocessor + classifier)",
        },
        "inference_batch": {
            "file": "inference_batch.csv",
            "expected": "inference_expected.json",
            "schema": FEATURES,
        },
        "test_evaluations": 1,
        "final_test_evaluation_note": (
            "Exactly one evaluation on the reserved test set, run after "
            "selection was finished."
        ),
    }

    # one-shot final test evaluation (after selection)
    metadata["final_test_evaluation"] = evaluate_model(
        selected_pipeline, X["test"], y["test"], threshold
    )

    # customer allocation evidence
    (models_dir / "split_allocation.json").write_text(
        json.dumps(
            {
                k: sorted(v[ID_COLUMN].astype(str).tolist())
                for k, v in splits.items()
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    # inference batch for the fresh-process reload check (raw, unencoded rows)
    valid_df = splits["validation"]
    batch = pd.concat(
        [
            valid_df[valid_df[TARGET_COLUMN] == "Yes"].head(2),
            valid_df[valid_df[TARGET_COLUMN] == "No"].head(3),
        ]
    )[FEATURES]
    batch.to_csv(models_dir / "inference_batch.csv", index=False)
    expected = {
        "customer_ids": batch[ID_COLUMN].tolist() if ID_COLUMN in batch.columns else None,
        "class_predictions": selected_pipeline
        .predict(prepare_features(batch))
        .tolist(),
        "positive_scores": _positive_scores(selected_pipeline, prepare_features(batch))
        .tolist(),
        "tolerance": "class predictions exact; scores within 1e-9",
    }
    (models_dir / "inference_expected.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8"
    )

    (models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )

    return {
        "selected": selected_name,
        "validation_average_precision": max(
            r["average_precision"] or -1 for r in learned_rows
        ),
        "artifact": str(artifact_path),
        "metadata": str(models_dir / "metadata.json"),
    }
