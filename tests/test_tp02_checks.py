"""TP02 section 5: focused reliability checks.

Each test maps to one required piece of evidence in the brief. These are
behavior checks, not model-quality tests: they verify the *machinery*
(allocations, learned statistics, no-refit, documented policies, visible
failures, save/reload) rather than the score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from telco_churn.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_pipeline,
    prepare_features,
)
from telco_churn.prediction import load_pipeline, predict_customers
from telco_churn.training import LABEL_MAP, make_splits, load_raw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


# --- Evidence 1: customer IDs do not overlap across splits -----------------


def test_customer_ids_disjoint_across_splits(split_frames):
    train_ids = set(split_frames["train"]["customerID"])
    valid_ids = set(split_frames["validation"]["customerID"])
    test_ids = set(split_frames["test"]["customerID"])
    assert not (train_ids & valid_ids)
    assert not (train_ids & test_ids)
    assert not (valid_ids & test_ids)


def test_recorded_allocation_matches_recreated_split(recorded_allocation, allocations):
    """The recorded seed-42 allocation is exactly reproducible."""
    for split_name in ("train", "validation", "test"):
        assert sorted(recorded_allocation[split_name]) == allocations[split_name]


def test_split_sizes_match_design(split_frames):
    sizes = {k: len(v) for k, v in split_frames.items()}
    total = sum(sizes.values())
    assert sizes["train"] / total == pytest.approx(0.6, abs=0.01)
    assert sizes["validation"] / total == pytest.approx(0.2, abs=0.01)
    assert sizes["test"] / total == pytest.approx(0.2, abs=0.01)


# --- Evidence 2: learned statistics come from training data only -----------


def test_imputer_statistic_is_training_median(fitted_logistic, split_frames):
    imputer = fitted_logistic.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    train_features = prepare_features(split_frames["train"])
    training_medians = train_features[NUMERIC_FEATURES].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, training_medians)


def test_full_dataset_median_would_differ__guards_against_leakage(split_frames):
    """The TotalCharges median over ALL rows differs from the train-only
    median (11 blank rows are spread across splits). If a fitted statistic
    ever equals the full-data value, preprocessing saw leakage data."""
    train_features = prepare_features(split_frames["train"])
    all_features = prepare_features(load_raw())
    train_median = train_features["TotalCharges"].median()
    full_median = all_features["TotalCharges"].median()
    assert train_median != full_median


# --- Evidence 3: prediction does not refit or change statistics ------------


def test_predict_leaves_imputer_statistics_unchanged(fitted_logistic, split_frames):
    imputer = fitted_logistic.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    before = imputer.statistics_.copy()
    batch = split_frames["validation"].head(10)
    fitted_logistic.predict(prepare_features(batch))
    np.testing.assert_array_equal(imputer.statistics_, before)


# --- Evidence 4: documented policies for missing numeric / unseen category --


def test_missing_numeric_follows_imputation_policy(fitted_logistic):
    """A blank TotalCharges is accepted and imputed with training median."""
    raw = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [70.35],
            "TotalCharges": ["   "],  # whitespace-only blank: allowed missing
            "Contract": ["Two year"],
            "InternetService": ["Fiber optic"],
            "PaymentMethod": ["Credit card (automatic)"],
        }
    )
    features = prepare_features(raw)
    assert np.isnan(features.loc[0, "TotalCharges"])  # blank -> missing, by policy
    prediction = fitted_logistic.predict(features)
    assert prediction[0] in (0, 1)  # pipeline handled it; no crash


def test_malformed_nonblank_numeric_fails_loudly():
    """Nonblank malformed text is never silently coerced to missing."""
    raw = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [70.35],
            "TotalCharges": ["seventy"],  # malformed nonblank -> must error
            "Contract": ["Two year"],
            "InternetService": ["Fiber optic"],
            "PaymentMethod": ["Credit card (automatic)"],
        }
    )
    with pytest.raises(ValueError):
        prepare_features(raw)


def test_unseen_category_ignored_but_flagged_as_limitation(fitted_logistic):
    """handle_unknown='ignore': unseen category -> all-zero encoding, no crash."""
    raw = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [70.35],
            "TotalCharges": [845.5],
            "Contract": ["Bespoke quantum plan"],  # never seen in training
            "InternetService": ["Fiber optic"],
            "PaymentMethod": ["Credit card (automatic)"],
        }
    )
    prediction = fitted_logistic.predict(prepare_features(raw))
    assert prediction[0] in (0, 1)
    # The caveat, stated in code: ignoring does NOT validate the category.
    # (Reliability for such records is a documented limitation, not proven.)


# --- Evidence 5: a missing required feature fails visibly ------------------


def test_missing_feature_fails_visibly():
    raw = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [70.35],
            # TotalCharges missing entirely
            "Contract": ["Two year"],
            "InternetService": ["Fiber optic"],
            "PaymentMethod": ["Credit card (automatic)"],
        }
    )
    with pytest.raises(KeyError, match="TotalCharges"):
        prepare_features(raw)


def test_unknown_extra_column_is_dropped_not_silently_used():
    """Input validation: extra columns are outside the contract -> dropped."""
    raw = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [70.35],
            "TotalCharges": [845.5],
            "Contract": ["Two year"],
            "InternetService": ["Fiber optic"],
            "PaymentMethod": ["Credit card (automatic)"],
            "Churn": ["Yes"],  # target column sneaking in: must not be used
        }
    )
    features = prepare_features(raw)
    assert "Churn" not in features.columns
    assert list(features.columns) == FEATURES  # exactly the six features


# --- Evidence 6: save/reload preserves predictions; inference needs no fit --


def test_reloaded_artifact_agrees_with_expected(tmp_path):
    """Fresh load from disk reproduces the training-time batch predictions."""
    pipeline = load_pipeline(MODELS_DIR / "selected_pipeline.joblib")
    batch = pd.read_csv(MODELS_DIR / "inference_batch.csv")
    expected = json.loads((MODELS_DIR / "inference_expected.json").read_text())

    result = predict_customers(pipeline, batch)
    assert result["class_predictions"] == expected["class_predictions"]
    assert all(
        abs(a - e) <= 1e-9
        for a, e in zip(result["positive_scores"], expected["positive_scores"])
    )


def test_inference_module_has_no_fit() -> None:
    """Static guard: prediction.py must contain no fit call."""
    source = Path(__file__).resolve().parents[1] / "telco_churn" / "prediction.py"
    code = source.read_text(encoding="utf-8")
    assert ".fit(" not in code
