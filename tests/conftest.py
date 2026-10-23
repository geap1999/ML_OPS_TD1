"""Shared pytest fixtures: the seed-42 customer allocations and split frames.

The split is created ONCE per session with the same code path as training
(`train_test_split`, stratified, seed 42) so every focused check in TP02
section 5 runs against the actual allocation, not a re-derived one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from telco_churn.pipeline import build_pipeline, prepare_features
from telco_churn.training import (
    LABEL_MAP,
    TARGET_COLUMN,
    load_raw,
    make_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def split_frames(raw_df) -> dict[str, pd.DataFrame]:
    return make_splits(raw_df, seed=42)


@pytest.fixture(scope="session")
def allocations(split_frames) -> dict[str, list[str]]:
    return {k: sorted(v["customerID"].astype(str).tolist()) for k, v in split_frames.items()}


@pytest.fixture(scope="session")
def recorded_metadata() -> dict:
    return json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def recorded_allocation(allocations) -> dict[str, list[str]]:
    """The allocation recorded at training time (models/split_allocation.json)."""
    return json.loads((MODELS_DIR / "split_allocation.json").read_text(encoding="utf-8"))


_FITTED_CACHE: dict[str, object] = {}


@pytest.fixture(scope="session")
def fitted_logistic():
    """One fitted logistic pipeline for the statistics/policy checks.

    Session-scoped on purpose: the 'prediction does not refit' check needs
    a fit to have happened exactly once, torn down with the session.
    """
    if "logistic" not in _FITTED_CACHE:
        frames = make_splits(load_raw(), seed=42)
        pipeline = build_pipeline(__import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(max_iter=1000))
        pipeline.fit(prepare_features(frames["train"]), frames["train"][TARGET_COLUMN].map(LABEL_MAP))
        _FITTED_CACHE["logistic"] = pipeline
    return _FITTED_CACHE["logistic"]
