"""Unfitted pipeline construction and documented input parsing (TP02).

Responsibilities split deliberately:

- ``prepare_features`` is a **stateless boundary function**: it turns raw
  CSV-shaped rows into the documented six-column feature frame. It learns
  nothing, so it may run identically at training and inference time.
- ``build_pipeline`` returns an **unfitted** ``sklearn.pipeline.Pipeline``
  (``ColumnTransformer`` preprocessing branches + a classifier). It never
  reads data, trains, or caches state: call it once per candidate so every
  model gets fresh, unshared preprocessing statistics.

Feature roles and parsing rules are declared in ``configs/train.json`` and
mirrored here as the single implementation of that contract.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
LABEL_MAP = {"No": 0, "Yes": 1}
POSITIVE_LABEL = "Yes"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw customer rows into the documented six-feature frame.

    Documented, stateless rules (identical at training and inference):

    1. Require exactly the six feature columns (a missing feature is a
       visible ``KeyError``, never a silent wrong prediction).
    2. Strip surrounding whitespace on every value.
    3. Numeric columns: empty or whitespace-only values become missing
       (``NaN``) and are imputed later inside the pipeline; any *other*
       non-numeric text raises ``ValueError`` — malformed nonblank text is
       never silently coerced into an allowed missing value.
    4. Categorical columns stay as strings; missing categories may pass
       and are imputed later with the training-time strategy.

    ``customerID`` and ``Churn`` are deliberately not part of the output:
    they must never become model inputs.
    """
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required feature columns: {missing}. "
            "The pipeline consumes the documented six-column schema."
        )

    out = df[FEATURES].copy()
    for col in out.columns:
        out[col] = out[col].astype("string").str.strip()

    for col in NUMERIC_FEATURES:
        stripped = out[col]
        blank = stripped.eq("")
        stripped = stripped.mask(blank)  # blank -> <NA>
        # errors="raise": any remaining non-numeric text fails loudly.
        out[col] = pd.to_numeric(stripped, errors="raise").astype(float)
    return out


def build_pipeline(classifier) -> Pipeline:
    """Return an UNFITTED preprocessing+classifier pipeline.

    Branches (settings documented in configs/train.json):

    - numeric: median imputation (learned from training rows only), then
      ``StandardScaler``;
    - categorical: most-frequent imputation, then ``OneHotEncoder`` with
      ``handle_unknown="ignore"`` — a category never seen in training is
      encoded as all-zeros instead of raising, which is the documented
      policy for unseen categories (with the TP02 caveat that this does
      not make predictions *reliable* for such records).

    Fitting this object once trains the imputers, the scaler, the encoder
    vocabulary and the classifier in a single call. Statistics learned by
    ``fit`` are reused by ``predict``; validation and inference records
    are never used to refit any of them.
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
