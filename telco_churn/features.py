from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSchema:
    """Roles of the dataset columns and the parsing rules for model inputs.

    This is the single place that knows which columns are inputs, identifiers
    or the target, and how raw values are parsed before they reach the
    scikit-learn pipeline. Parsing is *stateless*: it never learns anything
    from data, so applying it before the pipeline cannot leak information
    between splits, and the same rules run at training and inference time.

    Parsing rules (documented contract):

    * Numeric inputs become ``float64``. Empty or whitespace-only text becomes
      a missing value (``NaN``), which the pipeline imputes. Any other text
      that is not a number raises ``ValueError``; it is never silently turned
      into a missing value.
    * Categorical inputs become ``object`` columns of Python strings, with
      surrounding whitespace stripped. Empty or whitespace-only text becomes
      missing (``NaN``), which the pipeline imputes.
    * A required input column that is absent raises ``KeyError``.

    Attributes:
        id_column: Customer identifier; kept for split checks, never an input.
        target_column: Raw label column.
        positive_label: Raw label encoded as class ``1``.
        negative_label: Raw label encoded as class ``0``.
        numeric: Numeric input columns, in pipeline order.
        categorical: Categorical input columns, in pipeline order.
    """

    id_column: str
    target_column: str
    positive_label: str
    negative_label: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    def __post_init__(self) -> None:
        """Rejects schemas that would feed the ID or the target to the model.

        Raises:
            ValueError: If the ID or target column is listed as an input, or an
                input is listed twice.
        """
        inputs = self.feature_columns
        if self.id_column in inputs or self.target_column in inputs:
            raise ValueError("The ID and target columns must never be model inputs.")
        if len(set(inputs)) != len(inputs):
            raise ValueError(f"Duplicate input columns in schema: {inputs}.")

    @classmethod
    def from_config(cls, features: dict[str, Any]) -> FeatureSchema:
        """Builds a schema from the ``features`` section of the configuration.

        Args:
            features: Mapping with ``id_column``, ``target_column``,
                ``positive_label``, ``negative_label``, ``numeric`` and
                ``categorical`` keys.

        Returns:
            The corresponding schema.
        """
        return cls(
            id_column=features["id_column"],
            target_column=features["target_column"],
            positive_label=features["positive_label"],
            negative_label=features["negative_label"],
            numeric=tuple(features["numeric"]),
            categorical=tuple(features["categorical"]),
        )

    @property
    def feature_columns(self) -> list[str]:
        """Input columns in the order the pipeline expects them.

        Returns:
            Numeric columns followed by categorical columns.
        """
        return [*self.numeric, *self.categorical]

    def subset(self, columns: list[str]) -> FeatureSchema:
        """Returns a schema restricted to some of the input columns.

        Used by the milestone-1 model, which only sees two numeric inputs.

        Args:
            columns: Input columns to keep; each must belong to this schema.

        Returns:
            A schema with the same roles and only the requested inputs.

        Raises:
            ValueError: If a requested column is not an input of this schema.
        """
        unknown = sorted(set(columns) - set(self.feature_columns))
        if unknown:
            raise ValueError(f"Columns {unknown} are not inputs of this schema.")
        return FeatureSchema(
            id_column=self.id_column,
            target_column=self.target_column,
            positive_label=self.positive_label,
            negative_label=self.negative_label,
            numeric=tuple(c for c in self.numeric if c in columns),
            categorical=tuple(c for c in self.categorical if c in columns),
        )

    def validate_columns(self, frame: pd.DataFrame) -> None:
        """Fails loudly when a required input column is absent.

        Args:
            frame: Candidate input records.

        Raises:
            KeyError: If one or more input columns are missing.
        """
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise KeyError(
                f"Missing required input column(s) {missing}; expected {self.feature_columns}."
            )

    def parse_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Applies the documented stateless parsing rules to the input columns.

        The input frame is never modified; a new frame holding only the input
        columns (in schema order) is returned.

        Args:
            frame: Raw records, e.g. straight from ``pd.read_csv``. Extra
                columns such as the ID or the target are ignored.

        Returns:
            Parsed input columns with the documented dtypes.

        Raises:
            KeyError: If a required input column is missing.
            ValueError: If a numeric column contains non-blank text that is not
                a number.
        """
        self.validate_columns(frame)
        parsed = pd.DataFrame(index=frame.index)
        for column in self.numeric:
            parsed[column] = self._parse_numeric(frame[column], column)
        for column in self.categorical:
            parsed[column] = self._parse_categorical(frame[column])
        return parsed

    def encode_target(self, labels: pd.Series) -> pd.Series:
        """Encodes raw churn labels as integers (positive label -> ``1``).

        Args:
            labels: Raw target values.

        Returns:
            Integer labels ``0``/``1`` with the same index.

        Raises:
            ValueError: If a label is neither the positive nor the negative one.
        """
        mapping = {self.negative_label: 0, self.positive_label: 1}
        unexpected = sorted(set(labels.dropna().unique()) - set(mapping))
        if unexpected or labels.isna().any():
            raise ValueError(
                f"Unexpected target values {unexpected} (or missing labels); "
                f"allowed: {list(mapping)}."
            )
        return labels.map(mapping).astype("int64")

    def describe(self) -> dict[str, Any]:
        """Serialisable description of roles and input types for metadata.

        Returns:
            Feature roles, positive-class definition and the input contract.
        """
        return {
            "id_column": self.id_column,
            "target_column": self.target_column,
            "positive_class": {"label": self.positive_label, "encoded_as": 1},
            "negative_class": {"label": self.negative_label, "encoded_as": 0},
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "input_contract": {
                "numeric": "float64; blank text -> NaN (imputed); other non-numeric text -> error",
                "categorical": "string; blank -> NaN (imputed); unseen category -> all-zero "
                "one-hot (ignored)",
                "missing_column": "KeyError",
            },
        }

    @staticmethod
    def _parse_numeric(values: pd.Series, column: str) -> pd.Series:
        """Converts one numeric column, separating blanks from malformed text.

        ``pd.to_numeric(errors="raise")`` would reject the legitimate blanks,
        and ``errors="coerce"`` alone would hide malformed text as NaN. Blanks
        are therefore identified first, then coercion failures among the
        remaining non-blank values are reported as errors.

        Args:
            values: Raw column values.
            column: Column name, for error messages.

        Returns:
            The column as ``float64`` with blanks as ``NaN``.

        Raises:
            ValueError: If non-blank values cannot be parsed as numbers.
        """
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            return values.astype("float64")
        text = values.astype("string").str.strip()
        text = text.mask(text.eq(""))
        converted = pd.to_numeric(text, errors="coerce")
        malformed = converted.isna() & text.notna()
        if malformed.any():
            examples = text[malformed].unique()[:5].tolist()
            raise ValueError(
                f"Column {column!r} has {int(malformed.sum())} non-blank value(s) that are "
                f"not numbers, e.g. {examples}."
            )
        return converted.astype("float64")

    @staticmethod
    def _parse_categorical(values: pd.Series) -> pd.Series:
        """Normalises one categorical column to stripped strings or ``NaN``.

        Args:
            values: Raw column values.

        Returns:
            An ``object`` column of strings, with blanks as ``NaN``.
        """
        text = values.astype("string").str.strip()
        text = text.mask(text.eq(""))
        return text.astype(object).where(text.notna(), np.nan)
