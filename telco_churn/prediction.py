from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from telco_churn.artifacts import ArtifactStore, LoadedArtifact
from telco_churn.evaluation import ClassifierEvaluator
from telco_churn.features import FeatureSchema


class ChurnPredictor:
    """Inference-only wrapper around a saved, fitted churn pipeline.

    It never calls ``fit`` and never needs labels. Input handling is an
    explicit boundary: records are first checked against the documented
    schema and parsed with the same stateless rules as in training, then
    handed to the fitted pipeline, which reuses its learned statistics.

    Attributes:
        pipeline: Fitted preprocessing plus classifier.
        schema: Input contract recorded with the artifact.
        evaluator: Applies the recorded threshold to positive-class scores.
    """

    def __init__(self, pipeline: Pipeline, schema: FeatureSchema, threshold: float) -> None:
        """Initialises the predictor.

        Args:
            pipeline: Fitted preprocessing plus classifier.
            schema: Input columns and parsing rules.
            threshold: Probability cut-off for the churn class.
        """
        self.pipeline = pipeline
        self.schema = schema
        self.evaluator = ClassifierEvaluator(threshold=threshold)

    @classmethod
    def from_artifact(cls, directory: Path) -> ChurnPredictor:
        """Builds a predictor from a trusted artifact directory.

        Args:
            directory: Directory written by ``ArtifactStore.save``.

        Returns:
            A ready-to-use predictor.
        """
        return cls.from_loaded(ArtifactStore(directory).load())

    @classmethod
    def from_loaded(cls, artifact: LoadedArtifact) -> ChurnPredictor:
        """Builds a predictor from an already loaded artifact.

        Args:
            artifact: Pipeline and metadata.

        Returns:
            A ready-to-use predictor.
        """
        metadata = artifact.metadata
        features = metadata["features"]
        schema = FeatureSchema(
            id_column=features["id_column"],
            target_column=features["target_column"],
            positive_label=features["positive_class"]["label"],
            negative_label=features["negative_class"]["label"],
            numeric=tuple(features["numeric"]),
            categorical=tuple(features["categorical"]),
        )
        return cls(artifact.pipeline, schema, float(metadata["selection"]["threshold"]))

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        """Predicts churn for raw customer records.

        Args:
            records: One row per customer with the six input columns, as text
                or numbers. An ID column is passed through when present; a
                target column, if any, is ignored.

        Returns:
            Churn probability and predicted class (``1`` = churn) per record,
            with the customer ID first when available.

        Raises:
            KeyError: If a required input column is missing.
            ValueError: If a numeric input holds non-blank, non-numeric text.
        """
        features = self.schema.parse_features(records)
        scores, _ = self.evaluator.positive_scores(self.pipeline, features)
        result = pd.DataFrame(index=records.index)
        if self.schema.id_column in records.columns:
            result[self.schema.id_column] = records[self.schema.id_column]
        result["churn_probability"] = scores
        result["churn_prediction"] = self.evaluator.predict_labels(scores)
        return result
