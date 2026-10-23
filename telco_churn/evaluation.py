from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics of one fitted model on one labelled dataset.

    Attributes:
        n_samples: Number of evaluated records.
        positive_rate: Share of true positives (``Churn = Yes``) in the data.
        score_type: Which model API produced the continuous scores.
        threshold: Score cut-off used to turn scores into class predictions.
        average_precision: Area under the precision-recall curve, from scores.
        roc_auc: Area under the ROC curve, from scores.
        accuracy: Share of correct class predictions.
        precision: Positive predictive value at ``threshold``.
        recall: True positive rate at ``threshold``.
        f1: Harmonic mean of precision and recall at ``threshold``.
        confusion_matrix: Counts with explicit label order and orientation.
    """

    n_samples: int
    positive_rate: float
    score_type: str
    threshold: float
    average_precision: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Converts the result to plain JSON-serialisable types.

        Returns:
            The metrics as a dictionary.
        """
        return asdict(self)


class ClassifierEvaluator:
    """Computes the agreed metric set for binary churn classifiers.

    Continuous positive-class scores feed ranking metrics (average precision,
    ROC AUC); thresholded class predictions feed accuracy, precision, recall,
    F1 and the confusion matrix. The positive class is ``1`` (``Churn = Yes``).

    Attributes:
        threshold: Score cut-off; a record is predicted positive when its
            score is greater than or equal to it.
        positive_class: Encoded label of the positive class.
    """

    def __init__(self, threshold: float = 0.5, positive_class: int = 1) -> None:
        """Initialises the evaluator.

        Args:
            threshold: Score cut-off for class predictions.
            positive_class: Encoded label of the positive class.
        """
        self.threshold = threshold
        self.positive_class = positive_class

    def positive_scores(self, model: Any, features: pd.DataFrame) -> tuple[np.ndarray, str]:
        """Returns continuous scores for the positive class.

        Uses ``predict_proba`` when available, picking the column whose entry
        in ``classes_`` is the positive class (never assuming it is the last
        column). Falls back to ``decision_function``, whose scores are *not*
        probabilities.

        Args:
            model: Fitted classifier or pipeline.
            features: Input records.

        Returns:
            Scores as a 1-D array and the name of the API that produced them.

        Raises:
            TypeError: If the model exposes neither scoring API.
        """
        if hasattr(model, "predict_proba"):
            column = list(model.classes_).index(self.positive_class)
            return model.predict_proba(features)[:, column], "predict_proba"
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(features)), "decision_function"
        raise TypeError(f"{type(model).__name__} exposes no predict_proba or decision_function.")

    def predict_labels(self, scores: np.ndarray) -> np.ndarray:
        """Applies the threshold policy to positive-class scores.

        Args:
            scores: Positive-class scores.

        Returns:
            Encoded class predictions (``1`` when ``score >= threshold``).
        """
        return (scores >= self.threshold).astype("int64")

    def evaluate(self, model: Any, features: pd.DataFrame, labels: pd.Series) -> EvaluationResult:
        """Scores a fitted model on labelled records.

        Args:
            model: Fitted classifier or pipeline.
            features: Input records.
            labels: Encoded true labels (``0``/``1``).

        Returns:
            The full metric set.
        """
        scores, score_type = self.positive_scores(model, features)
        predicted = self.predict_labels(scores)
        matrix = confusion_matrix(labels, predicted, labels=[0, self.positive_class])
        (tn, fp), (fn, tp) = matrix.tolist()
        return EvaluationResult(
            n_samples=len(labels),
            positive_rate=float(np.mean(labels == self.positive_class)),
            score_type=score_type,
            threshold=self.threshold,
            average_precision=float(average_precision_score(labels, scores)),
            roc_auc=float(roc_auc_score(labels, scores)),
            accuracy=float(accuracy_score(labels, predicted)),
            precision=float(precision_score(labels, predicted, zero_division=0)),
            recall=float(recall_score(labels, predicted, zero_division=0)),
            f1=float(f1_score(labels, predicted, zero_division=0)),
            confusion_matrix={
                "labels": [0, self.positive_class],
                "label_meaning": {"0": "No churn", "1": "Churn (positive class)"},
                "orientation": "rows = true label, columns = predicted label",
                "matrix": matrix.tolist(),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
        )
