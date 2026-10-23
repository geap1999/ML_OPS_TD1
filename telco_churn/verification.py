from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from telco_churn.artifacts import ArtifactStore
from telco_churn.pipeline import PipelineInspector
from telco_churn.prediction import ChurnPredictor
from telco_churn.provenance import RunProvenance


class FitCalledError(RuntimeError):
    """Raised when inference code tries to train the loaded pipeline."""


class ArtifactVerifier:
    """Proves that a saved pipeline predicts like the in-memory original.

    Meant to run in a fresh Python process with no access to training code
    paths or labels. It loads the trusted artifact, disables ``fit`` on the
    pipeline and on every step, predicts the saved raw reference batch, and
    compares the result with the predictions recorded before saving. It also
    checks that prediction leaves the learned preprocessing statistics
    unchanged.

    Attributes:
        directory: Artifact directory.
        tolerance: Maximum absolute probability difference accepted.
    """

    def __init__(self, directory: Path, tolerance: float = 1e-12) -> None:
        """Initialises the verifier.

        Args:
            directory: Artifact directory.
            tolerance: Maximum absolute probability difference accepted.
        """
        self.directory = Path(directory)
        self.tolerance = tolerance

    def verify(self) -> dict[str, Any]:
        """Runs the reload check.

        Returns:
            Evidence record; ``passed`` is ``True`` only if every check holds.
        """
        artifact = ArtifactStore(self.directory).load()
        self._forbid_fit(artifact.pipeline)
        inputs = artifact.reference_inputs()
        expected = artifact.reference_predictions()
        target = artifact.metadata["features"]["target_column"]

        before = PipelineInspector.learned_statistics(artifact.pipeline)
        actual = ChurnPredictor.from_loaded(artifact).predict(inputs)
        after = PipelineInspector.learned_statistics(artifact.pipeline)

        difference = np.abs(
            actual["churn_probability"].to_numpy() - expected["churn_probability"].to_numpy()
        )
        checks = {
            "labels_absent_from_inputs": target not in inputs.columns,
            "same_customer_order": actual.iloc[:, 0].tolist() == expected.iloc[:, 0].tolist(),
            "probabilities_within_tolerance": bool(difference.max() <= self.tolerance),
            "class_predictions_identical": bool(
                (
                    actual["churn_prediction"].to_numpy()
                    == expected["churn_prediction"].to_numpy()
                ).all()
            ),
            "learned_statistics_unchanged_by_predict": before == after,
            "fit_disabled_during_inference": True,
        }
        return {
            "checked_utc": RunProvenance.timestamp(),
            "artifact": self.directory.name,
            "pipeline_sha256": artifact.metadata["artifact_files"][ArtifactStore.PIPELINE],
            "n_records": len(inputs),
            "tolerance_abs_probability": self.tolerance,
            "max_abs_probability_difference": float(difference.max()),
            "checks": checks,
            "passed": all(checks.values()),
        }

    @staticmethod
    def _forbid_fit(pipeline: Any) -> None:
        """Replaces every ``fit``-like method with one that raises.

        Args:
            pipeline: Loaded pipeline, modified in place for this process only.
        """

        def refuse(*_args: Any, **_kwargs: Any) -> None:
            raise FitCalledError("fit was called during inference.")

        estimators = [pipeline, *(step for _, step in pipeline.steps)]
        for estimator in estimators:
            for method in ("fit", "fit_transform", "partial_fit"):
                if hasattr(estimator, method):
                    setattr(estimator, method, refuse)
