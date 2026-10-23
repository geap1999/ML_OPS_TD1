from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
from typing import Any
import warnings

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from telco_churn.provenance import RunProvenance
from telco_churn.reporting import ReportWriter


@dataclass(frozen=True)
class LoadedArtifact:
    """A fitted pipeline together with the metadata saved beside it.

    Attributes:
        pipeline: Fitted preprocessing plus classifier.
        metadata: Content of ``metadata.json``.
        directory: Artifact directory it was loaded from.
    """

    pipeline: Pipeline
    metadata: dict[str, Any]
    directory: Path

    def reference_inputs(self) -> pd.DataFrame:
        """Reads the raw input batch saved at training time.

        Returns:
            Raw records (ID plus input columns, no labels), all as text.
        """
        return pd.read_csv(self.directory / ArtifactStore.REFERENCE_INPUTS, dtype=str)

    def reference_predictions(self) -> pd.DataFrame:
        """Reads the predictions the in-memory pipeline made on that batch.

        Returns:
            Customer ID, churn probability and predicted class per record.
        """
        return pd.read_csv(self.directory / ArtifactStore.REFERENCE_PREDICTIONS)


class ArtifactStore:
    """Saves and loads a fitted pipeline and its reproduction record.

    Layout of an artifact directory:

    * ``pipeline.joblib``: the fitted preprocessing *and* classifier.
    * ``metadata.json``: data hash, seed, feature roles, preprocessing,
      parameters, metrics, threshold, code and environment identity, and the
      SHA-256 of every other file in the directory.
    * ``split_allocation.csv``: which customer went to which partition.
    * ``reference_inputs.csv`` / ``reference_predictions.csv``: a small raw
      batch and the predictions made before saving, used to prove that a
      reloaded pipeline predicts the same thing.

    Joblib files are pickles, which can execute code when loaded. ``load``
    therefore refuses a ``pipeline.joblib`` whose hash differs from the one
    recorded in ``metadata.json``. Only load artifacts from a trusted source.

    Attributes:
        directory: Artifact directory.
    """

    PIPELINE = "pipeline.joblib"
    METADATA = "metadata.json"
    ALLOCATION = "split_allocation.csv"
    REFERENCE_INPUTS = "reference_inputs.csv"
    REFERENCE_PREDICTIONS = "reference_predictions.csv"

    def __init__(self, directory: Path) -> None:
        """Initialises the store.

        Args:
            directory: Artifact directory to write to or read from.
        """
        self.directory = Path(directory)

    def save(
        self,
        pipeline: Pipeline,
        metadata: dict[str, Any],
        allocation: pd.DataFrame,
        reference_inputs: pd.DataFrame,
        reference_predictions: pd.DataFrame,
    ) -> Path:
        """Writes the fitted pipeline and its reproduction record.

        Args:
            pipeline: Fitted pipeline to persist.
            metadata: Run metadata; file hashes are added to it.
            allocation: Customer-to-partition table.
            reference_inputs: Raw batch (no labels) for the reload check.
            reference_predictions: Predictions of ``pipeline`` on that batch.

        Returns:
            Path of the written ``metadata.json``.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.directory / self.PIPELINE)
        ReportWriter.write_csv(self.directory / self.ALLOCATION, allocation)
        ReportWriter.write_csv(self.directory / self.REFERENCE_INPUTS, reference_inputs)
        ReportWriter.write_csv(self.directory / self.REFERENCE_PREDICTIONS, reference_predictions)
        files = [self.PIPELINE, self.ALLOCATION, self.REFERENCE_INPUTS, self.REFERENCE_PREDICTIONS]
        record = {
            **metadata,
            "artifact_files": {
                name: RunProvenance.sha256_file(self.directory / name) for name in files
            },
        }
        return ReportWriter.write_json(self.directory / self.METADATA, record)

    def load(self) -> LoadedArtifact:
        """Loads a trusted artifact after checking its integrity.

        Returns:
            The fitted pipeline and its metadata.

        Raises:
            FileNotFoundError: If the pipeline or metadata file is missing.
            ValueError: If ``pipeline.joblib`` does not match its recorded hash.
        """
        metadata = json.loads((self.directory / self.METADATA).read_text(encoding="utf-8"))
        pipeline_path = self.directory / self.PIPELINE
        expected = metadata["artifact_files"][self.PIPELINE]
        actual = RunProvenance.sha256_file(pipeline_path)
        if actual != expected:
            raise ValueError(
                f"{pipeline_path} has SHA-256 {actual}, but metadata records {expected}; "
                "refusing to unpickle an unverified file."
            )
        self._warn_on_version_drift(metadata)
        return LoadedArtifact(
            pipeline=joblib.load(pipeline_path), metadata=metadata, directory=self.directory
        )

    @staticmethod
    def _warn_on_version_drift(metadata: dict[str, Any]) -> None:
        """Warns when installed libraries differ from the training environment.

        Args:
            metadata: Artifact metadata with ``environment.packages``.
        """
        recorded = metadata.get("environment", {}).get("packages", {})
        for name in ("scikit-learn", "numpy", "scipy", "joblib"):
            installed = importlib.metadata.version(name)
            if name in recorded and recorded[name] != installed:
                warnings.warn(
                    f"{name} {installed} is installed but the artifact was trained with "
                    f"{recorded[name]}; predictions may differ or loading may fail.",
                    stacklevel=3,
                )
