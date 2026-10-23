from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from telco_churn.artifacts import ArtifactStore
from telco_churn.config import EstimatorConfig, SelectionConfig, TrainingConfig
from telco_churn.dataset import EXPECTED_SHA256, read_raw
from telco_churn.evaluation import ClassifierEvaluator, EvaluationResult
from telco_churn.features import FeatureSchema
from telco_churn.pipeline import ChurnPipelineBuilder, EstimatorRegistry, PipelineInspector
from telco_churn.prediction import ChurnPredictor
from telco_churn.provenance import RunProvenance
from telco_churn.reporting import ReportWriter
from telco_churn.splitting import DatasetSplits, DatasetSplitter

METRIC_DEFINITIONS = {
    "positive_class": "Churn = Yes, encoded as 1",
    "average_precision": "sklearn average_precision_score on positive-class probabilities",
    "roc_auc": "sklearn roc_auc_score on positive-class probabilities",
    "accuracy": "share of correct class predictions at the threshold",
    "precision": "TP / (TP + FP) at the threshold (0 if no positive prediction)",
    "recall": "TP / (TP + FN) at the threshold",
    "f1": "harmonic mean of precision and recall at the threshold",
    "threshold_rule": "predict churn when positive-class probability >= threshold",
    "fit_seconds": "wall-clock time of one pipeline.fit on this machine; descriptive only",
}


@dataclass(frozen=True)
class TrainingData:
    """The verified raw snapshot used for a run.

    Attributes:
        records: Raw records exactly as read from the CSV.
        sha256: SHA-256 of the CSV bytes.
        source_path: Location of the CSV.
    """

    records: pd.DataFrame
    sha256: str
    source_path: Path

    @classmethod
    def load(cls, config: TrainingConfig) -> TrainingData:
        """Reads the raw CSV and refuses bytes other than the pinned snapshot.

        Args:
            config: Run configuration with the source path.

        Returns:
            The raw records and their hash.

        Raises:
            FileNotFoundError: If the CSV has not been downloaded.
            ValueError: If the CSV's SHA-256 differs from the recorded one.
        """
        records = read_raw(config.source_path)
        sha256 = RunProvenance.sha256_file(config.source_path)
        if sha256 != EXPECTED_SHA256:
            raise ValueError(
                f"{config.source_path} has SHA-256 {sha256}, expected {EXPECTED_SHA256}. "
                "Training only runs on the original snapshot."
            )
        return cls(records=records, sha256=sha256, source_path=config.source_path)


@dataclass
class PreparedSplits:
    """Parsed features and encoded labels of every partition.

    Attributes:
        raw: The raw partitions (including IDs and labels).
        features: Parsed input columns per partition name.
        labels: Encoded labels per partition name.
    """

    raw: DatasetSplits
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    labels: dict[str, pd.Series] = field(default_factory=dict)

    @classmethod
    def prepare(cls, splits: DatasetSplits, schema: FeatureSchema) -> PreparedSplits:
        """Applies the stateless parsing rules and target encoding per partition.

        Parsing learns nothing from data, so running it on each partition
        separately is leakage-free; every learned statistic is fitted later
        by the pipeline on the training partition only.

        Args:
            splits: Raw partitions.
            schema: Feature roles and parsing rules.

        Returns:
            Model-ready inputs and labels for train, validation and test.
        """
        prepared = cls(raw=splits)
        for name, frame in splits.items():
            prepared.features[name] = schema.parse_features(frame)
            prepared.labels[name] = schema.encode_target(frame[schema.target_column])
        return prepared


@dataclass(frozen=True)
class CandidateResult:
    """Outcome of fitting and validating one candidate.

    Attributes:
        config: Candidate configuration.
        pipeline: The fitted pipeline.
        fit_seconds: Wall-clock fit duration.
        validation: Validation metrics.
    """

    config: EstimatorConfig
    pipeline: Pipeline
    fit_seconds: float
    validation: EvaluationResult

    @property
    def is_baseline(self) -> bool:
        """Whether this candidate is the non-learning reference.

        Returns:
            ``True`` for the dummy majority-class model.
        """
        return self.config.estimator == "dummy"

    def to_row(self) -> dict[str, Any]:
        """Flattens the result into one comparison-table row.

        Returns:
            Name, configuration, fit time and validation metrics.
        """
        metrics = self.validation
        matrix = metrics.confusion_matrix
        classifier = self.pipeline.named_steps["classifier"]
        return {
            "model": self.config.name,
            "estimator": type(classifier).__name__,
            "is_baseline": self.is_baseline,
            "complexity_rank": self.config.complexity_rank,
            "params": ReportWriter.to_json_safe(classifier.get_params()),
            "fit_seconds": round(self.fit_seconds, 4),
            "average_precision": metrics.average_precision,
            "roc_auc": metrics.roc_auc,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "tn": matrix["tn"],
            "fp": matrix["fp"],
            "fn": matrix["fn"],
            "tp": matrix["tp"],
        }


@dataclass(frozen=True)
class SelectionDecision:
    """Result of applying the pre-registered selection rule.

    Attributes:
        selected: Name of the chosen candidate.
        best_by_metric: Candidate with the highest raw metric value.
        tied: Candidates within tolerance of the best, simplest first.
        rationale: Plain-language explanation of the outcome.
    """

    selected: str
    best_by_metric: str
    tied: list[str]
    rationale: str


class ModelSelector:
    """Applies the selection rule written in the configuration.

    Only learned candidates compete; the baseline gives context. The best
    validation score wins, except that any candidate within ``tolerance`` of
    it counts as tied, and ties go to the simplest model (lowest
    ``complexity_rank``). The test partition is never consulted.

    Attributes:
        config: Metric, tolerance and rule text.
    """

    def __init__(self, config: SelectionConfig) -> None:
        """Initialises the selector.

        Args:
            config: Metric, tolerance and rule text.
        """
        self.config = config

    def select(self, results: list[CandidateResult]) -> SelectionDecision:
        """Chooses one learned candidate from validation results.

        Args:
            results: Validation outcomes of all candidates.

        Returns:
            The decision and its rationale.

        Raises:
            ValueError: If there is no learned candidate.
        """
        learned = [result for result in results if not result.is_baseline]
        if not learned:
            raise ValueError("Selection needs at least one learned candidate.")
        metric = self.config.metric

        def score(result: CandidateResult) -> float:
            return getattr(result.validation, metric)

        best = max(learned, key=score)
        tied = sorted(
            (r for r in learned if score(best) - score(r) <= self.config.tolerance),
            key=lambda r: (r.config.complexity_rank, -score(r)),
        )
        chosen = tied[0]
        if chosen is best:
            rationale = (
                f"{best.config.name} has the highest validation {metric} "
                f"({score(best):.4f}) and no simpler candidate is within "
                f"{self.config.tolerance}."
            )
        else:
            rationale = (
                f"{best.config.name} has the highest validation {metric} ({score(best):.4f}), "
                f"but {chosen.config.name} ({score(chosen):.4f}) is within the "
                f"{self.config.tolerance} tolerance and simpler, so it is selected."
            )
        return SelectionDecision(
            selected=chosen.config.name,
            best_by_metric=best.config.name,
            tied=[r.config.name for r in tied],
            rationale=rationale,
        )


class FirstModelExperiment:
    """Milestone 1: two numeric inputs, one classifier and a majority baseline.

    Both models are fitted on the training partition only and evaluated on
    validation. This result is context, not part of the fair comparison: its
    feature set differs from the six-feature candidates.

    Attributes:
        config: Run configuration.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialises the experiment.

        Args:
            config: Run configuration.
        """
        self.config = config

    def run(self) -> dict[str, Any]:
        """Fits and evaluates the first model and the baseline, then saves the report.

        Returns:
            The report written to ``outputs.first_model_report``.
        """
        data = TrainingData.load(self.config)
        full_schema = FeatureSchema.from_config(self.config.features)
        schema = full_schema.subset(self.config.first_model_features)
        splits = DatasetSplitter(self.config.split, schema.id_column, schema.target_column).split(
            data.records
        )
        prepared = PreparedSplits.prepare(splits, schema)
        builder = ChurnPipelineBuilder(schema, self.config.preprocessing)
        registry = EstimatorRegistry(self.config.split.seed)
        evaluator = ClassifierEvaluator(threshold=self.config.selection.threshold)
        baseline = EstimatorConfig(
            name="majority_baseline", estimator="dummy", params={"strategy": "most_frequent"}
        )

        models = {}
        for estimator_config in (baseline, self.config.first_model):
            pipeline = builder.build(registry.create(estimator_config))
            pipeline.fit(prepared.features["train"], prepared.labels["train"])
            models[estimator_config.name] = evaluator.evaluate(
                pipeline, prepared.features["validation"], prepared.labels["validation"]
            ).to_dict()

        report = {
            "milestone": "TP02 section 1 - first working model",
            "run_utc": RunProvenance.timestamp(),
            "data_sha256": data.sha256,
            "split": {"seed": self.config.split.seed, "sizes": splits.sizes()},
            "features": list(schema.feature_columns),
            "fitted_on": "train",
            "evaluated_on": "validation",
            "metric_definitions": METRIC_DEFINITIONS,
            "models": models,
            "note": "Not comparable as an algorithm improvement with the six-feature "
            "candidates: the feature set differs.",
        }
        ReportWriter.write_json(self.config.outputs.first_model_report, report)
        return report


class TrainingWorkflow:
    """Trains, compares, selects and saves the churn pipeline (TP02 sections 2-4).

    Steps: load and verify the snapshot, split once with the configured seed,
    parse inputs, fit a fresh pipeline per candidate on the training
    partition, evaluate each on validation, apply the pre-registered
    selection rule, then save the selected fitted pipeline with its metadata.
    The test partition is allocated and recorded but never evaluated.

    Attributes:
        config: Run configuration.
        schema: Feature roles and parsing rules.
        builder: Creates fresh unfitted pipelines.
        registry: Creates fresh classifiers from configuration.
        evaluator: Computes the agreed metrics.
        selector: Applies the selection rule.
        provenance: Records code and environment identity.
    """

    REFERENCE_BATCH_SIZE = 10

    def __init__(self, config: TrainingConfig) -> None:
        """Initialises the workflow from a configuration.

        Args:
            config: Run configuration.
        """
        self.config = config
        self.schema = FeatureSchema.from_config(config.features)
        self.builder = ChurnPipelineBuilder(self.schema, config.preprocessing)
        self.registry = EstimatorRegistry(config.split.seed)
        self.evaluator = ClassifierEvaluator(threshold=config.selection.threshold)
        self.selector = ModelSelector(config.selection)
        self.provenance = RunProvenance()

    def run(self) -> dict[str, Any]:
        """Executes the full training workflow and writes all outputs.

        Returns:
            Summary with the comparison rows, the decision and output paths.
        """
        data = TrainingData.load(self.config)
        splits = DatasetSplitter(
            self.config.split, self.schema.id_column, self.schema.target_column
        ).split(data.records)
        prepared = PreparedSplits.prepare(splits, self.schema)

        results = [self.fit_candidate(candidate, prepared) for candidate in self.config.candidates]
        decision = self.selector.select(results)
        comparison = self._write_comparison(results, decision, splits)

        selected = next(r for r in results if r.config.name == decision.selected)
        metadata_path = self._save_artifact(selected, decision, data, splits, prepared)
        return {
            "comparison": comparison,
            "decision": decision,
            "metadata_path": metadata_path,
        }

    def fit_candidate(
        self, candidate: EstimatorConfig, prepared: PreparedSplits
    ) -> CandidateResult:
        """Fits one fresh pipeline on training data and scores it on validation.

        Args:
            candidate: Candidate configuration.
            prepared: Parsed partitions.

        Returns:
            The fitted pipeline, its fit time and validation metrics.
        """
        pipeline = self.builder.build(self.registry.create(candidate))
        start = time.perf_counter()
        pipeline.fit(prepared.features["train"], prepared.labels["train"])
        fit_seconds = time.perf_counter() - start
        validation = self.evaluator.evaluate(
            pipeline, prepared.features["validation"], prepared.labels["validation"]
        )
        return CandidateResult(candidate, pipeline, fit_seconds, validation)

    def _write_comparison(
        self, results: list[CandidateResult], decision: SelectionDecision, splits: DatasetSplits
    ) -> dict[str, Any]:
        """Saves the comparison table as JSON and CSV.

        Args:
            results: Validation outcomes of all candidates.
            decision: Applied selection rule.
            splits: Partitions, for sizes.

        Returns:
            The JSON payload that was written.
        """
        rows = [result.to_row() for result in results]
        payload = {
            "run_utc": RunProvenance.timestamp(),
            "evaluated_on": "validation",
            "test_set_used": False,
            "split": {"seed": self.config.split.seed, "sizes": splits.sizes()},
            "features": self.schema.feature_columns,
            "threshold": self.config.selection.threshold,
            "metric_definitions": METRIC_DEFINITIONS,
            "selection": {
                "metric": self.config.selection.metric,
                "tolerance": self.config.selection.tolerance,
                "rule": self.config.selection.rule,
                "decision": asdict(decision),
            },
            "rows": rows,
        }
        ReportWriter.write_json(self.config.outputs.comparison_json, payload)
        table = pd.DataFrame(rows).drop(columns=["params"])
        ReportWriter.write_csv(self.config.outputs.comparison_csv, table)
        return payload

    def _save_artifact(
        self,
        selected: CandidateResult,
        decision: SelectionDecision,
        data: TrainingData,
        splits: DatasetSplits,
        prepared: PreparedSplits,
    ) -> Any:
        """Persists the selected fitted pipeline with its reproduction record.

        Args:
            selected: The chosen candidate.
            decision: Applied selection rule.
            data: Verified snapshot.
            splits: Partitions, for the allocation table.
            prepared: Parsed partitions, for learned-statistic evidence.

        Returns:
            Path of the written metadata file.
        """
        reference_inputs = splits.validation.head(self.REFERENCE_BATCH_SIZE)[
            [self.schema.id_column, *self.schema.feature_columns]
        ]
        predictor = ChurnPredictor(selected.pipeline, self.schema, self.config.selection.threshold)
        reference_predictions = predictor.predict(reference_inputs)
        classifier = selected.pipeline.named_steps["classifier"]
        metadata = {
            "artifact": {
                "name": self.config.outputs.artifact_dir.name,
                "created_utc": RunProvenance.timestamp(),
                "format": "joblib (pickle); load only from a trusted source",
                "content": "fitted sklearn Pipeline: preprocessor + classifier",
                "train_command": "uv run python scripts/train.py",
            },
            "data": {
                "source_path": self.config.source_path,
                "sha256": data.sha256,
                "n_records": len(data.records),
            },
            "split": {
                "seed": self.config.split.seed,
                "fractions": asdict(self.config.split),
                "sizes": splits.sizes(),
                "stratified_on": self.schema.target_column,
                "allocation_file": ArtifactStore.ALLOCATION,
                "test_set_used_for_selection": False,
            },
            "features": self.schema.describe(),
            "preprocessing": {
                **self.builder.describe(),
                "learned_statistics": PipelineInspector.learned_statistics(selected.pipeline),
            },
            "model": {
                "name": selected.config.name,
                "estimator": type(classifier).__name__,
                "params": classifier.get_params(),
            },
            "selection": {
                "metric": self.config.selection.metric,
                "tolerance": self.config.selection.tolerance,
                "rule": self.config.selection.rule,
                "threshold": self.config.selection.threshold,
                "decision": asdict(decision),
                "comparison_file": self.config.outputs.comparison_json,
            },
            "metrics": {"validation": selected.validation.to_dict()},
            "config": {
                "path": self.config.path,
                "sha256": RunProvenance.sha256_file(self.config.path),
            },
            "code": self.provenance.code(),
            "environment": self.provenance.environment(),
        }
        store = ArtifactStore(self.config.outputs.artifact_dir)
        return store.save(
            selected.pipeline,
            ReportWriter.to_json_safe(metadata),
            splits.allocation(),
            reference_inputs,
            reference_predictions,
        )
