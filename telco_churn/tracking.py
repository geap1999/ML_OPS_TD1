from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Any

from matplotlib.figure import Figure
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
import mlflow.sklearn
import pandas as pd

from telco_churn.artifacts import ArtifactStore
from telco_churn.config import EstimatorConfig, TrackingConfig, TrainingConfig
from telco_churn.evaluation import EvaluationResult
from telco_churn.prediction import ChurnPredictor
from telco_churn.provenance import RunProvenance
from telco_churn.registry import ModelRegistrar, RegisteredVersion
from telco_churn.reporting import ReportWriter
from telco_churn.training import (
    METRIC_DEFINITIONS,
    CandidateResult,
    PreparedSplits,
    SelectionDecision,
    TrainingData,
    TrainingWorkflow,
)


class ConfusionMatrixFigure:
    """Draws a validation confusion matrix with its class order written out.

    Rows are true labels and columns are predicted labels, both in the order
    ``[0, 1]`` where ``0`` is "No churn" and ``1`` is "Churn" (the positive
    class). The figure is built with Matplotlib's object API, so no display
    or global pyplot state is needed.
    """

    CLASS_NAMES = ("0 = No churn", "1 = Churn")

    @classmethod
    def save(cls, result: EvaluationResult, title: str, path: Path) -> Path:
        """Writes the confusion matrix of one evaluation as a PNG.

        Args:
            result: Validation metrics holding the matrix counts.
            title: Figure title, usually the run name.
            path: Destination PNG file.

        Returns:
            The written path.
        """
        matrix = result.confusion_matrix["matrix"]
        figure = Figure(figsize=(4.6, 4.2), dpi=120)
        axes = figure.subplots()
        axes.imshow(matrix, cmap="Blues")
        axes.set_xticks([0, 1], labels=cls.CLASS_NAMES)
        axes.set_yticks([0, 1], labels=cls.CLASS_NAMES, rotation=90, va="center")
        axes.set_xlabel(f"Predicted label (threshold {result.threshold:g})")
        axes.set_ylabel("True label")
        axes.set_title(f"{title}\nvalidation, n = {result.n_samples}", fontsize=10)
        limit = max(max(row) for row in matrix) / 2
        for row in range(2):
            for column in range(2):
                count = matrix[row][column]
                axes.text(
                    column,
                    row,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if count > limit else "black",
                )
        figure.tight_layout()
        figure.savefig(path)
        return path


@dataclass(frozen=True)
class ExperimentContext:
    """Facts shared by every candidate run of one tracked comparison.

    Attributes:
        data: The verified raw snapshot.
        prepared: Parsed train, validation and test partitions.
        allocation: Customer-to-partition table.
        allocation_sha256: SHA-256 of the allocation table as CSV bytes.
        code: Git commit, dirty flag and package source hashes.
        environment: Interpreter and library versions.
        config_sha256: SHA-256 of the configuration file.
        train_dataset: MLflow record describing the training partition.
        validation_dataset: MLflow record describing the validation partition.
        reference_inputs: Small raw validation batch, without labels.
    """

    data: TrainingData
    prepared: PreparedSplits
    allocation: pd.DataFrame
    allocation_sha256: str
    code: dict[str, Any]
    environment: dict[str, Any]
    config_sha256: str
    train_dataset: PandasDataset
    validation_dataset: PandasDataset
    reference_inputs: pd.DataFrame


@dataclass(frozen=True)
class TrackedRun:
    """One finished candidate run and the model logged inside it.

    Attributes:
        result: Fitted pipeline, fit time and validation metrics.
        run_id: MLflow run that produced the result.
        run_name: Descriptive name of that run.
        model_uri: URI of the logged model (preprocessing plus classifier).
        model_id: Identifier of the logged model.
    """

    result: CandidateResult
    run_id: str
    run_name: str
    model_uri: str
    model_id: str

    def to_row(self) -> dict[str, Any]:
        """Flattens the run into one row of the selection table.

        Returns:
            Run identity, changed setting and validation metrics.
        """
        metrics = self.result.validation
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "candidate": self.result.config.name,
            "is_baseline": self.result.is_baseline,
            "model_uri": self.model_uri,
            "val_average_precision": metrics.average_precision,
            "val_precision": metrics.precision,
            "val_recall": metrics.recall,
            "val_f1": metrics.f1,
            "fit_seconds": round(self.result.fit_seconds, 4),
        }


class TrackedTrainingWorkflow:
    """Runs the TP02 comparison with one MLflow run per candidate (TP03).

    Nothing about the modelling changes: the same configuration, split,
    pipeline builder, evaluator and selection rule as ``TrainingWorkflow``
    are reused. This class only adds explicit tracking around them:

    * parameters: estimator type, the classifier's actual hyperparameters,
      split seed, threshold and preprocessing settings;
    * metrics: validation average precision, precision, recall, F1 (plus
      accuracy, ROC AUC, confusion counts and fit time);
    * tags: Git commit, dirty flag, SHA-256 of the original CSV, of the
      configuration and of the customer allocation;
    * artifacts: the validation confusion matrix (PNG and JSON), a
      provenance JSON, the customer allocation and a small reference batch
      with the predictions made before logging;
    * dataset inputs: training and validation records with distinct contexts;
    * the complete fitted pipeline, logged with ``mlflow.sklearn.log_model``.

    After the runs, the pre-registered rule selects one candidate on
    validation, and its logged model is registered and given the alias. The
    test partition is never evaluated.

    Attributes:
        config: Run configuration.
        tracking: Tracking server, experiment and registry names.
        workflow: The untracked TP02 workflow whose steps are reused.
        client: Client of the tracking server.
    """

    PROVENANCE_FILE = "provenance.json"
    CONFUSION_DIR = "evaluation"
    SPLIT_DIR = "split"
    REFERENCE_DIR = "reference"

    def __init__(self, config: TrainingConfig) -> None:
        """Initialises the workflow and points MLflow at the tracking server.

        Args:
            config: Run configuration with a ``tracking`` section.

        Raises:
            ValueError: If the configuration has no ``tracking`` section.
        """
        if config.tracking is None:
            raise ValueError(f"{config.path} has no 'tracking' section.")
        self.config = config
        self.tracking: TrackingConfig = config.tracking
        self.workflow = TrainingWorkflow(config)
        mlflow.set_tracking_uri(self.tracking.tracking_uri)
        self.client = mlflow.MlflowClient()

    def run(self) -> dict[str, Any]:
        """Tracks every candidate, selects one and registers its model.

        Returns:
            The record written to ``tracking.run_report``.
        """
        experiment = mlflow.set_experiment(self.tracking.experiment)
        context = self._build_context()
        runs = [self.track_candidate(candidate, context) for candidate in self.config.candidates]

        decision = self.workflow.selector.select([run.result for run in runs])
        selected = next(run for run in runs if run.result.config.name == decision.selected)
        self._tag_decision(runs, selected, decision)
        version = ModelRegistrar(self.tracking, self.client).register(
            selected.model_uri,
            tags={
                "source_run_id": selected.run_id,
                "git_commit": str(context.code["git_commit"]),
                "data_sha256": context.data.sha256,
                "selection_metric": self.config.selection.metric,
            },
        )
        return self._write_report(context, experiment.experiment_id, runs, selected, version)

    def track_candidate(
        self, candidate: EstimatorConfig, context: ExperimentContext
    ) -> TrackedRun:
        """Fits and validates one candidate inside its own MLflow run.

        Args:
            candidate: Candidate configuration.
            context: Facts shared by all runs of this comparison.

        Returns:
            The finished run and its logged model.
        """
        prepared = context.prepared
        with mlflow.start_run(run_name=candidate.tracked_name) as run:
            mlflow.set_tags(self._tags(candidate, context))
            mlflow.log_input(context.train_dataset, context="training")
            mlflow.log_input(context.validation_dataset, context="validation")

            result = self.workflow.fit_candidate(candidate, prepared)
            classifier = result.pipeline.named_steps["classifier"]
            mlflow.log_params(self._params(candidate, classifier))

            model_info = mlflow.sklearn.log_model(
                sk_model=result.pipeline,
                name=self.tracking.model_name,
                input_example=prepared.features["train"].head(3),
                serialization_format="cloudpickle",
            )
            mlflow.log_metrics(
                self._metrics(result),
                model_id=model_info.model_id,
                dataset=context.validation_dataset,
            )
            self._log_files(result, candidate, context)
            return TrackedRun(
                result=result,
                run_id=run.info.run_id,
                run_name=candidate.tracked_name,
                model_uri=model_info.model_uri,
                model_id=model_info.model_id,
            )

    def export_comparison(self, experiment_id: str) -> pd.DataFrame:
        """Exports the finished runs of the experiment from the tracking server.

        The table is read back with ``mlflow.search_runs`` instead of being
        copied from memory, so it shows what the server actually stored.

        Args:
            experiment_id: Experiment holding the candidate runs.

        Returns:
            One row per run, best validation average precision first.
        """
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.val_average_precision DESC"],
        )
        columns = {
            "run_id": "run_id",
            "tags.mlflow.runName": "run_name",
            "params.model_type": "model_type",
            "tags.role": "role",
            "tags.selected": "selected",
            "metrics.val_average_precision": "val_average_precision",
            "metrics.val_precision": "val_precision",
            "metrics.val_recall": "val_recall",
            "metrics.val_f1": "val_f1",
            "metrics.fit_seconds": "fit_seconds",
            "params.threshold": "threshold",
            "params.split_seed": "split_seed",
            "tags.git_commit": "git_commit",
            "tags.git_dirty": "git_dirty",
            "tags.data_sha256": "data_sha256",
            "start_time": "start_time",
        }
        table = runs[[column for column in columns if column in runs.columns]]
        return table.rename(columns=columns)

    def _build_context(self) -> ExperimentContext:
        """Loads the data once and prepares what every run records.

        Returns:
            The shared context of this comparison.
        """
        data, prepared = self.workflow.load_partitions()
        schema = self.workflow.schema
        allocation = prepared.raw.allocation()
        with tempfile.TemporaryDirectory() as folder:
            allocation_file = ReportWriter.write_csv(
                Path(folder) / ArtifactStore.ALLOCATION, allocation
            )
            allocation_sha256 = RunProvenance.sha256_file(allocation_file)
        return ExperimentContext(
            data=data,
            prepared=prepared,
            allocation=allocation,
            allocation_sha256=allocation_sha256,
            code=self.workflow.provenance.code(),
            environment=self.workflow.provenance.environment(),
            config_sha256=RunProvenance.sha256_file(self.config.path),
            train_dataset=self._dataset(prepared, "train", "telco_train"),
            validation_dataset=self._dataset(prepared, "validation", "telco_validation"),
            reference_inputs=prepared.raw.validation.head(TrainingWorkflow.REFERENCE_BATCH_SIZE)[
                [schema.id_column, *schema.feature_columns]
            ],
        )

    def _dataset(self, prepared: PreparedSplits, partition: str, name: str) -> PandasDataset:
        """Describes one partition as an MLflow dataset record.

        The record holds the six parsed inputs and the raw target, so its
        schema and digest describe what was used. It is metadata only: MLflow
        does not store the rows, and the digest is not the CSV's SHA-256.

        Args:
            prepared: Parsed partitions.
            partition: ``"train"`` or ``"validation"``.
            name: Dataset name shown in the UI.

        Returns:
            The dataset record.
        """
        target = self.workflow.schema.target_column
        raw = dict(prepared.raw.items())[partition]
        frame = prepared.features[partition].assign(**{target: raw[target]})
        return mlflow.data.from_pandas(
            frame, source=ReportWriter.relative(self.config.source_path), name=name, targets=target
        )

    def _params(self, candidate: EstimatorConfig, classifier: Any) -> dict[str, Any]:
        """Collects the settings that were actually used for one candidate.

        Hyperparameters are read from the fitted classifier, not from the
        configuration, so defaults that were never written down are recorded
        too.

        Args:
            candidate: Candidate configuration.
            classifier: The fitted classifier step.

        Returns:
            Parameters to log on the run.
        """
        preprocessing = asdict(self.config.preprocessing)
        hyperparameters = ReportWriter.to_json_safe(classifier.get_params())
        return {
            "model_type": type(classifier).__name__,
            "candidate": candidate.name,
            "split_seed": self.config.split.seed,
            "threshold": self.config.selection.threshold,
            "primary_metric": self.config.selection.metric,
            "n_features": len(self.workflow.schema.feature_columns),
            **{f"prep__{key}": value for key, value in preprocessing.items()},
            **{f"clf__{key}": value for key, value in hyperparameters.items()},
        }

    def _tags(self, candidate: EstimatorConfig, context: ExperimentContext) -> dict[str, str]:
        """Builds the identity tags of one run.

        Args:
            candidate: Candidate configuration.
            context: Facts shared by all runs of this comparison.

        Returns:
            Tags linking the run to its code, data and customer allocation.
        """
        return {
            "git_commit": str(context.code["git_commit"]),
            "git_dirty": str(context.code["git_dirty"]),
            "data_sha256": context.data.sha256,
            "config_sha256": context.config_sha256,
            "split_allocation_sha256": context.allocation_sha256,
            "role": "baseline" if candidate.estimator == "dummy" else "candidate",
            "evaluated_on": "validation",
            "positive_class": "Churn = Yes, encoded as 1",
        }

    @staticmethod
    def _metrics(result: CandidateResult) -> dict[str, float]:
        """Names the validation metrics of one candidate for logging.

        Args:
            result: Fitted candidate with its validation metrics.

        Returns:
            Numeric metrics with a ``val_`` prefix, plus the fit duration.
        """
        validation = result.validation
        matrix = validation.confusion_matrix
        return {
            "val_average_precision": validation.average_precision,
            "val_precision": validation.precision,
            "val_recall": validation.recall,
            "val_f1": validation.f1,
            "val_accuracy": validation.accuracy,
            "val_roc_auc": validation.roc_auc,
            "val_true_negatives": float(matrix["tn"]),
            "val_false_positives": float(matrix["fp"]),
            "val_false_negatives": float(matrix["fn"]),
            "val_true_positives": float(matrix["tp"]),
            "fit_seconds": result.fit_seconds,
        }

    def _log_files(
        self, result: CandidateResult, candidate: EstimatorConfig, context: ExperimentContext
    ) -> None:
        """Logs the artifact files of the active run.

        Args:
            result: Fitted candidate with its validation metrics.
            candidate: Candidate configuration.
            context: Facts shared by all runs of this comparison.
        """
        predictor = ChurnPredictor(
            result.pipeline, self.workflow.schema, self.config.selection.threshold
        )
        reference_predictions = predictor.predict(context.reference_inputs)
        with tempfile.TemporaryDirectory() as folder:
            figure = ConfusionMatrixFigure.save(
                result.validation,
                candidate.tracked_name,
                Path(folder) / "val_confusion_matrix.png",
            )
            mlflow.log_artifact(str(figure), artifact_path=self.CONFUSION_DIR)
            allocation = ReportWriter.write_csv(
                Path(folder) / ArtifactStore.ALLOCATION, context.allocation
            )
            mlflow.log_artifact(str(allocation), artifact_path=self.SPLIT_DIR)
            for name, frame in (
                (ArtifactStore.REFERENCE_INPUTS, context.reference_inputs),
                (ArtifactStore.REFERENCE_PREDICTIONS, reference_predictions),
            ):
                written = ReportWriter.write_csv(Path(folder) / name, frame)
                mlflow.log_artifact(str(written), artifact_path=self.REFERENCE_DIR)
        mlflow.log_dict(
            result.validation.confusion_matrix,
            f"{self.CONFUSION_DIR}/val_confusion_matrix.json",
        )
        mlflow.log_dict(self._provenance(result, candidate, context), self.PROVENANCE_FILE)

    def _provenance(
        self, result: CandidateResult, candidate: EstimatorConfig, context: ExperimentContext
    ) -> dict[str, Any]:
        """Builds the compact configuration and provenance record of one run.

        Args:
            result: Fitted candidate with its validation metrics.
            candidate: Candidate configuration.
            context: Facts shared by all runs of this comparison.

        Returns:
            A JSON-serialisable record of what produced the result.
        """
        schema = self.workflow.schema
        splits = context.prepared.raw
        classifier = result.pipeline.named_steps["classifier"]
        return ReportWriter.to_json_safe(
            {
                "candidate": {
                    "name": candidate.name,
                    "run_name": candidate.tracked_name,
                    "estimator": type(classifier).__name__,
                    "params": classifier.get_params(),
                    "is_baseline": result.is_baseline,
                },
                "data": {
                    "source_path": self.config.source_path,
                    "sha256": context.data.sha256,
                    "n_records": len(context.data.records),
                    "note": "The hash identifies the CSV bytes; it does not store them.",
                },
                "split": {
                    "seed": self.config.split.seed,
                    "fractions": asdict(self.config.split),
                    "sizes": splits.sizes(),
                    "stratified_on": schema.target_column,
                    "allocation_artifact": f"{self.SPLIT_DIR}/{ArtifactStore.ALLOCATION}",
                    "allocation_sha256": context.allocation_sha256,
                    "test_set_used_for_selection": False,
                },
                "features": schema.describe(),
                "preprocessing": self.workflow.builder.describe(),
                "evaluation": {
                    "evaluated_on": "validation",
                    "primary_metric": self.config.selection.metric,
                    "threshold": self.config.selection.threshold,
                    "selection_rule": self.config.selection.rule,
                    "metric_definitions": METRIC_DEFINITIONS,
                    "confusion_matrix": {
                        "labels": result.validation.confusion_matrix["labels"],
                        "orientation": result.validation.confusion_matrix["orientation"],
                    },
                },
                "config": {"path": self.config.path, "sha256": context.config_sha256},
                "code": context.code,
                "environment": context.environment,
            }
        )

    def _tag_decision(
        self, runs: list[TrackedRun], selected: TrackedRun, decision: SelectionDecision
    ) -> None:
        """Marks every finished run with the outcome of the selection rule.

        Args:
            runs: All candidate runs.
            selected: The run whose model was chosen.
            decision: Applied selection rule.
        """
        for run in runs:
            self.client.set_tag(run.run_id, "selected", str(run is selected).lower())
        self.client.set_tag(selected.run_id, "selection_rationale", decision.rationale)

    def _write_report(
        self,
        context: ExperimentContext,
        experiment_id: str,
        runs: list[TrackedRun],
        selected: TrackedRun,
        version: RegisteredVersion,
    ) -> dict[str, Any]:
        """Saves the handover record and the comparison exported from the server.

        The code identity is the one captured before the first run started,
        which is also what the runs were tagged with; the report files written
        here would otherwise make the tree look modified.

        Args:
            context: Facts shared by all runs of this comparison.
            experiment_id: Experiment holding the runs.
            runs: All candidate runs.
            selected: The run whose model was chosen.
            version: Registered version carrying the alias.

        Returns:
            The JSON record that was written.
        """
        ReportWriter.write_csv(self.tracking.comparison_csv, self.export_comparison(experiment_id))
        decision = self.workflow.selector.select([run.result for run in runs])
        record = {
            "recorded_utc": RunProvenance.timestamp(),
            "tracking_uri": self.tracking.tracking_uri,
            "experiment": {"name": self.tracking.experiment, "id": experiment_id},
            "git_commit": context.code["git_commit"],
            "git_dirty": context.code["git_dirty"],
            "data_sha256": context.data.sha256,
            "split_allocation_sha256": context.allocation_sha256,
            "evaluated_on": "validation",
            "test_set_used": False,
            "threshold": self.config.selection.threshold,
            "selection": {
                "metric": self.config.selection.metric,
                "tolerance": self.config.selection.tolerance,
                "rule": self.config.selection.rule,
                "decision": asdict(decision),
            },
            "runs": [run.to_row() for run in runs],
            "selected": {
                "run_id": selected.run_id,
                "run_name": selected.run_name,
                "model_uri": selected.model_uri,
                "registered_model": version.name,
                "version": version.version,
                "alias": version.alias,
                "alias_uri": self.tracking.alias_uri,
            },
        }
        ReportWriter.write_json(self.tracking.run_report, record)
        return record
