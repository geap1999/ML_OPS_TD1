from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from telco_churn.config import TrackingConfig
from telco_churn.features import FeatureSchema
from telco_churn.pipeline import PipelineInspector
from telco_churn.prediction import ChurnPredictor
from telco_churn.provenance import RunProvenance
from telco_churn.verification import ArtifactVerifier

PROVENANCE_FILE = "provenance.json"
REFERENCE_INPUTS = "reference/reference_inputs.csv"
REFERENCE_PREDICTIONS = "reference/reference_predictions.csv"


@dataclass(frozen=True)
class RegisteredVersion:
    """Identity of one version of a registered model.

    Four identifiers answer four different questions: the run ID says which
    execution produced the result, the logged model URI says which recorded
    model to load, name plus version says which version of the managed model
    it is, and the alias says which version currently holds a named role.

    Attributes:
        name: Registered model name.
        version: Version number assigned by the registry.
        alias: Alias through which the version was requested or assigned, or
            ``None`` when it was addressed by number.
        run_id: Run that logged the model.
        source: Logged model URI the version was registered from.
    """

    name: str
    version: str
    alias: str | None
    run_id: str
    source: str


class ModelRegistrar:
    """Registers a logged model and points the handover alias at it.

    The alias is a mutable pointer, not a copy of the model and not a
    deployment: assigning it to a later version moves it. The version number
    returned here is the reproducible reference.

    Attributes:
        tracking: Registry name and alias.
        client: Client of the tracking server.
    """

    def __init__(self, tracking: TrackingConfig, client: mlflow.MlflowClient) -> None:
        """Initialises the registrar.

        Args:
            tracking: Registry name and alias.
            client: Client of the tracking server.
        """
        self.tracking = tracking
        self.client = client

    def register(self, model_uri: str, tags: dict[str, str]) -> RegisteredVersion:
        """Creates a new registered version and moves the alias to it.

        Args:
            model_uri: URI of the logged model returned by ``log_model``.
            tags: Identity tags stored on the registered version.

        Returns:
            The version that now carries the alias.
        """
        name = self.tracking.registered_model
        version = mlflow.register_model(model_uri, name, tags=tags)
        self.client.set_registered_model_alias(name, self.tracking.alias, version.version)
        return RegisteredVersion(
            name=name,
            version=str(version.version),
            alias=self.tracking.alias,
            run_id=str(version.run_id),
            source=model_uri,
        )


@dataclass(frozen=True)
class RegisteredModel:
    """A pipeline recovered from the registry with the record of its run.

    Attributes:
        pipeline: Fitted preprocessing plus classifier, as logged.
        version: Registry identity the URI resolved to at load time.
        provenance: The ``provenance.json`` logged by the source run.
        requested_uri: URI that was asked for, for example the alias URI.
    """

    pipeline: Pipeline
    version: RegisteredVersion
    provenance: dict[str, Any]
    requested_uri: str

    def predictor(self) -> ChurnPredictor:
        """Builds the inference wrapper from the run's own record.

        The feature schema and the threshold come from the source run's
        provenance file, not from the local configuration, so inference uses
        the input contract the model was trained with.

        Returns:
            A predictor applying the documented stateless parsing rules.
        """
        features = self.provenance["features"]
        schema = FeatureSchema(
            id_column=features["id_column"],
            target_column=features["target_column"],
            positive_label=features["positive_class"]["label"],
            negative_label=features["negative_class"]["label"],
            numeric=tuple(features["numeric"]),
            categorical=tuple(features["categorical"]),
        )
        return ChurnPredictor(
            self.pipeline, schema, float(self.provenance["evaluation"]["threshold"])
        )


class RegisteredModelLoader:
    """Recovers a registered pipeline from the tracking server, without training.

    Loading needs the tracking server (for registry and run metadata), the
    artifact store (for the model files) and an environment that can import
    scikit-learn and this package. It never calls ``fit``. MLflow models are
    pickles: load only models you logged yourself or trust.

    Attributes:
        tracking_uri: Address of the tracking server.
        client: Client of the tracking server.
    """

    def __init__(self, tracking_uri: str) -> None:
        """Initialises the loader and points MLflow at the tracking server.

        Args:
            tracking_uri: Address of the tracking server.
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.MlflowClient()

    def resolve(self, model_uri: str) -> RegisteredVersion:
        """Finds which registered version a registry URI currently means.

        Args:
            model_uri: ``models:/<name>@<alias>`` or ``models:/<name>/<version>``.

        Returns:
            The registry identity behind the URI right now.

        Raises:
            ValueError: If the URI is not a registry URI of one of those forms.
        """
        prefix = "models:/"
        reference = model_uri.removeprefix(prefix)
        if not model_uri.startswith(prefix) or not reference:
            raise ValueError(f"Not a registry model URI: {model_uri!r}.")
        if "@" in reference:
            name, alias = reference.split("@", 1)
            found = self.client.get_model_version_by_alias(name, alias)
        elif "/" in reference:
            name, number = reference.split("/", 1)
            alias = None
            found = self.client.get_model_version(name, number)
        else:
            raise ValueError(f"{model_uri!r} names neither an alias nor a version.")
        return RegisteredVersion(
            name=name,
            version=str(found.version),
            alias=alias,
            run_id=str(found.run_id),
            source=str(found.source),
        )

    def load(self, model_uri: str) -> RegisteredModel:
        """Loads the fitted pipeline behind a registry URI.

        Args:
            model_uri: Registry URI, usually the alias URI.

        Returns:
            The pipeline, the version it resolved to and its run's record.
        """
        version = self.resolve(model_uri)
        pipeline = mlflow.sklearn.load_model(model_uri)
        provenance = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{PROVENANCE_FILE}")
        return RegisteredModel(
            pipeline=pipeline, version=version, provenance=provenance, requested_uri=model_uri
        )

    def reference_batch(self, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Reads the reference batch logged by the source run.

        Args:
            run_id: Run that logged the model.

        Returns:
            The raw inputs (as text, without labels) and the predictions the
            in-memory pipeline made on them before it was logged.
        """
        inputs = mlflow.artifacts.load_text(f"runs:/{run_id}/{REFERENCE_INPUTS}")
        predictions = mlflow.artifacts.load_text(f"runs:/{run_id}/{REFERENCE_PREDICTIONS}")
        return (
            pd.read_csv(io.StringIO(inputs), dtype=str, keep_default_na=False),
            pd.read_csv(io.StringIO(predictions)),
        )


class RegistryVerifier:
    """Proves that the registered pipeline predicts like the original.

    Meant for a fresh Python process. It loads the model through the registry
    URI, disables ``fit`` on the pipeline and its steps, predicts the
    reference batch logged by the source run and compares the result with
    the predictions recorded before logging.

    Attributes:
        loader: Recovers models from the tracking server.
        tolerance: Maximum absolute probability difference accepted.
    """

    def __init__(self, loader: RegisteredModelLoader, tolerance: float = 1e-12) -> None:
        """Initialises the verifier.

        Args:
            loader: Recovers models from the tracking server.
            tolerance: Maximum absolute probability difference accepted.
        """
        self.loader = loader
        self.tolerance = tolerance

    def verify(self, model_uri: str) -> dict[str, Any]:
        """Runs the reload check.

        Args:
            model_uri: Registry URI, usually the alias URI.

        Returns:
            Evidence record; ``passed`` is ``True`` only if every check holds.
        """
        model = self.loader.load(model_uri)
        ArtifactVerifier.forbid_fit(model.pipeline)
        inputs, expected = self.loader.reference_batch(model.version.run_id)
        target = model.provenance["features"]["target_column"]

        before = PipelineInspector.learned_statistics(model.pipeline)
        actual = model.predictor().predict(inputs)
        after = PipelineInspector.learned_statistics(model.pipeline)

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
            "tracking_uri": self.loader.tracking_uri,
            "requested_uri": model_uri,
            "registered_model": model.version.name,
            "version": model.version.version,
            "alias": model.version.alias,
            "source_run_id": model.version.run_id,
            "logged_model_uri": model.version.source,
            "git_commit": model.provenance["code"]["git_commit"],
            "data_sha256": model.provenance["data"]["sha256"],
            "n_records": len(inputs),
            "tolerance_abs_probability": self.tolerance,
            "max_abs_probability_difference": float(difference.max()),
            "checks": checks,
            "passed": all(checks.values()),
        }
