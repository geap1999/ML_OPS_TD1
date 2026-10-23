from __future__ import annotations

from typing import Any, ClassVar

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from telco_churn.config import EstimatorConfig, PreprocessingConfig
from telco_churn.features import FeatureSchema


class EstimatorRegistry:
    """Allow-list of classifier families that a configuration may request.

    Configurations name estimators by key instead of by import path, so a
    configuration file can never make the code import arbitrary objects.

    Attributes:
        ESTIMATORS: Mapping from configuration key to estimator class.
    """

    ESTIMATORS: ClassVar[dict[str, type[BaseEstimator]]] = {
        "dummy": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
    }

    def __init__(self, seed: int) -> None:
        """Initialises the registry.

        Args:
            seed: Random state given to every estimator that accepts one and
                whose configuration does not set it explicitly.
        """
        self.seed = seed

    def create(self, config: EstimatorConfig) -> ClassifierMixin:
        """Instantiates a new, unfitted classifier from its configuration.

        Args:
            config: Registry key and constructor parameters.

        Returns:
            A fresh classifier instance.

        Raises:
            KeyError: If the registry key is unknown.
        """
        if config.estimator not in self.ESTIMATORS:
            raise KeyError(
                f"Unknown estimator {config.estimator!r}; allowed: {sorted(self.ESTIMATORS)}."
            )
        estimator_class = self.ESTIMATORS[config.estimator]
        params: dict[str, Any] = dict(config.params)
        if "random_state" in estimator_class().get_params() and "random_state" not in params:
            params["random_state"] = self.seed
        return estimator_class(**params)


class ChurnPipelineBuilder:
    """Constructs *unfitted* preprocessing-plus-classifier pipelines.

    The builder never reads data and never trains: ``build`` only assembles
    estimators. Each call creates brand-new preprocessing objects and a clone
    of the classifier, so candidates never share fitted state.

    The resulting pipeline has two steps:

    * ``preprocessor``: a ``ColumnTransformer`` routing numeric inputs to the
      ``num`` branch (imputer then scaler) and categorical inputs to the
      ``cat`` branch (imputer then one-hot encoder). Columns not in the schema
      (for example ``customerID``) are dropped.
    * ``classifier``: the estimator being trained.

    Attributes:
        schema: Input columns and their roles.
        preprocessing: Imputation, scaling and encoding settings.
    """

    SCALERS: ClassVar[dict[str, type[BaseEstimator]]] = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
    }

    def __init__(self, schema: FeatureSchema, preprocessing: PreprocessingConfig) -> None:
        """Initialises the builder.

        Args:
            schema: Input columns and their roles.
            preprocessing: Imputation, scaling and encoding settings.

        Raises:
            KeyError: If the configured scaler is not supported.
        """
        if preprocessing.numeric_scaler not in self.SCALERS:
            raise KeyError(
                f"Unknown scaler {preprocessing.numeric_scaler!r}; "
                f"allowed: {sorted(self.SCALERS)}."
            )
        self.schema = schema
        self.preprocessing = preprocessing

    def build(self, classifier: ClassifierMixin) -> Pipeline:
        """Assembles a fresh, unfitted pipeline around a classifier.

        Args:
            classifier: Classifier to place after preprocessing. It is cloned,
                so the caller's instance is never fitted by the pipeline.

        Returns:
            An unfitted ``Pipeline`` with ``preprocessor`` and ``classifier``
            steps.
        """
        return Pipeline(
            [
                ("preprocessor", self.build_preprocessor()),
                ("classifier", clone(classifier)),
            ]
        )

    def build_preprocessor(self) -> ColumnTransformer:
        """Assembles a fresh, unfitted ``ColumnTransformer``.

        Returns:
            The preprocessing step, with a ``num`` branch and, when the schema
            has categorical inputs, a ``cat`` branch.
        """
        branches = [("num", self._numeric_branch(), list(self.schema.numeric))]
        if self.schema.categorical:
            branches.append(("cat", self._categorical_branch(), list(self.schema.categorical)))
        return ColumnTransformer(branches, remainder="drop")

    def describe(self) -> dict[str, Any]:
        """Serialisable summary of the preprocessing recipe for metadata.

        Returns:
            Branch composition and settings.
        """
        return {
            "num": {
                "columns": list(self.schema.numeric),
                "imputer": f"SimpleImputer(strategy={self.preprocessing.numeric_imputer!r})",
                "scaler": self.SCALERS[self.preprocessing.numeric_scaler].__name__,
            },
            "cat": {
                "columns": list(self.schema.categorical),
                "imputer": f"SimpleImputer(strategy={self.preprocessing.categorical_imputer!r})",
                "encoder": "OneHotEncoder("
                f"handle_unknown={self.preprocessing.unknown_categories!r}, sparse_output=False)",
            },
            "remainder": "drop",
            "fitted_on": "training split only",
        }

    def _numeric_branch(self) -> Pipeline:
        """Creates the numeric branch: imputation then scaling.

        Returns:
            An unfitted numeric preprocessing pipeline.
        """
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy=self.preprocessing.numeric_imputer)),
                ("scaler", self.SCALERS[self.preprocessing.numeric_scaler]()),
            ]
        )

    def _categorical_branch(self) -> Pipeline:
        """Creates the categorical branch: imputation then one-hot encoding.

        Returns:
            An unfitted categorical preprocessing pipeline.
        """
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy=self.preprocessing.categorical_imputer)),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown=self.preprocessing.unknown_categories,
                        sparse_output=False,
                    ),
                ),
            ]
        )


class PipelineInspector:
    """Reads the state that a fitted churn pipeline learned during ``fit``.

    Used to record learned statistics in metadata and to prove that
    prediction does not change them.
    """

    @staticmethod
    def learned_statistics(pipeline: Pipeline) -> dict[str, Any]:
        """Extracts the statistics the preprocessing learned during ``fit``.

        Args:
            pipeline: A fitted pipeline built by ``ChurnPipelineBuilder``.

        Returns:
            Numeric imputer fill values and scaler parameters, and the
            categories known to the encoder, keyed by column, as plain floats
            and strings.
        """
        transformers = pipeline.named_steps["preprocessor"].named_transformers_
        numeric = transformers["num"]
        columns = [str(c) for c in numeric.feature_names_in_]
        scaler = numeric.named_steps["scaler"]
        stats: dict[str, Any] = {
            "num_imputer_statistics": PipelineInspector._by_column(
                columns, numeric.named_steps["imputer"].statistics_
            ),
            "num_scaler": {
                name: PipelineInspector._by_column(columns, getattr(scaler, name))
                for name in ("mean_", "scale_", "data_min_", "data_max_")
                if hasattr(scaler, name)
            },
        }
        if "cat" in transformers:
            categorical = transformers["cat"]
            encoder = categorical.named_steps["encoder"]
            stats["cat_categories"] = {
                str(column): [str(value) for value in categories]
                for column, categories in zip(
                    categorical.feature_names_in_, encoder.categories_, strict=True
                )
            }
        return stats

    @staticmethod
    def _by_column(columns: list[str], values: Any) -> dict[str, float]:
        """Pairs per-column learned values with their column names.

        Args:
            columns: Column names in transformer order.
            values: One learned value per column.

        Returns:
            Mapping from column name to float value.
        """
        return {column: float(value) for column, value in zip(columns, values, strict=True)}
