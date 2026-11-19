from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from telco_churn.config import DEFAULT_CONFIG_PATH, EstimatorConfig, OutputConfig, TrainingConfig
from telco_churn.features import FeatureSchema
from telco_churn.pipeline import ChurnPipelineBuilder, EstimatorRegistry
from telco_churn.prediction import ChurnPredictor
from telco_churn.splitting import DatasetSplits, DatasetSplitter
from telco_churn.training import PreparedSplits, TrainingData


@pytest.fixture(scope="session")
def config() -> TrainingConfig:
    """Loads the committed training configuration.

    Returns:
        The project configuration.
    """
    return TrainingConfig.from_json(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def schema(config: TrainingConfig) -> FeatureSchema:
    """Builds the six-feature schema from the configuration.

    Args:
        config: Project configuration.

    Returns:
        The feature schema.
    """
    return FeatureSchema.from_config(config.features)


@pytest.fixture(scope="session")
def data(config: TrainingConfig) -> TrainingData:
    """Loads the verified raw snapshot, skipping when it is not downloaded.

    Args:
        config: Project configuration.

    Returns:
        The raw records and their hash.
    """
    if not config.source_path.exists():
        pytest.skip("Raw CSV missing: run `uv run python scripts/download_data.py` first.")
    return TrainingData.load(config)


@pytest.fixture(scope="session")
def splits(config: TrainingConfig, schema: FeatureSchema, data: TrainingData) -> DatasetSplits:
    """Splits the snapshot with the configured seed.

    Args:
        config: Project configuration.
        schema: Feature schema.
        data: Raw snapshot.

    Returns:
        Train, validation and test partitions.
    """
    splitter = DatasetSplitter(config.split, schema.id_column, schema.target_column)
    return splitter.split(data.records)


@pytest.fixture(scope="session")
def prepared(splits: DatasetSplits, schema: FeatureSchema) -> PreparedSplits:
    """Parses every partition.

    Args:
        splits: Raw partitions.
        schema: Feature schema.

    Returns:
        Parsed features and encoded labels.
    """
    return PreparedSplits.prepare(splits, schema)


@pytest.fixture()
def fitted_pipeline(config: TrainingConfig, schema: FeatureSchema, prepared: PreparedSplits):
    """Fits a fresh logistic-regression pipeline on the training partition.

    Function-scoped so a test that mutates or inspects state gets its own.

    Args:
        config: Project configuration.
        schema: Feature schema.
        prepared: Parsed partitions.

    Returns:
        The fitted pipeline.
    """
    registry = EstimatorRegistry(config.split.seed)
    builder = ChurnPipelineBuilder(schema, config.preprocessing)
    classifier = registry.create(
        EstimatorConfig(name="lr", estimator="logistic_regression", params={"max_iter": 1000})
    )
    pipeline = builder.build(classifier)
    pipeline.fit(prepared.features["train"], prepared.labels["train"])
    return pipeline


@pytest.fixture()
def isolated_config(config: TrainingConfig, tmp_path: Path) -> TrainingConfig:
    """Copies the configuration with every output redirected to a temp folder.

    Args:
        config: Project configuration.
        tmp_path: Per-test temporary directory.

    Returns:
        A configuration that cannot overwrite committed reports or models.
    """
    outputs = OutputConfig(
        first_model_report=tmp_path / "first_model.json",
        comparison_json=tmp_path / "model_comparison.json",
        comparison_csv=tmp_path / "model_comparison.csv",
        artifact_dir=tmp_path / "churn_pipeline",
    )
    return replace(config, outputs=outputs)


@pytest.fixture()
def customer(splits: DatasetSplits, schema: FeatureSchema) -> pd.DataFrame:
    """Returns one raw validation customer without its label.

    Args:
        splits: Raw partitions.
        schema: Feature schema.

    Returns:
        A one-row raw frame with the ID and the six inputs.
    """
    return splits.validation.head(1)[[schema.id_column, *schema.feature_columns]].copy()


SAMPLE_THRESHOLD = 0.5


@pytest.fixture()
def sample_customers() -> pd.DataFrame:
    """Builds sixteen invented customers for fast tests.

    Nothing here is read from the Telco CSV. Both classes are present
    (seven churners), every categorical input takes several values, and
    ``Contract`` never takes the value ``"Three year"``, which the tests use
    as a category absent from training.

    Returns:
        A new frame per test with the ID, the six raw inputs and the raw
        target, so a test can modify its copy freely.
    """
    month, one_year, two_year = "Month-to-month", "One year", "Two year"
    check, mailed = "Electronic check", "Mailed check"
    bank, card = "Bank transfer (automatic)", "Credit card (automatic)"
    rows = [
        ("S-01", 1, 70.35, 70.35, month, "Fiber optic", check, "Yes"),
        ("S-02", 2, 89.10, 178.20, month, "Fiber optic", check, "Yes"),
        ("S-03", 5, 99.65, 498.25, month, "Fiber optic", mailed, "Yes"),
        ("S-04", 8, 75.00, 600.00, month, "DSL", check, "Yes"),
        ("S-05", 3, 55.20, 165.60, month, "DSL", mailed, "Yes"),
        ("S-06", 12, 95.50, 1146.00, one_year, "Fiber optic", check, "Yes"),
        ("S-07", 4, 80.85, 323.40, month, "Fiber optic", card, "Yes"),
        ("S-08", 34, 56.95, 1936.30, one_year, "DSL", mailed, "No"),
        ("S-09", 45, 42.30, 1903.50, one_year, "DSL", bank, "No"),
        ("S-10", 60, 20.05, 1203.00, two_year, "No", mailed, "No"),
        ("S-11", 72, 110.00, 7920.00, two_year, "Fiber optic", card, "No"),
        ("S-12", 24, 19.80, 475.20, two_year, "No", bank, "No"),
        ("S-13", 50, 65.25, 3262.50, two_year, "DSL", card, "No"),
        ("S-14", 15, 25.10, 376.50, month, "No", mailed, "No"),
        ("S-15", 66, 85.40, 5636.40, one_year, "Fiber optic", bank, "No"),
        ("S-16", 30, 49.95, 1498.50, month, "DSL", check, "No"),
    ]
    columns = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "Churn",
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture()
def sample_features(sample_customers: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Keeps only the six agreed input columns of the invented customers.

    Args:
        sample_customers: Invented customers.
        schema: Feature schema.

    Returns:
        A valid inference batch without ``customerID`` and without ``Churn``.
    """
    return sample_customers[schema.feature_columns].copy()


@pytest.fixture()
def sample_labels(sample_customers: pd.DataFrame, schema: FeatureSchema) -> pd.Series:
    """Encodes the invented target with the project rule (``Yes`` -> ``1``).

    Args:
        sample_customers: Invented customers.
        schema: Feature schema.

    Returns:
        Integer labels, ``1`` meaning churn.
    """
    return schema.encode_target(sample_customers[schema.target_column])


@pytest.fixture()
def sample_pipeline(
    config: TrainingConfig,
    schema: FeatureSchema,
    sample_features: pd.DataFrame,
    sample_labels: pd.Series,
) -> Pipeline:
    """Fits the project's real pipeline builder on the invented customers.

    The preprocessing comes from the committed configuration and the
    classifier is a logistic regression with the configured seed, so a
    change in ``ChurnPipelineBuilder`` or in the preprocessing policy changes
    what these tests see.

    Args:
        config: Project configuration.
        schema: Feature schema.
        sample_features: Six raw input columns.
        sample_labels: Encoded target.

    Returns:
        A fitted preprocessing-plus-classifier pipeline.
    """
    classifier = EstimatorRegistry(config.split.seed).create(
        EstimatorConfig(name="lr", estimator="logistic_regression", params={"max_iter": 1000})
    )
    pipeline = ChurnPipelineBuilder(schema, config.preprocessing).build(classifier)
    pipeline.fit(schema.parse_features(sample_features), sample_labels)
    return pipeline


@pytest.fixture()
def sample_predictor(sample_pipeline: Pipeline, schema: FeatureSchema) -> ChurnPredictor:
    """Wraps the fitted pipeline in the project's inference boundary.

    Args:
        sample_pipeline: Pipeline fitted on the invented customers.
        schema: Feature schema.

    Returns:
        The predictor that validates, parses and scores raw records.
    """
    return ChurnPredictor(sample_pipeline, schema, threshold=SAMPLE_THRESHOLD)
