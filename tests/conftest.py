from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from telco_churn.config import DEFAULT_CONFIG_PATH, EstimatorConfig, OutputConfig, TrainingConfig
from telco_churn.features import FeatureSchema
from telco_churn.pipeline import ChurnPipelineBuilder, EstimatorRegistry
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
