from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from telco_churn.dataset import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.json"


@dataclass(frozen=True)
class SplitConfig:
    """Settings of the stratified train/validation/test split.

    Attributes:
        seed: Random seed shared by both split stages.
        train_fraction: Share of customers used for fitting.
        validation_fraction: Share used to compare candidates.
        test_fraction: Share reserved and never used for selection.
    """

    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        """Checks that the three fractions describe a full partition.

        Raises:
            ValueError: If the fractions do not sum to 1 or one is not positive.
        """
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(f <= 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must be positive and sum to 1, got {fractions}.")


@dataclass(frozen=True)
class EstimatorConfig:
    """One classifier to build, identified by a registry key.

    Attributes:
        name: Human-readable candidate name used in reports.
        estimator: Key of the estimator in ``EstimatorRegistry``.
        params: Keyword arguments passed to the estimator constructor.
        complexity_rank: Lower means simpler; used to break selection ties.
    """

    name: str
    estimator: str
    params: dict[str, Any] = field(default_factory=dict)
    complexity_rank: int = 0


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing recipe applied identically to every candidate.

    Attributes:
        numeric_imputer: ``SimpleImputer`` strategy for numeric inputs.
        numeric_scaler: ``"standard"`` or ``"minmax"``.
        categorical_imputer: ``SimpleImputer`` strategy for categorical inputs.
        unknown_categories: ``OneHotEncoder`` ``handle_unknown`` policy.
    """

    numeric_imputer: str
    numeric_scaler: str
    categorical_imputer: str
    unknown_categories: str


@dataclass(frozen=True)
class SelectionConfig:
    """Selection rule fixed before looking at the comparison.

    Attributes:
        metric: Validation metric to maximise.
        tolerance: Score gap under which two candidates count as tied.
        rule: Plain-language statement of the rule, copied into reports.
        threshold: Probability cut-off for class predictions.
    """

    metric: str
    tolerance: float
    rule: str
    threshold: float


@dataclass(frozen=True)
class OutputConfig:
    """Where each run writes its reports and artifact.

    Attributes:
        first_model_report: JSON report of the milestone-1 model.
        comparison_json: Machine-readable comparison table (JSON).
        comparison_csv: Same comparison table as CSV.
        artifact_dir: Directory holding the fitted pipeline and its metadata.
    """

    first_model_report: Path
    comparison_json: Path
    comparison_csv: Path
    artifact_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    """Complete, validated content of ``configs/train.json``.

    Every run setting (split seed, feature roles, candidates, selection rule,
    output paths) lives in the configuration file rather than in source code;
    this class turns it into immutable, typed objects so the rest of the
    package never reads raw dictionaries.

    Attributes:
        path: File the configuration was read from.
        source_path: Raw CSV used for training.
        split: Split settings.
        features: Raw feature-role mapping (see ``FeatureSchema.from_config``).
        first_model_features: Inputs of the milestone-1 model.
        first_model: Estimator of the milestone-1 model.
        preprocessing: Shared preprocessing recipe.
        candidates: Baseline and learned classifiers to compare.
        selection: Selection rule.
        outputs: Output locations.
        raw: The parsed JSON, kept verbatim for run metadata.
    """

    path: Path
    source_path: Path
    split: SplitConfig
    features: dict[str, Any]
    first_model_features: list[str]
    first_model: EstimatorConfig
    preprocessing: PreprocessingConfig
    candidates: list[EstimatorConfig]
    selection: SelectionConfig
    outputs: OutputConfig
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path = DEFAULT_CONFIG_PATH) -> TrainingConfig:
        """Reads and validates a training configuration file.

        Relative paths inside the file are resolved against the project root,
        so the same configuration works from any working directory.

        Args:
            path: JSON configuration file.

        Returns:
            The parsed configuration.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If a required section or field is missing.
            ValueError: If the split fractions are inconsistent.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        outputs = raw["outputs"]
        first = raw["first_model"]
        return cls(
            path=Path(path),
            source_path=cls._resolve(raw["data"]["source_path"]),
            split=SplitConfig(**raw["split"]),
            features=raw["features"],
            first_model_features=list(first["features"]),
            first_model=EstimatorConfig(
                name="first_model", estimator=first["estimator"], params=first["params"]
            ),
            preprocessing=PreprocessingConfig(**raw["preprocessing"]),
            candidates=[EstimatorConfig(**candidate) for candidate in raw["candidates"]],
            selection=SelectionConfig(**raw["selection"]),
            outputs=OutputConfig(**{key: cls._resolve(value) for key, value in outputs.items()}),
            raw=raw,
        )

    @staticmethod
    def _resolve(relative: str) -> Path:
        """Resolves a configuration path against the project root.

        Args:
            relative: Path as written in the configuration file.

        Returns:
            An absolute path.
        """
        return (PROJECT_ROOT / relative).resolve()
