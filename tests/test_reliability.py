import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from telco_churn.artifacts import ArtifactStore
from telco_churn.dataset import PROJECT_ROOT
from telco_churn.features import FeatureSchema
from telco_churn.pipeline import PipelineInspector
from telco_churn.prediction import ChurnPredictor
from telco_churn.splitting import DatasetSplitter
from telco_churn.training import TrainingWorkflow


class TestSplits:
    """Customer allocation is disjoint, stratified and reproducible."""

    def test_customer_ids_do_not_overlap(self, splits):
        """No customer ID appears in two partitions, and none is lost."""
        ids = {name: set(frame[splits.id_column]) for name, frame in splits.items()}
        assert not ids["train"] & ids["validation"]
        assert not ids["train"] & ids["test"]
        assert not ids["validation"] & ids["test"]
        assert sum(len(v) for v in ids.values()) == 7043

    def test_same_seed_gives_same_allocation(self, config, schema, data, splits):
        """Re-splitting with the configured seed reproduces the allocation."""
        again = DatasetSplitter(config.split, schema.id_column, schema.target_column).split(
            data.records
        )
        pd.testing.assert_frame_equal(again.allocation(), splits.allocation())

    def test_sizes_and_stratification(self, splits, schema):
        """Partitions are about 60/20/20 and keep the churn rate."""
        sizes = splits.sizes()
        assert sizes == {"train": 4225, "validation": 1409, "test": 1409}
        rates = [
            frame[schema.target_column].eq(schema.positive_label).mean()
            for _, frame in splits.items()
        ]
        assert max(rates) - min(rates) < 0.005


class TestTrainingOnlyStatistics:
    """Learned preprocessing statistics come from the training partition only."""

    def test_imputer_medians_match_training_data(self, fitted_pipeline, prepared, schema):
        """Numeric fill values equal the training medians, not full-data medians."""
        learned = PipelineInspector.learned_statistics(fitted_pipeline)["num_imputer_statistics"]
        train = prepared.features["train"][list(schema.numeric)]
        everything = pd.concat([prepared.features[name] for name in prepared.features])
        for column in schema.numeric:
            assert learned[column] == pytest.approx(train[column].median())
        full_medians = everything[list(schema.numeric)].median()
        assert any(
            learned[column] != pytest.approx(full_medians[column]) for column in schema.numeric
        ), "check cannot tell training-only from full-data statistics"

    def test_scaler_means_match_imputed_training_data(self, fitted_pipeline, prepared, schema):
        """Scaler means equal the means of the median-imputed training columns."""
        stats = PipelineInspector.learned_statistics(fitted_pipeline)
        train = prepared.features["train"][list(schema.numeric)]
        imputed = train.fillna(train.median())
        for column in schema.numeric:
            assert stats["num_scaler"]["mean_"][column] == pytest.approx(imputed[column].mean())

    def test_prediction_does_not_refit(self, fitted_pipeline, prepared):
        """Predicting on other data, even shifted data, leaves fitted state unchanged."""
        before = copy.deepcopy(PipelineInspector.learned_statistics(fitted_pipeline))
        shifted = prepared.features["validation"].copy()
        shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 10
        fitted_pipeline.predict_proba(prepared.features["validation"])
        fitted_pipeline.predict_proba(shifted)
        assert PipelineInspector.learned_statistics(fitted_pipeline) == before


class TestDocumentedInputPolicy:
    """Permitted missingness and unseen categories follow the documented policy."""

    def test_blank_total_charges_is_imputed_with_training_median(
        self, fitted_pipeline, customer, schema
    ):
        """A blank TotalCharges is transformed exactly like the training median."""
        median = PipelineInspector.learned_statistics(fitted_pipeline)["num_imputer_statistics"][
            "TotalCharges"
        ]
        blank = customer.assign(TotalCharges="  ")
        filled = customer.assign(TotalCharges=str(median))
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        np.testing.assert_allclose(
            preprocessor.transform(schema.parse_features(blank)),
            preprocessor.transform(schema.parse_features(filled)),
        )

    def test_unseen_category_is_encoded_as_all_zeros(self, fitted_pipeline, customer, schema):
        """An unseen Contract value activates no Contract indicator, and still scores."""
        unseen = schema.parse_features(customer.assign(Contract="Five year"))
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        names = list(preprocessor.get_feature_names_out())
        encoded = preprocessor.transform(unseen)[0]
        contract = [i for i, name in enumerate(names) if name.startswith("cat__Contract_")]
        assert contract and np.all(encoded[contract] == 0)
        probability = fitted_pipeline.predict_proba(unseen)[0, 1]
        assert 0.0 <= probability <= 1.0

    def test_malformed_numeric_text_is_rejected(self, customer, schema):
        """Non-blank text that is not a number is an error, not a missing value."""
        with pytest.raises(ValueError, match="TotalCharges"):
            schema.parse_features(customer.assign(TotalCharges="twelve"))


class TestSchemaFailures:
    """A missing required feature fails visibly instead of predicting silently."""

    def test_missing_column_fails_at_the_boundary(self, fitted_pipeline, customer, schema):
        """The predictor raises KeyError naming the missing column."""
        predictor = ChurnPredictor(fitted_pipeline, schema, threshold=0.5)
        with pytest.raises(KeyError, match="TotalCharges"):
            predictor.predict(customer.drop(columns=["TotalCharges"]))

    def test_missing_column_also_fails_inside_the_pipeline(self, fitted_pipeline, prepared):
        """Even without the boundary check, the ColumnTransformer refuses the batch."""
        with pytest.raises(ValueError):
            fitted_pipeline.predict_proba(prepared.features["validation"].drop(columns=["tenure"]))

    def test_schema_refuses_id_or_target_as_input(self):
        """The ID and target can never be configured as model inputs."""
        with pytest.raises(ValueError):
            FeatureSchema("customerID", "Churn", "Yes", "No", ("tenure", "customerID"), ())

    def test_unexpected_target_label_is_rejected(self, schema):
        """Only the two documented labels can be encoded."""
        with pytest.raises(ValueError):
            schema.encode_target(pd.Series(["Yes", "Maybe"]))


class TestArtifactReload:
    """The saved artifact reloads in a fresh process and predicts identically."""

    @pytest.fixture()
    def trained(self, isolated_config):
        """Runs the full training workflow into a temporary folder.

        Args:
            isolated_config: Configuration with outputs in ``tmp_path``.

        Returns:
            The workflow summary.
        """
        return TrainingWorkflow(isolated_config).run()

    def test_fresh_process_reload_matches(self, trained, isolated_config, tmp_path):
        """A separate interpreter reproduces the saved predictions without fitting."""
        report = tmp_path / "reload_check.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "verify_artifact.py"),
                "--artifact",
                str(isolated_config.outputs.artifact_dir),
                "--report",
                str(report),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        record = json.loads(report.read_text(encoding="utf-8"))
        assert record["passed"] and record["max_abs_probability_difference"] <= 1e-12

    def test_selection_never_used_test_set(self, trained):
        """The comparison is computed on validation and flags the test set unused."""
        assert trained["comparison"]["evaluated_on"] == "validation"
        assert trained["comparison"]["test_set_used"] is False

    def test_tampered_artifact_is_refused(self, trained, isolated_config):
        """A pipeline file that no longer matches its recorded hash is not unpickled."""
        pipeline_file: Path = isolated_config.outputs.artifact_dir / ArtifactStore.PIPELINE
        pipeline_file.write_bytes(pipeline_file.read_bytes() + b"\0")
        with pytest.raises(ValueError, match="refusing to unpickle"):
            ArtifactStore(isolated_config.outputs.artifact_dir).load()
