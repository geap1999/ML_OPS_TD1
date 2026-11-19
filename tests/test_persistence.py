import joblib
import numpy as np
import pytest

from telco_churn.artifacts import ArtifactStore
from telco_churn.prediction import ChurnPredictor


class TestJoblibRoundTrip:
    """Test D: a pipeline saved with joblib and reloaded behaves the same.

    The file lives in pytest's ``tmp_path``, so the TP02 and TP03 models are
    never overwritten. Only artifacts written by the test itself are loaded.
    """

    def test_reload_preserves_predictions(
        self, tmp_path, sample_pipeline, sample_features, schema
    ):
        """Reloaded probabilities match within 1e-12 and classes match exactly."""
        artifact = tmp_path / "pipeline.joblib"
        features = schema.parse_features(sample_features)
        probabilities_before = sample_pipeline.predict_proba(features)
        classes_before = sample_pipeline.predict(features)

        joblib.dump(sample_pipeline, artifact)
        reloaded = joblib.load(artifact)

        assert reloaded is not sample_pipeline
        np.testing.assert_allclose(
            reloaded.predict_proba(features), probabilities_before, rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(reloaded.predict(features), classes_before)

    def test_reloaded_object_still_contains_the_preprocessing(
        self, tmp_path, sample_pipeline, sample_features, schema
    ):
        """The saved object accepts unencoded columns, so it is not the classifier alone.

        Defect caught: saving only the classifier, which would need already
        scaled and one-hot encoded inputs and so could not score raw records.
        """
        artifact = tmp_path / "pipeline.joblib"
        joblib.dump(sample_pipeline, artifact)
        reloaded = joblib.load(artifact)

        assert list(reloaded.named_steps) == ["preprocessor", "classifier"]
        predictions = reloaded.predict(schema.parse_features(sample_features))
        assert len(predictions) == len(sample_features)


class TestArtifactStoreRoundTrip:
    """The project's own save helper stores a complete, verifiable pipeline."""

    @pytest.fixture()
    def saved(self, tmp_path, sample_pipeline, sample_predictor, sample_customers, schema):
        """Saves the fitted pipeline with ``ArtifactStore`` into a temporary folder.

        Args:
            tmp_path: Per-test temporary directory.
            sample_pipeline: Pipeline fitted on the invented customers.
            sample_predictor: Inference boundary around that pipeline.
            sample_customers: Invented customers.
            schema: Feature schema.

        Returns:
            The store, pointing at the written artifact directory.
        """
        inputs = sample_customers[[schema.id_column, *schema.feature_columns]]
        store = ArtifactStore(tmp_path / "churn_pipeline")
        store.save(
            sample_pipeline,
            metadata={"features": schema.describe(), "selection": {"threshold": 0.5}},
            allocation=sample_customers[[schema.id_column]].assign(split="train"),
            reference_inputs=inputs,
            reference_predictions=sample_predictor.predict(inputs),
        )
        return store

    def test_loaded_artifact_reproduces_reference_predictions(self, saved):
        """Predicting the saved raw batch after loading gives the saved predictions.

        The reference batch is read back from CSV as text, so this also checks
        that the input parsing gives the same numbers after a round trip.
        """
        artifact = saved.load()
        predictor = ChurnPredictor.from_loaded(artifact)

        actual = predictor.predict(artifact.reference_inputs())
        expected = artifact.reference_predictions()

        np.testing.assert_allclose(
            actual["churn_probability"].to_numpy(),
            expected["churn_probability"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            actual["churn_prediction"].to_numpy(), expected["churn_prediction"].to_numpy()
        )

    def test_modified_pipeline_file_is_refused(self, saved):
        """A pipeline file that no longer matches its recorded hash is not unpickled."""
        pipeline_file = saved.directory / ArtifactStore.PIPELINE
        pipeline_file.write_bytes(pipeline_file.read_bytes() + b"\0")
        with pytest.raises(ValueError, match="refusing to unpickle"):
            saved.load()
