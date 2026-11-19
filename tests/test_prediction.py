import copy

import numpy as np

from telco_churn.pipeline import PipelineInspector

POSITIVE_CLASS = 1
THRESHOLD = 0.5


class TestProbabilityContract:
    """Test C: probabilities are well formed and the churn column is found by label.

    Class order documented here: the target is encoded ``No`` -> ``0`` and
    ``Yes`` -> ``1``, so ``classes_`` is ``[0, 1]`` and the churn probability
    is the column whose entry in ``classes_`` is ``1``.
    """

    def test_probability_matrix_is_well_formed(self, sample_pipeline, sample_features, schema):
        """One row per customer, one column per class, valid probabilities."""
        features = schema.parse_features(sample_features)

        probabilities = sample_pipeline.predict_proba(features)

        assert probabilities.shape == (len(sample_features), 2)
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0.0).all()
        assert (probabilities <= 1.0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)

    def test_positive_class_is_looked_up_in_classes(
        self, sample_pipeline, sample_predictor, sample_features, schema
    ):
        """The reported churn probability is the ``classes_ == 1`` column.

        Defect caught: taking a fixed column position, which would return the
        no-churn probability if the class order ever changed.
        """
        assert list(sample_pipeline.classes_) == [0, POSITIVE_CLASS]
        positive_column = list(sample_pipeline.classes_).index(POSITIVE_CLASS)
        features = schema.parse_features(sample_features)
        expected = sample_pipeline.predict_proba(features)[:, positive_column]

        result = sample_predictor.predict(sample_features)

        np.testing.assert_allclose(result["churn_probability"].to_numpy(), expected)

    def test_predictions_follow_the_threshold_rule(self, sample_predictor, sample_features):
        """A customer is predicted as churn exactly when the score reaches the threshold."""
        result = sample_predictor.predict(sample_features)
        expected = (result["churn_probability"].to_numpy() >= THRESHOLD).astype("int64")
        np.testing.assert_array_equal(result["churn_prediction"].to_numpy(), expected)

    def test_one_output_row_per_customer_in_the_same_order(
        self, sample_predictor, sample_customers
    ):
        """Output rows line up with input customers, including a one-row batch."""
        result = sample_predictor.predict(sample_customers)
        assert result["customerID"].tolist() == sample_customers["customerID"].tolist()
        assert list(result.columns) == ["customerID", "churn_probability", "churn_prediction"]

        single = sample_predictor.predict(sample_customers.iloc[[5]])
        assert len(single) == 1
        assert single["churn_probability"].iloc[0] == result["churn_probability"].iloc[5]

    def test_scores_respond_to_the_inputs(self, sample_predictor, sample_features):
        """A long two-year customer scores lower than a new month-to-month one.

        A direction check, not an accuracy claim: it catches a model that
        ignores its inputs, for example one fitted on shuffled columns.
        """
        result = sample_predictor.predict(sample_features)
        new_monthly = result["churn_probability"].iloc[0]
        long_two_year = result["churn_probability"].iloc[10]
        assert new_monthly > long_two_year


class TestInferenceIsStateless:
    """Predicting never changes what the pipeline learned during ``fit``."""

    def test_prediction_leaves_learned_statistics_unchanged(
        self, sample_pipeline, sample_predictor, sample_features
    ):
        """Imputer medians, scaler parameters and known categories are untouched.

        Defect caught: an inference path that calls ``fit_transform`` and so
        refits the imputer or scaler on the batch being scored.
        """
        before = copy.deepcopy(PipelineInspector.learned_statistics(sample_pipeline))
        shifted = sample_features.copy()
        shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 10
        shifted.loc[0, "Contract"] = "Three year"

        sample_predictor.predict(shifted)

        assert PipelineInspector.learned_statistics(sample_pipeline) == before
