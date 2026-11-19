import numpy as np
import pandas as pd
import pytest

REQUIRED_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
]
UNSEEN_CONTRACT = "Three year"


class TestRequiredFeatures:
    """Test A: the six agreed features are required, and nothing else is used."""

    def test_schema_requires_exactly_the_six_agreed_features(self, schema):
        """The configured inputs are the six TP02 features, without ID or target."""
        assert schema.feature_columns == REQUIRED_FEATURES
        assert schema.id_column not in schema.feature_columns
        assert schema.target_column not in schema.feature_columns

    @pytest.mark.parametrize("missing", REQUIRED_FEATURES)
    def test_missing_required_feature_is_rejected(
        self, sample_predictor, sample_features, missing
    ):
        """Dropping any required feature raises KeyError naming that feature.

        Defect caught: a prediction that silently continues with a column
        missing, for example after a rename upstream.
        """
        broken = sample_features.drop(columns=[missing])
        with pytest.raises(KeyError, match=rf"Missing required input column\(s\) \['{missing}'\]"):
            sample_predictor.predict(broken)

    def test_id_and_target_do_not_influence_predictions(
        self, sample_predictor, sample_customers, sample_features
    ):
        """Scores are the same with or without ``customerID`` and ``Churn`` in the batch.

        Defect caught: the target or the identifier leaking into the model
        inputs. The labels are also flipped to prove they are ignored.
        """
        with_extras = sample_customers.copy()
        with_extras["Churn"] = with_extras["Churn"].map({"Yes": "No", "No": "Yes"})
        from_full = sample_predictor.predict(with_extras)
        from_features = sample_predictor.predict(sample_features)
        np.testing.assert_array_equal(
            from_full["churn_probability"].to_numpy(),
            from_features["churn_probability"].to_numpy(),
        )
        assert "customerID" in from_full.columns
        assert "Churn" not in from_full.columns


class TestSupportedInputVariation:
    """Test B: permitted missing values and unknown categories follow the policy."""

    def test_unseen_category_was_really_absent_from_training(
        self, sample_pipeline, sample_features
    ):
        """The category used below is unknown to the fixture and to the fitted encoder."""
        assert UNSEEN_CONTRACT not in set(sample_features["Contract"])
        categorical = sample_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        contract_index = list(categorical.feature_names_in_).index("Contract")
        known = categorical.named_steps["encoder"].categories_[contract_index]
        assert UNSEEN_CONTRACT not in set(known)

    def test_missing_numeric_and_unseen_category_are_scored(
        self, sample_predictor, sample_features
    ):
        """A missing number and an unseen category still give one valid score per row.

        Policy under test: median imputation for numbers and
        ``OneHotEncoder(handle_unknown="ignore")`` for categories. Defect
        caught: switching either policy to a rejection, or a crash on ``NaN``.
        This shows the batch is accepted, not that those scores are accurate.
        """
        varied = sample_features.copy()
        varied.loc[0, "TotalCharges"] = np.nan
        varied.loc[1, "Contract"] = UNSEEN_CONTRACT

        result = sample_predictor.predict(varied)

        assert len(result) == len(varied)
        probabilities = result["churn_probability"].to_numpy()
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
        assert set(result["churn_prediction"].unique()) <= {0, 1}

    def test_variation_in_one_row_leaves_other_rows_unchanged(
        self, sample_predictor, sample_features
    ):
        """Only the modified customers get different scores."""
        varied = sample_features.copy()
        varied.loc[0, "TotalCharges"] = np.nan
        varied.loc[1, "Contract"] = UNSEEN_CONTRACT
        before = sample_predictor.predict(sample_features)["churn_probability"].to_numpy()
        after = sample_predictor.predict(varied)["churn_probability"].to_numpy()
        np.testing.assert_array_equal(after[2:], before[2:])
        assert after[1] != before[1]

    def test_blank_text_is_treated_like_a_missing_number(self, sample_predictor, sample_features):
        """Whitespace-only numeric text scores exactly like ``NaN``.

        This is the Telco CSV case: ``TotalCharges`` is blank for new customers.
        """
        as_nan = sample_features.copy()
        as_nan.loc[0, "TotalCharges"] = np.nan
        as_blank = sample_features.astype({"TotalCharges": str})
        as_blank.loc[0, "TotalCharges"] = "   "
        np.testing.assert_allclose(
            sample_predictor.predict(as_blank)["churn_probability"].to_numpy(),
            sample_predictor.predict(as_nan)["churn_probability"].to_numpy(),
        )

    def test_malformed_numeric_text_is_rejected(self, sample_predictor, sample_features):
        """Non-blank text that is not a number is an error, never a silent ``NaN``.

        Defect caught: replacing the parser with ``errors="coerce"``, which
        would impute typos instead of reporting them.
        """
        malformed = sample_features.astype({"MonthlyCharges": str})
        malformed.loc[3, "MonthlyCharges"] = "seventy"
        with pytest.raises(ValueError, match="'MonthlyCharges' has 1 non-blank value"):
            sample_predictor.predict(malformed)

    def test_prediction_does_not_modify_the_input_batch(self, sample_predictor, sample_features):
        """The caller's frame is left exactly as it was."""
        original = sample_features.copy(deep=True)
        sample_predictor.predict(sample_features)
        pd.testing.assert_frame_equal(sample_features, original)
