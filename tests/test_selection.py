from dataclasses import replace

import pytest

from telco_churn.config import EstimatorConfig, SelectionConfig
from telco_churn.evaluation import EvaluationResult
from telco_churn.training import CandidateResult, ModelSelector


class TestSelectionRule:
    """The pre-registered rule: best validation score, ties go to the simpler model.

    Added after reading the coverage report: the tie-break branch of
    ``ModelSelector.select`` was never executed, because on the real data no
    candidate is within the tolerance of the best one. The candidates here
    are hand-made results, so no model is trained.
    """

    @pytest.fixture()
    def selector(self) -> ModelSelector:
        """Builds a selector with the project's metric and a 0.01 tolerance.

        Returns:
            The selector under test.
        """
        return ModelSelector(
            SelectionConfig(metric="average_precision", tolerance=0.01, rule="", threshold=0.5)
        )

    @staticmethod
    def candidate(name: str, estimator: str, rank: int, score: float) -> CandidateResult:
        """Makes a candidate result with a chosen validation average precision.

        Args:
            name: Candidate name.
            estimator: Registry key; ``"dummy"`` marks the baseline.
            rank: Complexity rank, lower meaning simpler.
            score: Validation average precision to report.

        Returns:
            A result that carries no fitted pipeline.
        """
        validation = EvaluationResult(
            n_samples=100,
            positive_rate=0.25,
            score_type="predict_proba",
            threshold=0.5,
            average_precision=score,
            roc_auc=0.8,
            accuracy=0.8,
            precision=0.6,
            recall=0.5,
            f1=0.55,
            confusion_matrix={},
        )
        config = EstimatorConfig(name=name, estimator=estimator, complexity_rank=rank)
        return CandidateResult(
            config=config, pipeline=None, fit_seconds=0.0, validation=validation
        )

    def test_simpler_candidate_wins_inside_the_tolerance(self, selector):
        """A simpler model 0.005 behind the best is selected instead of it."""
        results = [
            self.candidate("logistic", "logistic_regression", rank=1, score=0.655),
            self.candidate("forest", "random_forest", rank=3, score=0.660),
        ]
        decision = selector.select(results)
        assert decision.selected == "logistic"
        assert decision.best_by_metric == "forest"
        assert decision.tied == ["logistic", "forest"]

    def test_best_candidate_wins_outside_the_tolerance(self, selector):
        """A simpler model 0.02 behind the best does not displace it."""
        results = [
            self.candidate("logistic", "logistic_regression", rank=1, score=0.640),
            self.candidate("forest", "random_forest", rank=3, score=0.660),
        ]
        decision = selector.select(results)
        assert decision.selected == "forest"
        assert decision.tied == ["forest"]

    def test_baseline_is_never_selected(self, selector):
        """The majority baseline is context only, even with the highest score."""
        results = [
            self.candidate("baseline", "dummy", rank=0, score=0.990),
            self.candidate("tree", "decision_tree", rank=2, score=0.600),
        ]
        assert selector.select(results).selected == "tree"

    def test_selection_without_a_learned_candidate_is_an_error(self, selector):
        """With only the baseline there is nothing to select."""
        baseline = self.candidate("baseline", "dummy", rank=0, score=0.265)
        with pytest.raises(ValueError, match="at least one learned candidate"):
            selector.select([baseline])

    def test_selection_ignores_metrics_other_than_the_configured_one(self, selector):
        """A higher recall does not change a decision made on average precision."""
        forest = self.candidate("forest", "random_forest", rank=3, score=0.660)
        tree = self.candidate("tree", "decision_tree", rank=2, score=0.600)
        tree = replace(tree, validation=replace(tree.validation, recall=0.99, f1=0.99))
        assert selector.select([tree, forest]).selected == "forest"

    def test_tied_candidates_are_listed_simplest_first(self, selector):
        """Candidates inside the tolerance are reported in order of complexity."""
        results = [
            self.candidate("forest", "random_forest", rank=3, score=0.660),
            self.candidate("tree", "decision_tree", rank=2, score=0.655),
            self.candidate("logistic", "logistic_regression", rank=1, score=0.652),
        ]
        assert selector.select(results).tied == ["logistic", "tree", "forest"]
