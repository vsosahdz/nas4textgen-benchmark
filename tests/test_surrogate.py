"""The slate, and the rule that a predictor must beat the floors to count.

Constructed data throughout, so the right answer is known. The measures are the interesting
part: global rank, rank restricted to the good end, and precision at a budget, because they
answer different questions and on the real table they disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.surrogate import (
    EXCLUDED,
    PRECISION_AT,
    build_slate,
    kendall_tau,
    precision_at_k,
    report,
    score_slate,
    top_decile_tau,
)


class TestMeasures:
    def test_a_perfect_ordering_scores_one(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        assert kendall_tau(actual, actual) == pytest.approx(1.0)

    def test_a_reversed_ordering_scores_minus_one(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        assert kendall_tau(-actual, actual) == pytest.approx(-1.0)

    def test_precision_at_k_counts_overlap_not_correlation(self):
        """The measure a budget is spent against: of the k runs recommended, how many were
        worth doing. Unlike a correlation it does not care about the rest of the ordering."""
        actual = np.array([1.0, 2.0, 3.0, 100.0, 200.0])
        # Gets the best three right, scrambles the rest.
        predicted = np.array([3.0, 1.0, 2.0, 500.0, 400.0])
        assert precision_at_k(predicted, actual, 3) == 1.0
        assert kendall_tau(predicted, actual) < 1.0

    def test_the_top_decile_measure_ignores_the_easy_part(self):
        """A model can score well globally by getting the gap between tiny and large
        architectures right — which parameter count already gets right — while shuffling
        everything near the front, where the ordering is all that matters."""
        # A hundred architectures: the best ten are tightly packed, the rest spread wide.
        actual = np.concatenate([np.arange(10.0), np.arange(100, 190.0)])
        predicted = actual.copy()
        predicted[:10] = predicted[:10][::-1]   # only the good end is wrong, and entirely

        assert kendall_tau(predicted, actual) > 0.95, "globally it looks fine"
        assert top_decile_tau(predicted, actual) == pytest.approx(-1.0), (
            "at the good end the ordering is exactly backwards"
        )

    def test_precision_is_undefined_rather_than_zero_beyond_the_sample(self):
        actual = np.array([1.0, 2.0])
        assert np.isnan(precision_at_k(actual, actual, 10))


class TestDisqualification:
    def _data(self, n=120, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.integers(1, 6, size=(n, 5)).astype(float)
        params = X[:, 0] * 1e6 + rng.normal(0, 1e4, n)
        # Perplexity falls with the first gene, plus an interaction a linear model misses.
        y = 500 / X[:, 0] + 20 * (X[:, 1] == X[:, 2]) + rng.normal(0, 2, n)
        return X, y, params

    def test_a_candidate_beating_neither_floor_is_disqualified(self):
        """A predictor that cannot outrank sorting by parameter count is not a predictor,
        and reporting that is more useful than hiding it."""
        from nas4textgen.surrogate.slate import Candidate

        X, y, params = self._data()
        noise = Candidate(
            "pure_noise", "tier1",
            lambda: _ConstantModel(), why="a control that must be disqualified",
        )
        scores = {s.name: s for s in score_slate(build_slate()[:2] + [noise], X, y, params)}
        assert scores["pure_noise"].disqualified

    def test_the_floors_are_scored_like_everything_else(self):
        X, y, params = self._data()
        scores = {s.name: s for s in score_slate(build_slate()[:4], X, y, params)}
        assert not np.isnan(scores["sort_by_parameters"].tau)
        assert not np.isnan(scores["linear_on_genes"].tau)

    def test_the_report_names_the_floor_and_the_gain(self):
        X, y, params = self._data()
        rep = report(score_slate(build_slate()[:4], X, y, params), len(y), 1099)
        assert rep["trivial_floors"]["sort_by_parameters"] is not None
        assert rep["gain_over_floor"] is not None
        assert "marginal gain" in rep["trivial_floors"]["rule"]


class _ConstantModel:
    def fit(self, X, y):  # noqa: N803
        return self

    def predict(self, X):  # noqa: N803
        return np.zeros(len(X))


class TestSlateComposition:
    def test_every_candidate_states_why_it_is_there(self):
        for candidate in build_slate():
            assert len(candidate.why) > 25, f"{candidate.name} has no stated purpose"

    def test_the_slate_contains_both_floors(self):
        names = {c.name for c in build_slate()}
        assert {"sort_by_parameters", "linear_on_genes"} <= names

    def test_at_least_one_candidate_is_fit_on_the_selection_criterion(self):
        """Selection is by Kendall tau, and fitting for squared error while selecting on
        rank is the mismatch the prior work made. The slate contains a model that does not
        make it, whether or not it wins."""
        assert any(c.ranking_loss for c in build_slate())

    def test_the_uncertainty_tier_is_populated(self):
        """A wave-based sweep can use predictive variance to choose what to measure next,
        and a point prediction cannot answer that."""
        assert sum(c.uncertainty for c in build_slate()) >= 2

    def test_the_prior_work_models_are_carried_rather_than_dropped(self):
        """Keeping them lets the paper show they are dominated instead of asserting it."""
        names = {c.name for c in build_slate()}
        assert {"svr_rbf", "mlp"} <= names

    def test_a_missing_backend_is_reported_not_skipped(self):
        """A slate that quietly shrinks to what happened to be installed is a slate whose
        result depends on the machine."""
        X, y, params = np.zeros((30, 5)), np.arange(30.0), np.arange(30.0)
        from nas4textgen.surrogate.slate import Candidate

        absent = Candidate("absent_thing", "tier1", None, why="not installed here")
        scores = {s.name: s for s in score_slate([absent], X, y, params)}
        assert scores["absent_thing"].unavailable == "not installed here"
        assert scores["absent_thing"].disqualified

    def test_every_precision_budget_is_small_enough_to_be_a_real_budget(self):
        assert max(PRECISION_AT) <= 50


class TestExclusions:
    def test_each_excluded_family_says_when_it_would_apply(self):
        """Reviewer R2 asked for graph predictors, and 'we did not try it' is a weaker
        answer than 'it does not apply, and here is the condition under which it would'."""
        assert "graph_neural_network" in EXCLUDED
        for family, reason in EXCLUDED.items():
            assert len(reason) > 100, f"{family} is dismissed without an argument"
            assert "pplicable" in reason or "own study" in reason, (
                f"{family} does not say what would change the answer"
            )


class TestBudgetCurve:
    """How many labels a surrogate needs before it is worth fitting.

    The question the campaign faces, and the trivial floor is what makes it answerable: the
    floor needs no training at all, so a surrogate that only overtakes it at a hundred
    labels has to justify those hundred against a baseline that was free from the start.
    """

    def _data(self, n=200, seed=1):
        rng = np.random.default_rng(seed)
        X = rng.integers(1, 7, size=(n, 5)).astype(float)
        params = X[:, 0] * 1e6 + X[:, 1] * 1e5
        y = 400 / X[:, 0] + 15 * (X[:, 1] == X[:, 2]) + rng.normal(0, 3, n)
        costs = np.column_stack([params, X[:, 4] * 1e5])
        return X, y, costs, params

    def _curve(self, sizes=(25, 50, 100)):
        from nas4textgen.surrogate import measure

        X, y, costs, params = self._data()
        slate = [c for c in build_slate()
                 if c.name in ("sort_by_parameters", "random_forest")]
        return measure(slate, X, y, costs, params, sizes=sizes, repeats=3)

    def test_the_floor_does_not_move_with_training_size(self):
        """It needs no labels, which is the whole reason it is the thing to beat."""
        curve = self._curve()
        floor = curve.for_candidate("sort_by_parameters")
        taus = [p.tau for p in floor]
        assert len(taus) >= 2
        assert max(taus) - min(taus) < 1e-9

    def test_a_learned_model_improves_with_labels(self):
        curve = self._curve()
        points = curve.for_candidate("random_forest")
        assert points[-1].tau > points[0].tau

    def test_every_repeat_is_scored_on_the_same_held_out_set(self):
        """Otherwise top-k stability intersects recommendations over different
        architectures and reports near zero however stable the model is. The first version
        did exactly that and its 0.10 was an artefact.
        """
        curve = self._curve()
        for point in curve.points:
            if point.candidate == "sort_by_parameters":
                # Deterministic and trained on nothing, so its recommendation cannot move.
                assert point.topk_stability == pytest.approx(1.0)

    def test_stability_is_undefined_for_a_single_fit(self):
        """One recommendation is trivially identical to itself, which says nothing about
        whether refitting would change it."""
        from nas4textgen.surrogate import measure

        X, y, costs, params = self._data()
        slate = [c for c in build_slate() if c.name == "random_forest"]
        curve = measure(slate, X, y, costs, params, sizes=(50,), repeats=1)
        assert all(np.isnan(p.topk_stability) for p in curve.points)

    def test_rows_to_beat_the_floor_is_reported_or_absent(self):
        """Absent is a finding: it means the surrogate never overtook a free baseline
        within the budget measured."""
        curve = self._curve()
        payload = curve.as_dict()
        for name, info in payload["per_candidate"].items():
            assert "rows_to_beat_parameter_floor" in info
            if name == "sort_by_parameters":
                assert info["rows_to_beat_parameter_floor"] is None

    def test_the_front_distance_uses_bounds_over_all_objectives(self):
        """Taking them from the costs alone left a two-element vector normalising a
        three-column front, which numpy refused rather than quietly doing something else."""
        curve = self._curve(sizes=(50,))
        assert all(
            p.igd_plus_to_measured_front == p.igd_plus_to_measured_front
            for p in curve.points
        ), "the front distance must be defined"


def test_the_top_decile_set_changes_with_coverage():
    """Why a rise in top-decile tau between two table sizes is not a clean improvement.

    The decile is the best tenth of whatever rows exist, so it names different architectures
    as coverage grows. Comparing 0.4667 at 176 rows to 0.7067 at 321 mixes "the model
    learned more" with "the question changed", and the docstring says so rather than leaving
    a reader to infer it. The budget curve is the comparison that holds the evaluation set
    fixed.
    """
    import inspect

    from nas4textgen.surrogate import top_decile_tau

    doc = inspect.getdoc(top_decile_tau)
    assert "Not comparable across tables of different size" in doc
    assert "budget curve" in doc

    small = np.arange(50.0, 150.0)          # 100 architectures
    large = np.arange(40.0, 240.0)          # 200, including better ones
    # The best tenth of each names a different set, which is the whole point.
    assert set(np.sort(small)[:10]) != set(np.sort(large)[:20])
