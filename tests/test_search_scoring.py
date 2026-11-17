"""Scoring a run against the exact front, as a function of what it spent.

Both properties tested here are departures from what NAS papers usually report, and both
matter for the claim the second paper makes.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymoo")

from nas4textgen.search import igd_plus, normalise, score_trace  # noqa: E402

# A simple convex front on [0, 1]^2, minimised.
TRUE_FRONT = np.array([[0.0, 1.0], [0.25, 0.5], [0.5, 0.25], [1.0, 0.0]])
LOWER = np.zeros(2)
UPPER = np.ones(2)


class TestIgdPlus:
    def test_recovering_the_front_exactly_scores_zero(self):
        assert igd_plus(TRUE_FRONT, TRUE_FRONT) == pytest.approx(0.0)

    def test_a_worse_front_scores_higher(self):
        near = TRUE_FRONT + 0.05
        far = TRUE_FRONT + 0.4
        assert igd_plus(near, TRUE_FRONT) < igd_plus(far, TRUE_FRONT)

    def test_dominating_the_reference_is_not_penalised(self):
        """The correction over plain IGD. A point better than the reference in every
        objective is further away in Euclidean terms and should still score zero — plain
        IGD punishes it for having gone past."""
        better = TRUE_FRONT - 0.2
        assert igd_plus(better, TRUE_FRONT) == pytest.approx(0.0)

        euclidean = np.sqrt(
            ((TRUE_FRONT[:, None, :] - better[None, :, :]) ** 2).sum(axis=2)
        ).min(axis=1).mean()
        assert euclidean > 0.2, "plain IGD would charge for dominating the reference"

    def test_an_empty_front_is_infinitely_far(self):
        assert igd_plus(np.empty((0, 2)), TRUE_FRONT) == float("inf")


class TestNormalisation:
    def test_bounds_come_from_the_measured_space_not_the_run(self):
        """Normalising per run makes each score depend on its own spread, so a run that
        found a narrow slice of the front looks well-converged."""
        narrow = np.array([[0.40, 0.30], [0.42, 0.29]])
        table_lower, table_upper = np.zeros(2), np.ones(2) * 100

        shared = normalise(narrow, table_lower, table_upper)
        per_run = normalise(narrow, narrow.min(axis=0), narrow.max(axis=0))

        assert shared.max() < 0.01, "on the table's scale this run covers almost nothing"
        assert per_run.max() == pytest.approx(1.0), "on its own scale it looks complete"

    def test_a_degenerate_objective_does_not_divide_by_zero(self):
        points = np.array([[1.0, 5.0], [2.0, 5.0]])
        result = normalise(points, np.array([1.0, 5.0]), np.array([2.0, 5.0]))
        assert np.isfinite(result).all()


class TestQueryTrace:
    def _trace(self, order: np.ndarray, **kw):
        return score_trace(order, TRUE_FRONT, LOWER, UPPER, **kw)

    def test_the_trace_has_one_entry_per_true_evaluation(self):
        queried = np.array([[0.8, 0.8], [0.25, 0.5], [0.5, 0.25]])
        trace = self._trace(queried)
        assert list(trace.evaluations) == [1, 2, 3]
        assert len(trace.igd_plus) == 3

    def test_igd_plus_improves_as_front_points_are_discovered(self):
        queried = np.array([[0.9, 0.9], [0.25, 0.5], [0.5, 0.25], [0.0, 1.0], [1.0, 0.0]])
        trace = self._trace(queried)
        assert list(trace.igd_plus) == sorted(trace.igd_plus, reverse=True)
        assert trace.igd_plus[-1] == pytest.approx(0.0)

    def test_budget_lookup_reports_what_was_known_at_that_point(self):
        """The question a practitioner asks: how good is this after N trainings, since N
        trainings is what they can afford."""
        queried = np.array([[0.9, 0.9], [0.9, 0.9], [0.25, 0.5], [0.5, 0.25],
                            [0.0, 1.0], [1.0, 0.0]])
        trace = self._trace(queried)
        assert trace.at_budget(2) > trace.at_budget(6)
        assert trace.at_budget(6) == pytest.approx(0.0)
        assert trace.at_budget(0) == float("inf")

    def test_evaluations_to_reach_states_efficiency_the_way_it_is_used(self):
        """Not 'how good after N' but 'how many trainings to get this good'."""
        queried = np.array([[0.9, 0.9], [0.25, 0.5], [0.5, 0.25], [0.0, 1.0], [1.0, 0.0]])
        trace = self._trace(queried)
        assert trace.evaluations_to_reach(0.0) == 5
        assert trace.evaluations_to_reach(-1.0) is None

    def test_a_longer_run_cannot_be_flattered_by_its_endpoint_alone(self):
        """Final hypervolume rewards whoever was given the most budget. The trace does
        not: a run that reached the front in three evaluations is visibly better than one
        that took twenty, even though both end identically."""
        front = np.array([[0.0, 1.0], [0.25, 0.5], [0.5, 0.25], [1.0, 0.0]])
        fast = self._trace(front)
        slow = self._trace(np.vstack([np.full((16, 2), 0.9), front]))

        assert fast.igd_plus[-1] == pytest.approx(slow.igd_plus[-1]), "same endpoint"
        assert fast.at_budget(4) < slow.at_budget(4), "different efficiency"
        assert fast.evaluations_to_reach(1e-9) < slow.evaluations_to_reach(1e-9)

    def test_wasted_queries_are_carried_in_the_record(self):
        trace = self._trace(TRUE_FRONT, infeasible_queries=17, duplicate_queries=42)
        payload = trace.as_dict()
        assert payload["infeasible_queries"] == 17
        assert payload["duplicate_queries"] == 42


class TestHypervolumeCurves:
    """Three curves, because they answer three questions and one of them is a trap."""

    def test_the_archive_curve_rises_for_any_algorithm_including_random(self):
        """Which is why it cannot, on its own, say anything about the search.

        The archive only grows, so its hypervolume is non-decreasing by construction.
        Reporting it alone would make random search look as well-behaved as a method with
        a real monotonicity guarantee.
        """
        rng = np.random.default_rng(0)
        queried = rng.random((40, 2))
        trace = score_trace(queried, TRUE_FRONT, LOWER, UPPER)

        assert np.all(np.diff(trace.hypervolume) >= -1e-12)
        assert np.all(np.diff(trace.hypervolume_unnormalised) >= -1e-12)

    def test_the_unnormalised_curve_is_in_the_objectives_own_units(self):
        """A 3% normalised improvement is uninterpretable until it is a number of
        parameters and a number of perplexity points."""
        lower = np.array([3.3e6, 200.0])
        upper = np.array([1.45e8, 900.0])
        queried = np.array([[1.0e7, 400.0], [5.0e7, 300.0]])

        trace = score_trace(queried, np.array([[3.3e6, 250.0]]), lower, upper)

        assert trace.hypervolume[-1] <= 2.0, "normalised lives in a unit box"
        assert trace.hypervolume_unnormalised[-1] > 1e9, "raw is in parameter-perplexity"

    def test_a_population_curve_can_fall_where_the_archive_cannot(self):
        """The distinction the archive curve hides: what the algorithm carries is not what
        it has seen, and a truncating survival operator can discard hypervolume."""
        queried = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5], [0.95, 0.95]])
        # The algorithm keeps only the newest point at each step.
        snapshots = [queried[i : i + 1] for i in range(len(queried))]

        trace = score_trace(queried, TRUE_FRONT, LOWER, UPPER,
                            population_snapshots=snapshots)

        assert np.all(np.diff(trace.hypervolume) >= -1e-12), "the archive still rises"
        assert not trace.population_is_monotone, "the population does not"

    def test_without_snapshots_the_population_curve_mirrors_the_archive(self):
        """Honest but uninformative, and the docstring says so rather than implying the
        run was monotone."""
        rng = np.random.default_rng(1)
        trace = score_trace(rng.random((20, 2)), TRUE_FRONT, LOWER, UPPER)
        assert np.allclose(trace.population_hypervolume, trace.hypervolume)
