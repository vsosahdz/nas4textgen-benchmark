"""Whether the ranking at one budget predicts the ranking at four times it.

The sweep's fixed token budget is what makes it an architecture comparison rather than a
repeat of the prior work's defect. It also puts every architecture at a different distance
from its own compute-optimal point, and the sweep's finding that width beats depth could be
a statement about the budget rather than about the shape. This is the experiment that
separates them.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.azure.budget_study import (
    BUDGET_MULTIPLE,
    compare,
    plan,
    select,
)
from nas4textgen.space.cost import forward_flops_per_token
from nas4textgen.space.enumerate import enumerate_with_ceiling


@pytest.fixture(scope="module")
def measured():
    """Stand-in for what the sweep has measured: a spread over the space."""
    space = enumerate_with_ceiling().genotypes
    return list(space[::3])


class TestSelection:
    def test_the_subset_spans_the_compute_range(self):
        """A sample that missed the expensive end could not see the effect it exists to
        look for: the budget's distance from compute-optimal is a function of size."""
        space = enumerate_with_ceiling().genotypes
        chosen = select(list(space), size=50)

        flops = [m.flops_per_token for m in chosen]
        whole = [forward_flops_per_token(g) for g in space]
        assert np.log10(max(flops) / min(flops)) > 0.8 * np.log10(max(whole) / min(whole))

    def test_it_is_stratified_rather_than_random(self):
        """The cheap end of this space is dense, so a uniform draw would over-represent it
        and under-sample exactly where the budget is furthest from optimal."""
        space = enumerate_with_ceiling().genotypes
        chosen = select(list(space), size=40)
        flops = sorted(m.flops_per_token for m in chosen)

        # Roughly even in log compute: no half of the sample crowded into one decade.
        gaps = np.diff(np.log10(flops))
        assert gaps.max() < 8 * np.median(gaps[gaps > 0])

    def test_each_member_records_which_stratum_it_came_from(self, measured):
        chosen = select(measured, size=20)
        assert sorted(m.stratum for m in chosen) == list(range(1, len(chosen) + 1))

    def test_the_subset_is_drawn_only_from_what_was_measured(self, measured):
        """A rank correlation needs both rankings over the same architectures."""
        available = {g.as_tuple() for g in measured}
        for member in select(measured, size=20):
            assert member.genotype.as_tuple() in available

    def test_asking_for_more_than_was_measured_is_refused(self):
        with pytest.raises(ValueError, match="cannot supply a subset"):
            select(list(enumerate_with_ceiling().genotypes)[:10], size=50)


class TestPlan:
    def test_the_long_budget_is_four_times_the_short_one(self, measured):
        detail = plan(measured, 100_000_000, size=20)
        assert detail["long_budget"] == 100_000_000 * BUDGET_MULTIPLE
        assert detail["multiple"] == BUDGET_MULTIPLE

    def test_the_multiple_stays_under_the_repeat_data_ceiling(self, measured):
        """WikiText-103 holds 117.9M tokens, and repeated data stops behaving like fresh
        data around four passes. Going further would confound the budget question with a
        data question."""
        detail = plan(measured, 100_000_000, size=20)
        assert detail["epochs_at_long_budget"] <= 4.0
        assert "changing corpus" in detail["why_four"]

    def test_both_outcomes_are_declared_in_advance(self, measured):
        """Deciding what a result would mean before seeing it is what keeps the reading
        from being chosen to suit the number."""
        outcomes = plan(measured, 100_000_000, size=20)["what_each_outcome_means"]
        assert "sufficient" in outcomes["high_rank_correlation"]
        assert "not stable" in outcomes["low_rank_correlation"]


class TestComparison:
    def _budgets(self, n=30, reorder=False, seed=0):
        rng = np.random.default_rng(seed)
        keys = [(i, 32, 2, 2, 1) for i in range(n)]
        short = {k: 100.0 + i for i, k in enumerate(keys)}
        if reorder:
            values = rng.permutation([short[k] * 0.8 for k in keys])
        else:
            values = np.array([short[k] * 0.8 for k in keys])
        return short, dict(zip(keys, values, strict=True))

    def test_a_stable_ranking_scores_near_one(self):
        short, long = self._budgets()
        assert compare(short, long)["kendall_tau"] == pytest.approx(1.0)

    def test_a_reordered_ranking_scores_low(self):
        short, long = self._budgets(reorder=True)
        assert abs(compare(short, long)["kendall_tau"]) < 0.5

    def test_the_overlap_count_travels_with_the_correlation(self):
        """A correlation over eight architectures and one over fifty are different claims,
        and the count is what tells them apart."""
        short, long = self._budgets(n=30)
        assert compare(short, long)["architectures_at_both_budgets"] == 30

    def test_too_little_overlap_is_refused(self):
        short, long = self._budgets(n=5)
        with pytest.raises(ValueError, match="line through noise"):
            compare(short, long)

    def test_the_spread_of_improvement_is_reported_beside_the_correlation(self):
        """A uniform improvement cannot change a ranking, so a high correlation would then
        be a property of the design rather than evidence about the budget."""
        short, long = self._budgets()
        result = compare(short, long)

        spread = result["relative_improvement"]
        assert spread["max"] == pytest.approx(spread["min"], abs=1e-6), (
            "this fixture improves everything by the same fraction"
        )
        assert "uniform improvement cannot change a ranking" in result["interpretation"]

    def test_only_architectures_measured_at_both_budgets_are_compared(self):
        short, long = self._budgets(n=30)
        long.pop(next(iter(long)))
        assert compare(short, long)["architectures_at_both_budgets"] == 29


def test_the_long_budget_rows_cannot_be_mixed_into_the_short_table():
    """require_comparable checks the token budget, so the two sets are separate tables by
    construction. Mixing them would put unequal training back into the one artefact this
    project exists to keep free of it."""
    from nas4textgen.search import IncomparableRows, require_comparable

    base = {
        "precision": "float16", "data_vintage": "2026-08-13",
        "tokeniser_fingerprint": "855ec41a9d90094e", "token_budget": 100_000_000,
    }
    with pytest.raises(IncomparableRows, match="defect this project exists to correct"):
        require_comparable([base, {**base, "token_budget": 400_000_000}])
