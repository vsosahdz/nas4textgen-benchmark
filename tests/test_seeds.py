"""Where to spend seeds, which is not the same question as how many.

Uniform seeding is the obvious design and it pays three times over for pairs separated by
fifty standard deviations. Seed noise only changes a conclusion where two architectures are
close enough to reorder, and that is on and near the front.
"""

from __future__ import annotations

import numpy as np

from nas4textgen.azure.seeds import (
    BASE_SEEDS,
    FRONT_MARGIN_SIGMAS,
    FRONT_SEEDS,
    near_front,
    plan,
)

SIGMA = 0.651272  # measured: five seeds of one architecture


class TestNearFront:
    def test_the_front_itself_is_always_included(self):
        # cost increasing, quality improving: every point is non-dominated.
        objectives = np.array([[1.0, 10.0], [2.0, 8.0], [3.0, 6.0], [4.0, 4.0]])
        assert set(near_front(objectives, SIGMA).tolist()) == {0, 1, 2, 3}

    def test_a_point_within_noise_of_the_front_is_included(self):
        """It can be promoted by a single noisy measurement, so it is part of the question
        even though its point estimate is not on the front."""
        objectives = np.array([
            [1.0, 10.0],
            [2.0, 8.0],
            [2.0, 8.0 + 0.5 * SIGMA],   # dominated, but only just
        ])
        assert 2 in near_front(objectives, SIGMA).tolist()

    def test_a_point_far_behind_is_excluded(self):
        """Replicating it would spend compute on a conclusion that will not change."""
        objectives = np.array([
            [1.0, 10.0],
            [2.0, 8.0],
            [2.0, 8.0 + 50 * SIGMA],
        ])
        assert 2 not in near_front(objectives, SIGMA).tolist()

    def test_the_margin_governs_what_counts_as_near(self):
        objectives = np.array([[1.0, 10.0], [2.0, 8.0], [2.0, 8.0 + 3 * SIGMA]])
        tight = near_front(objectives, SIGMA, margin_sigmas=1.0).tolist()
        loose = near_front(objectives, SIGMA, margin_sigmas=5.0).tolist()
        assert 2 not in tight
        assert 2 in loose

    def test_only_the_quality_column_moves(self):
        """Cost is computed in closed form from the genotype and carries no seed noise, so
        the margin is subtracted from quality alone.

        Two attempts at this test asserted something false first. A point behind on cost
        *is* promoted when its quality improves enough — that is what a Pareto front is, not
        a bug. The property that actually holds is narrower: a point behind on **both**
        objectives by more than the margin stays out, because improving quality alone cannot
        rescue it.
        """
        dominated_on_both = np.array([
            [1.0, 1.0],
            [2.0, 10.0],   # worse on cost and far worse on quality
        ])
        assert near_front(dominated_on_both, SIGMA).tolist() == [0]

        # And the same point becomes reachable once the margin exceeds the quality gap,
        # which shows the margin is what moved.
        assert 1 in near_front(dominated_on_both, SIGMA, margin_sigmas=20.0).tolist()

    def test_zero_noise_reduces_to_the_exact_front(self):
        from nas4textgen.space.enumerate import non_dominated

        objectives = np.array([[1.0, 10.0], [2.0, 8.0], [2.0, 12.0]])
        assert set(near_front(objectives, 0.0).tolist()) == set(
            non_dominated(objectives).tolist()
        )


class TestPlan:
    def _spread(self, n: int) -> np.ndarray:
        """A convex trade-off with a cloud of dominated points behind it.

        The front alone would make every point near-front and the test vacuous, which is
        exactly what the pilot's twenty-three spread-over-compute architectures look like —
        nineteen of them are on the front. A real space has interior points.
        """
        rng = np.random.default_rng(0)
        cost = np.linspace(1, 100, n)
        quality = 500.0 / np.sqrt(cost)
        # Push most points well behind the frontier; leave a few on it.
        penalty = rng.uniform(20, 200, n)
        penalty[:: max(n // 8, 1)] = 0.0
        return np.column_stack([cost, quality + penalty])

    def test_replicating_only_near_the_front_costs_less_than_uniform(self):
        p = plan(self._spread(200), SIGMA)
        assert p.total_runs < p.uniform_runs
        assert 0 < p.as_dict()["saved_fraction"] < 1

    def test_every_architecture_still_gets_a_measurement(self):
        p = plan(self._spread(50), SIGMA)
        assert p.base_seeds == BASE_SEEDS >= 1
        assert p.total_runs >= p.architectures

    def test_front_depth_is_deep_enough_for_the_rope_helper(self):
        """The replicates have to support a standard deviation, not just exist."""
        from nas4textgen.statistics.rope import MIN_REPLICATES

        assert FRONT_SEEDS >= MIN_REPLICATES

    def test_the_record_states_both_the_reason_and_the_caveat(self):
        """The pilot's architectures were chosen to spread over compute and are unusually
        well separated, so a front over 1,099 points will be denser. Saying so is the
        difference between a projection and a claim."""
        payload = plan(self._spread(100), SIGMA).as_dict()
        assert "nine and a half sigma" in payload["why"]
        assert "denser" in payload["caveat"]
        assert payload["margin_sigmas"] == FRONT_MARGIN_SIGMAS

    def test_a_dense_front_costs_more_than_a_sparse_one(self):
        """Which is the honest direction: the more architectures compete for the front, the
        more of them a second measurement could reorder."""
        sparse = plan(self._spread(100), sigma=0.01)
        dense = plan(self._spread(100), sigma=50.0)
        assert dense.near_front > sparse.near_front
        assert dense.total_runs > sparse.total_runs


def test_seed_noise_is_smaller_than_typical_architecture_separation():
    """The premise the whole design rests on. If neighbouring architectures differed by
    less than the noise, one seed would be indefensible and five would be the floor."""
    # Cost-adjacent perplexities measured in the pilot.
    measured = [509.17, 493.47, 484.37, 474.41, 474.62, 469.43, 475.20, 281.77,
                260.56, 261.14, 209.70, 203.04, 189.77, 186.33, 149.60, 144.07,
                110.75, 106.16, 105.09, 80.16, 83.00, 68.53, 54.24]
    gaps = np.abs(np.diff(measured))
    assert np.median(gaps) / SIGMA > 5, "the median gap must dwarf the noise"
    assert (gaps < 2 * SIGMA).mean() < 0.2, "few enough near-ties for one seed to be sane"
