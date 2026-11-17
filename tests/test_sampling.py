"""Which sampling design to spend a training budget on.

The comparison exists to be able to say the campaign's own choice was wrong, so the floor
has to be in it. Uniform random is that floor: without it, maximin beats Latin hypercube
and the campaign looks like it picked the better of the two designs it tried.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.sampling import BUDGETS, SEEDS, compare_designs, evaluate_design


def _space():
    from nas4textgen.space.enumerate import enumerate_with_ceiling

    return enumerate_with_ceiling().genotypes[:120]


def _objectives(space):
    rng = np.random.default_rng(0)
    return np.column_stack([
        np.array([g.n * g.e for g in space], dtype=float),
        np.array([g.kv * g.n for g in space], dtype=float),
        rng.uniform(50, 400, size=len(space)),
    ])


class TestTheFloorIsInTheComparison:
    def test_uniform_random_is_one_of_the_designs(self):
        space = _space()
        comparison = compare_designs(space, _objectives(space),
                                     budgets=(20,), seeds=(0, 1))
        assert "uniform_random" in {r.design for r in comparison.results}, (
            "without the floor, the campaign grades its own choice against one alternative"
        )

    def test_enough_seeds_for_a_moderate_difference_to_be_detectable(self):
        """Three designs need eight blocks before a moderate agreement reaches
        significance. Five was inherited from the sweep and is too few."""
        assert len(SEEDS) >= 8


class TestTheTwoLossesAnswerDifferentQuestions:
    def test_a_subset_containing_the_whole_front_has_perfect_recall_and_no_gap(self):
        space = _space()
        objectives = _objectives(space)
        from nas4textgen.space.enumerate import non_dominated

        front = set(non_dominated(objectives).tolist())
        gap, recall, _ = evaluate_design(
            np.array(sorted(front)), objectives, front, space
        )
        assert recall == pytest.approx(1.0)
        assert gap == pytest.approx(0.0, abs=1e-12)

    def test_a_subset_can_be_close_in_objective_space_and_contain_none_of_the_front(self):
        """Which is why both are reported: a good trade-off curve and the actual
        architectures are different goals."""
        space = _space()
        objectives = _objectives(space)
        from nas4textgen.space.enumerate import non_dominated

        front = set(non_dominated(objectives).tolist())
        off_front = np.array([i for i in range(len(space)) if i not in front])
        gap, recall, _ = evaluate_design(off_front, objectives, front, space)
        assert recall == pytest.approx(0.0)
        assert gap > 0.0

    def test_fill_distance_is_reported_so_a_space_filling_win_is_visible(self):
        """Maximin minimises fill distance by construction and loses on front recovery.
        Reporting only the loss it wins would hide the finding; reporting only the loss it
        loses would hide the mechanism."""
        space = _space()
        comparison = compare_designs(space, _objectives(space),
                                     budgets=(30,), seeds=(0, 1, 2))
        fill = comparison.medians("fill_distance")[30]
        assert fill["maximin"] <= fill["uniform_random"], (
            "maximin should win the property it optimises"
        )


class TestBudgetsMatchTheCampaign:
    def test_the_budgets_are_the_sweep_wave_sizes(self):
        """So the answer is to the question the campaign faced, not a rounder one."""
        assert BUDGETS == (150, 300, 600)
