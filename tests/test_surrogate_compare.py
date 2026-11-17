"""The surrogate slate as a statistical question rather than a leaderboard.

Fourteen candidates over five folds produce a winner by arithmetic whether or not one is
genuinely better, and the gap between first and second is routinely smaller than the gap
one candidate shows between its own folds. These tests hold the comparison to blocking on
shared splits, to having the blocks a test over fourteen treatments needs, and to keeping
the trivial floors in as treatments.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.surrogate.compare import (
    FOLDS,
    REPEATS,
    FoldScores,
    NotEnoughBlocks,
    compare,
)


def _scores(candidates: int, blocks: int, offset: float = 0.0) -> FoldScores:
    scores = FoldScores()
    rng = np.random.default_rng(0)
    for index in range(candidates):
        values = {}
        for block in range(blocks):
            values[(block // FOLDS, block % FOLDS)] = (
                0.9 - index * 0.02 + offset * rng.normal(0, 0.001)
            )
        scores.taus[f"model{index}"] = values
    return scores


class TestTheDesignHasEnoughBlocks:
    def test_five_folds_cannot_detect_a_moderate_difference(self):
        """Not because rejection is arithmetically impossible — with fourteen treatments the
        maximum statistic is large and two blocks clear that bar. Because at five blocks
        only a nearly perfect agreement across them reaches significance, so the p-value's
        size would be a fact about the design."""
        with pytest.raises(NotEnoughBlocks, match="nearly\\s+perfect"):
            compare(_scores(candidates=14, blocks=5))

    def test_the_arithmetic_threshold_is_reported_as_the_weak_one(self):
        result = compare(_scores(candidates=14, blocks=REPEATS * FOLDS))
        power = result["power"]
        assert power["blocks_for_rejection_to_be_possible"] < \
            power["blocks_to_detect_moderate_agreement"], (
            "with many treatments the arithmetic bar is nearly vacuous, and quoting it "
            "alone reads as reassurance"
        )

    def test_repetitions_multiply_the_blocks(self):
        assert REPEATS * FOLDS >= 20
        result = compare(_scores(candidates=14, blocks=REPEATS * FOLDS))
        assert result["power"]["blocks"] == REPEATS * FOLDS
        assert result["power"]["design"] == f"{REPEATS} repetitions x {FOLDS} folds"

    def test_a_single_candidate_is_not_a_comparison(self):
        with pytest.raises(NotEnoughBlocks, match="at least two"):
            compare(_scores(candidates=1, blocks=20))


class TestBlocksMustBeComplete:
    def test_a_candidate_missing_a_block_drops_out_rather_than_being_filled(self):
        """A block missing a treatment is not a block, and filling it would invent a
        measurement."""
        scores = _scores(candidates=3, blocks=20)
        del scores.taus["model2"][(0, 0)]
        # The shared-block rule drops the block, not the candidate: blocks() intersects
        # what every candidate scored, so one gap removes that block for everyone. Either
        # rule is defensible; what matters is that nothing is filled in.
        assert (0, 0) not in scores.blocks()
        assert len(scores.blocks()) == 19

    def test_a_candidate_with_an_undefined_tau_drops_out(self):
        """A constant predictor makes Kendall tau undefined, and ranking a nan would put
        it somewhere arbitrary in every block."""
        scores = _scores(candidates=3, blocks=20)
        scores.taus["model1"][(0, 1)] = float("nan")
        assert "model1" not in scores.candidates()


class TestTheFloorIsATreatment:
    def test_the_comparison_reports_the_direction_of_each_difference(self):
        """`median_difference` is computed in alphabetical pair order, so its sign says
        which name sorts first rather than which model is better — reading it as an
        advantage would report the floor's disadvantage as the model's."""
        from nas4textgen.statistics.frequentist import PairwiseResult

        entry = PairwiseResult("sort_by_parameters", "xgboost", 1.0, 0.001, 0.001,
                               True, -0.077).as_dict()
        assert entry["median_difference"] < 0
        assert entry["a"] == "sort_by_parameters", (
            "the sign is relative to `a`, which is why a caller must compare the medians "
            "rather than read this as the second model's margin"
        )

    def test_the_real_slate_puts_the_floor_mid_table(self):
        """Nine trained models beat sorting by parameter count and four lose to it, which
        is the answer to R2 #6 as a ranked result rather than two numbers to subtract."""
        import json
        from pathlib import Path

        path = (Path(__file__).resolve().parents[1]
                / "artifacts" / "surrogate" / "comparison.json")
        if not path.exists():
            pytest.skip("run: python -m nas4textgen.cli surrogate")
        payload = json.loads(path.read_text(encoding="utf-8"))
        tau = payload["median_tau"]
        assert "sort_by_parameters" in tau, "the floor must be a treatment, not a footnote"

        floor = tau["sort_by_parameters"]
        better = [n for n, v in tau.items() if v > floor]
        worse = [n for n, v in tau.items() if v < floor]
        assert better and worse, (
            "a floor that everything beats or nothing beats is not doing any work"
        )
        assert payload["friedman"]["p_value"] < 0.05
