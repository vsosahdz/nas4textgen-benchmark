"""Statistics, checked against constructed data where the right answer is known.

Every case here is built so the correct conclusion is decided in advance: treatments that
are identical, treatments separated by a known margin, a slate with a planted winner. That
is the only way to test an inference procedure — running it on real results tells you what
it says, not whether it is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.statistics import (
    AggregatedPValue,
    Comparison,
    Declaration,
    InsufficientReplicates,
    MalformedComparison,
    friedman,
    from_replicates,
    from_seed_matrix,
    holm,
    plackett_luce,
    posthoc,
    refuse_averaged_pvalues,
    signed_rank,
)


def declaration(**overrides) -> Declaration:
    base = dict(
        question="Do these architectures differ on held-out perplexity?",
        treatments=("A", "B", "C"),
        blocks=tuple(f"corpus{i}" for i in range(8)),
        metric="perplexity",
        higher_is_better=False,
        rope=0.5,
        rope_source="seed replicates of the reference architecture",
    )
    base.update(overrides)
    return Declaration(**base)


# ── the declaration is the contract ────────────────────────────────────────────

class TestDeclaration:
    def test_a_rope_without_a_source_is_refused(self):
        """A ROPE with no measurement behind it is a threshold someone chose, and it
        decides which results get called equivalent."""
        with pytest.raises(MalformedComparison, match="threshold someone chose"):
            declaration(rope=0.5, rope_source="")

    def test_one_treatment_is_not_a_comparison(self):
        with pytest.raises(MalformedComparison):
            declaration(treatments=("A",))

    def test_data_must_match_the_declared_shape(self):
        """Declaring first is what makes the error rate mean anything; the data has to
        match what was declared, not the other way round."""
        with pytest.raises(MalformedComparison, match="8 blocks and 3 treatments"):
            Comparison(declaration(), np.zeros((5, 3)))

    def test_a_missing_measurement_is_refused_rather_than_dropped(self):
        """An unbalanced design filled with NaN ranks treatments over different block
        sets, and nothing in the output says so."""
        values = np.ones((8, 3))
        values[3, 1] = np.nan
        with pytest.raises(MalformedComparison, match="corpus3"):
            Comparison(declaration(), values)

    def test_ranking_is_within_block_so_scales_cannot_dominate(self):
        """Perplexity on WikiText and on TinyStories are not the same quantity; averaging
        them directly weights whichever corpus has larger numbers."""
        values = np.array([[10.0, 20.0, 30.0], [1000.0, 2000.0, 3000.0]])
        comparison = Comparison(declaration(blocks=("small", "large")), values)
        assert comparison.mean_ranks() == {"A": 1.0, "B": 2.0, "C": 3.0}

    def test_direction_is_declared_not_inferred(self):
        values = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        blocks = ("x", "y", "z")
        lower = Comparison(declaration(treatments=("A", "B"), blocks=blocks,
                                       higher_is_better=False), values)
        higher = Comparison(declaration(treatments=("A", "B"), blocks=blocks,
                                        higher_is_better=True), values)
        assert lower.mean_ranks()["A"] == 1.0
        assert higher.mean_ranks()["A"] == 2.0


# ── multiplicity ───────────────────────────────────────────────────────────────

class TestMultiplicity:
    def test_averaging_p_values_raises(self):
        """The spec forbids it; this makes the forbidden path fail rather than return a
        plausible number."""
        with pytest.raises(AggregatedPValue, match="no null distribution"):
            refuse_averaged_pvalues([0.01, 0.04, 0.9])

    def test_holm_is_uniformly_at_least_as_powerful_as_bonferroni(self):
        """Never worse anywhere, and strictly better after the first.

        The smallest p-value gets the same multiplier under both — that is where the two
        procedures agree by construction. Holm's gain is that each subsequent comparison
        is divided among the hypotheses still standing rather than all of them, so
        preferring Bonferroni discards real differences for nothing.
        """
        raw = [0.001, 0.012, 0.03, 0.2]
        adjusted, _ = holm(raw)
        bonferroni = [min(1.0, p * len(raw)) for p in raw]

        assert all(a <= b + 1e-12 for a, b in zip(adjusted, bonferroni, strict=True))
        assert adjusted[0] == pytest.approx(bonferroni[0]), "they agree on the smallest"
        assert adjusted[1] < bonferroni[1], "and Holm is strictly better from the second"
        assert adjusted[2] < bonferroni[2]

    def test_holm_adjusted_values_are_monotone(self):
        """Without the running maximum the sorted output can invert, and a comparison
        ends up with a smaller adjusted p-value than one that was more significant."""
        raw = [0.04, 0.005, 0.045, 0.049]
        adjusted, _ = holm(raw)
        by_raw = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
        assert by_raw == sorted(by_raw)

    def test_holm_controls_the_family_wise_error_rate_under_the_null(self):
        """Twenty pairwise tests on pure noise should almost never yield a rejection."""
        rng = np.random.default_rng(0)
        false_positives = 0
        trials = 400
        for _ in range(trials):
            raw = list(rng.uniform(size=20))
            _, rejected = holm(raw, alpha=0.05)
            false_positives += any(rejected)
        assert false_positives / trials < 0.08, "family-wise rate should sit near 0.05"


# ── the omnibus test gates the pairwise ones ───────────────────────────────────

class TestFriedman:
    def test_identical_treatments_are_not_declared_different(self):
        rng = np.random.default_rng(1)
        base = rng.normal(30, 3, size=(12, 1))
        values = np.repeat(base, 3, axis=1) + rng.normal(0, 0.01, size=(12, 3))
        comparison = Comparison(declaration(blocks=tuple(f"b{i}" for i in range(12))), values)
        assert not friedman(comparison).differs

    def test_a_planted_ordering_is_recovered(self):
        rng = np.random.default_rng(2)
        blocks = tuple(f"b{i}" for i in range(15))
        base = rng.normal(40, 5, size=(15, 1))
        # A is best on perplexity, then B, then C, on every block.
        values = np.hstack([base, base + 2.0, base + 4.0]) + rng.normal(0, 0.2, size=(15, 3))
        comparison = Comparison(declaration(blocks=blocks), values)

        result = friedman(comparison)
        assert result.differs
        ranks = result.mean_ranks
        assert ranks["A"] < ranks["B"] < ranks["C"]
        assert result.concordance > 0.9, "the blocks agree strongly about the order"

    def test_pairwise_comparison_is_refused_after_a_negative_omnibus(self):
        """Searching pairs after a negative omnibus has an error rate nobody can state,
        and it is where a result that will not replicate comes from."""
        rng = np.random.default_rng(3)
        blocks = tuple(f"b{i}" for i in range(12))
        values = rng.normal(30, 3, size=(12, 3))
        comparison = Comparison(declaration(blocks=blocks), values)

        omnibus = friedman(comparison)
        if omnibus.differs:
            pytest.skip("this draw happened to differ; the guard is tested by construction")
        with pytest.raises(ValueError, match="no pairwise comparison is licensed"):
            posthoc(comparison, omnibus)

    def test_posthoc_finds_the_planted_pair(self):
        rng = np.random.default_rng(4)
        blocks = tuple(f"b{i}" for i in range(20))
        base = rng.normal(40, 5, size=(20, 1))
        values = np.hstack([base, base + 0.05, base + 6.0]) + rng.normal(0, 0.1, size=(20, 3))
        comparison = Comparison(declaration(blocks=blocks), values)

        results = {(r.a, r.b): r for r in posthoc(comparison, friedman(comparison))}
        assert results[("A", "C")].rejected, "a six-point gap should survive correction"
        assert results[("B", "C")].rejected
        assert results[("A", "C")].p_adjusted >= results[("A", "C")].p_raw


# ── the Bayesian side ──────────────────────────────────────────────────────────

class TestBayesian:
    def test_no_rope_means_no_equivalence_claim(self):
        values = np.random.default_rng(5).normal(30, 2, size=(8, 3))
        comparison = Comparison(declaration(rope=None, rope_source=""), values)
        with pytest.raises(MalformedComparison, match="no meaning here"):
            signed_rank(comparison, "A", "B")

    def test_treatments_inside_the_rope_are_called_equivalent(self):
        """The statement a p-value cannot make: not 'we failed to find a difference' but
        'the difference is smaller than the noise floor'."""
        rng = np.random.default_rng(6)
        blocks = tuple(f"b{i}" for i in range(30))
        base = rng.normal(40, 5, size=(30, 1))
        values = np.hstack([base, base + 0.02, base + 9.0])
        comparison = Comparison(declaration(blocks=blocks, rope=0.5,
                                            rope_source="seed replicates"), values)

        verdict = signed_rank(comparison, "A", "B")
        assert verdict.probability_equivalent > 0.95
        assert "practically equivalent" in verdict.verdict

    def test_a_real_difference_is_attributed_to_the_right_treatment(self):
        rng = np.random.default_rng(7)
        blocks = tuple(f"b{i}" for i in range(30))
        base = rng.normal(40, 5, size=(30, 1))
        values = np.hstack([base, base + 0.02, base + 9.0])
        comparison = Comparison(declaration(blocks=blocks, rope=0.5,
                                            rope_source="seed replicates"), values)

        verdict = signed_rank(comparison, "A", "C")
        # Lower perplexity is better and A is 9 lower, so A wins.
        assert verdict.probability_a > 0.95
        assert verdict.verdict.startswith("A is practically better")

    def test_the_three_probabilities_partition_the_posterior(self):
        rng = np.random.default_rng(8)
        blocks = tuple(f"b{i}" for i in range(20))
        values = rng.normal(30, 3, size=(20, 3))
        comparison = Comparison(declaration(blocks=blocks, rope=0.4,
                                            rope_source="seed replicates"), values)
        v = signed_rank(comparison, "A", "B")
        assert v.probability_a + v.probability_equivalent + v.probability_b == pytest.approx(1.0)

    def test_a_verdict_is_reproducible_from_its_recorded_seed(self):
        rng = np.random.default_rng(9)
        blocks = tuple(f"b{i}" for i in range(20))
        values = rng.normal(30, 3, size=(20, 3))
        comparison = Comparison(declaration(blocks=blocks, rope=0.4,
                                            rope_source="seed replicates"), values)
        first = signed_rank(comparison, "A", "B", seed=123)
        again = signed_rank(comparison, "A", "B", seed=123)
        assert first.as_dict() == again.as_dict()

    def test_undecided_is_a_reported_outcome_not_a_silent_one(self):
        """Two blocks cannot support a 0.95 claim in any direction, and saying so is the
        honest output."""
        values = np.array([[30.0, 31.0], [31.0, 30.0]])
        comparison = Comparison(
            declaration(treatments=("A", "B"), blocks=("x", "y"), rope=0.1,
                        rope_source="seed replicates"),
            values,
        )
        assert signed_rank(comparison, "A", "B").verdict.startswith("undecided")


class TestSlateRanking:
    def test_the_planted_winner_is_most_probably_best(self):
        rng = np.random.default_rng(10)
        names = ("random", "nsga2", "smsemoa", "moead")
        blocks = tuple(f"seed{i}" for i in range(25))
        base = rng.normal(0.5, 0.05, size=(25, 1))
        # smsemoa best on a higher-is-better metric, random clearly worst.
        values = np.hstack([base - 0.15, base, base + 0.12, base + 0.01])
        values += rng.normal(0, 0.01, size=values.shape)

        comparison = Comparison(
            Declaration(question="Which optimiser reaches the front fastest?",
                        treatments=names, blocks=blocks, metric="hypervolume",
                        higher_is_better=True, rope=0.02,
                        rope_source="seed replicates of NSGA-II"),
            values,
        )
        ranking = plackett_luce(comparison)
        assert ranking.winner == "smsemoa"
        assert ranking.probability_best["smsemoa"] > 0.9
        assert ranking.expected_rank["random"] > ranking.expected_rank["smsemoa"]

    def test_probabilities_of_being_best_sum_to_one(self):
        rng = np.random.default_rng(11)
        blocks = tuple(f"b{i}" for i in range(15))
        values = rng.normal(30, 3, size=(15, 3))
        ranking = plackett_luce(Comparison(declaration(blocks=blocks), values))
        assert sum(ranking.probability_best.values()) == pytest.approx(1.0)

    def test_a_close_slate_reports_a_wide_rank_interval(self):
        """A treatment can be most probably best and still be indistinguishable; the
        interval is what stops the winner being read as settled."""
        rng = np.random.default_rng(12)
        blocks = tuple(f"b{i}" for i in range(10))
        values = rng.normal(30, 3, size=(10, 3))
        ranking = plackett_luce(Comparison(declaration(blocks=blocks), values))
        widths = [hi - lo for lo, hi in ranking.credible_rank_interval.values()]
        assert max(widths) >= 1, "noise should leave the ordering unsettled"


# ── the ROPE is measured ───────────────────────────────────────────────────────

class TestRope:
    def test_too_few_replicates_is_an_error_not_a_guess(self):
        with pytest.raises(InsufficientReplicates, match="rather than a guess"):
            from_replicates([30.1, 30.2, 29.9], metric="perplexity", source="reference")

    def test_identical_replicates_are_refused(self):
        """Zero measured noise means either a bit-exact stage or dependent replicates,
        and both need saying rather than producing a ROPE of zero."""
        with pytest.raises(InsufficientReplicates, match="measures no noise"):
            from_replicates([30.0] * 6, metric="perplexity", source="reference")

    def test_the_width_is_half_the_measured_standard_deviation(self):
        values = [30.0, 31.0, 29.0, 30.5, 29.5, 30.2]
        rope = from_replicates(values, metric="perplexity", source="reference at seed 0-5")
        assert rope.width == pytest.approx(0.5 * np.std(values, ddof=1))
        assert rope.replicates == 6

    def test_the_measurement_travels_with_the_rope(self):
        """The spec requires it reported alongside; a bare number invites re-derivation
        by whoever needs a different answer."""
        rope = from_replicates([30.0, 31.0, 29.0, 30.5, 29.5, 30.2],
                               metric="perplexity", source="reference at seed 0-5")
        payload = rope.as_dict()
        for key in ("rope", "metric", "source", "replicates", "std", "sigma_multiple"):
            assert key in payload
        assert "seed 0-5" in rope.description

    def test_pooling_across_configurations_does_not_rest_on_one_architecture(self):
        """A ROPE from a single architecture claims its noise is everyone's."""
        rng = np.random.default_rng(13)
        # Three architectures at very different perplexity levels, same seed noise.
        levels = np.array([[20.0], [60.0], [300.0]])
        matrix = levels + rng.normal(0, 1.0, size=(3, 6))

        pooled = from_seed_matrix(matrix, metric="perplexity", source="seed study")
        assert pooled.std == pytest.approx(1.0, abs=0.5), (
            "pooling must remove the between-architecture spread, which is 100x the noise"
        )
        naive = float(matrix.std(ddof=1))
        assert naive > 10 * pooled.std, "the naive spread would inflate the ROPE hugely"

    def test_a_single_seed_per_configuration_cannot_estimate_noise(self):
        with pytest.raises(InsufficientReplicates):
            from_seed_matrix(np.ones((6, 1)), metric="perplexity", source="x")
