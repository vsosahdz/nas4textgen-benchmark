"""Whether the transfer analysis measures what it says it measures.

The study asks a narrow question — only the front was retrained, so no second census exists
— and the risk is not that the arithmetic is wrong but that the arithmetic answers a
different question than the prose claims. These tests pin the question.
"""

from __future__ import annotations

import pytest

from nas4textgen.transfer import (
    DESIGN_RULES,
    Measured,
    NotEnoughRuns,
    analyse,
    collect,
    compare,
    evaluate_rules,
    measured_rope,
)


def manifest(genes, corpus, perplexity, seed=0, budget=100_000_000):
    return {
        "corpus": corpus,
        "token_budget": budget,
        "usable_for_benchmark": True,
        "genotype": {"values": list(genes)},
        "config": {"seed": seed},
        "objectives": {"perplexity": perplexity},
    }


class TestCollecting:
    def test_seeds_of_one_architecture_are_gathered_not_counted_separately(self):
        ms = [manifest((2, 32, 4, 4, 1), "tinystories", p, seed=s)
              for s, p in enumerate((10.0, 11.0, 12.0))]
        measured = collect(ms, "tinystories")
        assert len(measured) == 1
        only = next(iter(measured.values()))
        assert only.perplexities == (10.0, 11.0, 12.0)
        assert only.mean == pytest.approx(11.0)

    def test_a_different_budget_is_a_different_measurement(self):
        """The convergence study trained the same corpus at four times the tokens. Mixing
        those rows into the transfer comparison would compare budget with corpus."""
        ms = [manifest((2, 32, 4, 4, 1), "wikitext103", 10.0),
              manifest((2, 32, 4, 4, 1), "wikitext103", 8.0, budget=400_000_000)]
        assert len(collect(ms, "wikitext103")) == 1
        only = next(iter(collect(ms, "wikitext103").values()))
        assert only.perplexities == (10.0,)

    def test_a_run_not_usable_for_the_benchmark_is_left_out(self):
        bad = manifest((2, 32, 4, 4, 1), "babylm", 10.0)
        bad["usable_for_benchmark"] = False
        assert collect([bad], "babylm") == {}


class TestTheRope:
    def test_it_is_half_the_mean_seed_spread(self):
        """Measured from replicates, not chosen. A threshold picked after seeing the
        differences can make any gap look meaningful."""
        measured = [
            Measured((1, 2, 3, 4, 1), 1000, (10.0, 12.0)),   # sd 1.414
            Measured((2, 2, 3, 4, 1), 2000, (20.0, 20.0)),   # sd 0
        ]
        expected = 0.5 * ((2.0 ** 0.5) + 0.0) / 2
        assert measured_rope(measured) == pytest.approx(expected)

    def test_a_single_seed_contributes_no_estimate_rather_than_a_zero(self):
        """Counting an unreplicated architecture as zero noise would shrink the rope toward
        zero and make every difference look real."""
        assert measured_rope([Measured((1, 2, 3, 4, 1), 1000, (10.0,))]) == 0.0


class TestTheRuleTest:
    def test_rules_are_scored_against_front_membership_not_raw_quality(self):
        """The regression test for a real error in this module.

        Rules were first scored against the top quintile by perplexity, which guarantees a
        cost-efficiency rule fails: the architectures `h <= 2` selects are the small ones,
        and small models have worse perplexity by construction. `h <= 2` scored a lift of
        exactly zero on all three corpora, which read as a finding and was an artefact.
        """
        # Two small architectures with poor perplexity but very few parameters, and two
        # large ones with good perplexity. All four are non-dominated.
        measured = [
            Measured((1, 32, 4, 4, 1), 1_000, (30.0,)),
            Measured((2, 32, 4, 4, 1), 2_000, (25.0,)),
            Measured((8, 32, 4, 4, 1), 50_000, (12.0,)),
            Measured((16, 32, 4, 4, 1), 90_000, (11.0,)),
        ]
        retained = [m for m in measured if m.genotype[0] <= 2]
        outcomes = {o.rule: o for o in evaluate_rules(measured, retained)}

        # Scored against who stays on the front, the small-head rule is enriched, not zero.
        assert outcomes["h <= 2"].lift > 1.0

    def test_every_declared_rule_is_reported(self):
        measured = [Measured((2, 32, 4, 4, 1), 1000, (10.0,))]
        outcomes = evaluate_rules(measured, measured)
        assert {o.rule for o in outcomes} == set(DESIGN_RULES)

    def test_an_empty_retained_set_reports_nothing_rather_than_dividing_by_zero(self):
        assert evaluate_rules([Measured((2, 32, 4, 4, 1), 1000, (10.0,))], []) == []


class TestComparing:
    def _corpus(self, corpus, perplexities):
        return [manifest((h, 32, 4, 4, 1), corpus, p, seed=s)
                for h, p in perplexities for s in range(2)]

    def test_too_few_shared_architectures_refuses_rather_than_reporting(self):
        """Three points can produce a rank correlation of 1.0 by chance. Reporting it as
        evidence of transfer is the failure this refusal prevents."""
        search = collect(self._corpus("wikitext103", [(1, 10.0)]), "wikitext103")
        other = collect(self._corpus("babylm", [(1, 10.0)]), "babylm")
        with pytest.raises(NotEnoughRuns):
            compare(search, other, "babylm")

    def test_a_preserved_ordering_gives_a_correlation_of_one(self):
        pairs = [(1, 20.0), (2, 18.0), (4, 15.0), (8, 12.0), (16, 10.0)]
        search = collect(self._corpus("wikitext103", pairs), "wikitext103")
        other = collect(self._corpus("babylm", pairs), "babylm")
        result = compare(search, other, "babylm")
        assert result.kendall_tau == pytest.approx(1.0)
        assert result.architectures == 5

    def test_a_reversed_ordering_gives_minus_one(self):
        forward = [(1, 20.0), (2, 18.0), (4, 15.0), (8, 12.0), (16, 10.0)]
        reverse = [(1, 10.0), (2, 12.0), (4, 15.0), (8, 18.0), (16, 20.0)]
        search = collect(self._corpus("wikitext103", forward), "wikitext103")
        other = collect(self._corpus("babylm", reverse), "babylm")
        assert compare(search, other, "babylm").kendall_tau == pytest.approx(-1.0)


class TestTheWholeStudy:
    def test_the_search_corpus_is_not_compared_against_itself(self):
        pairs = [(h, 20.0 - h) for h in (1, 2, 4, 8, 16)]
        ms = ([manifest((h, 32, 4, 4, 1), "wikitext103", p) for h, p in pairs]
              + [manifest((h, 32, 4, 4, 1), "babylm", p, seed=s)
                 for h, p in pairs for s in range(2)])
        report = analyse(ms)
        assert [c.corpus for c in report.corpora] == ["babylm"]

    def test_no_runs_on_the_search_corpus_refuses(self):
        ms = [manifest((h, 32, 4, 4, 1), "babylm", 10.0 + h) for h in (1, 2, 4, 8)]
        with pytest.raises(NotEnoughRuns):
            analyse(ms)
