"""Whether the battery analysis can be trusted to say the battery was worth running.

The module exists to answer a reviewer objection, which makes it the module most likely to
be read as confirming what its author wanted. The tests here pin the parts where that would
show: the direction the result is read in, the sample size below which nothing is said, and
the threshold being fixed in advance rather than after the numbers arrive.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from nas4textgen.battery import (  # noqa: E402
    HEADLINE,
    MINIMUM_ARCHITECTURES,
    REDUNDANT_ABOVE,
    NoBatteryRuns,
    agreement,
    analyse,
    collect,
)


def manifest(genes, corpus, perplexity, battery):
    return {
        "corpus": corpus,
        "token_budget": 100_000_000,
        "genotype": {"values": list(genes), "readable": str(genes)},
        "objectives": {"perplexity": perplexity},
        "battery": battery,
    }


def blimp(accuracy):
    return {"blimp": {"macro_accuracy": accuracy, "chance": 0.5}}


def wave(count, accuracy_of):
    """`count` architectures, each with a perplexity and a BLiMP accuracy."""
    return [manifest((h, 32, 4, 4, 1), "tinystories", 20.0 - h * 0.1,
                     blimp(accuracy_of(h)))
            for h in range(count)]


class TestTheSampleSizeGuard:
    def test_a_handful_of_architectures_reports_nothing(self):
        """The canary wave produced three architectures and five metrics reading a flat
        +/-1.000, declared redundant. With three points a Kendall tau is +1, -1 or +/-0.333
        almost whatever the data say, so the threshold measured the sample size."""
        rows = collect(wave(3, lambda h: 0.5 + h * 0.01), "tinystories")
        assert agreement(rows, "blimp", "tinystories") is None

    def test_enough_architectures_reports(self):
        rows = collect(wave(MINIMUM_ARCHITECTURES, lambda h: 0.5 + h * 0.001),
                       "tinystories")
        result = agreement(rows, "blimp", "tinystories")
        assert result is not None
        assert result.architectures == MINIMUM_ARCHITECTURES

    def test_a_metric_that_ran_but_is_thin_is_named_rather_than_dropped(self):
        """Silence from a metric that was measured reads as a metric that was not licensed,
        and those are different facts about the run."""
        report = analyse(wave(3, lambda h: 0.5 + h * 0.01))
        assert ("tinystories", "blimp") in report.underpowered
        assert report.agreements == ()


class TestHowTheResultIsRead:
    def test_a_metric_tracking_perplexity_exactly_is_called_redundant(self):
        """The direction that matters. A battery member that orders the front the way the
        objective already does adds a column, not a fact, and the point of the analysis is
        to say so rather than to count metrics."""
        rows = collect(wave(40, lambda h: 0.9 - h * 0.01), "tinystories")
        result = agreement(rows, "blimp", "tinystories")
        assert abs(result.kendall_tau) == pytest.approx(1.0)
        assert result.redundant is True

    def test_a_metric_disagreeing_with_perplexity_is_the_case_for_the_battery(self):
        import random

        random.seed(0)
        shuffled = random.sample(range(40), 40)
        rows = collect(
            [manifest((h, 32, 4, 4, 1), "tinystories", 20.0 - h * 0.1,
                      blimp(0.5 + shuffled[h] * 0.001)) for h in range(40)],
            "tinystories")
        result = agreement(rows, "blimp", "tinystories")
        assert result.redundant is False

    def test_the_threshold_is_declared_not_derived(self):
        """A threshold chosen after seeing which metrics would clear it is not a threshold.
        It is a constant in the module, and this test is what keeps it one."""
        assert REDUNDANT_ABOVE == 0.9


class TestWhatItRefuses:
    def test_no_battery_anywhere_is_an_error_not_an_empty_report(self):
        plain = [{"corpus": "tinystories", "genotype": {"values": [1, 32, 4, 4, 1]},
                  "objectives": {"perplexity": 20.0}}]
        with pytest.raises(NoBatteryRuns):
            analyse(plain)

    def test_a_metric_that_recorded_an_error_is_not_correlated_against(self):
        """`unavailable` is a string where a number should be. Correlating against it would
        raise, or worse, coerce."""
        rows = collect(
            [manifest((h, 32, 4, 4, 1), "tinystories", 20.0 - h,
                      {"blimp": {"unavailable": "RuntimeError: out of memory"}})
             for h in range(40)],
            "tinystories")
        assert agreement(rows, "blimp", "tinystories") is None

    def test_every_headline_metric_names_the_key_it_reads(self):
        """Several battery entries hold a dozen numbers. Picking one silently would make
        the correlation depend on an undocumented choice."""
        for metric, (key, lower_is_better) in HEADLINE.items():
            assert isinstance(key, str) and key, metric
            assert isinstance(lower_is_better, bool), metric
