"""The cost-matched comparison, and the control that decides how to read it."""

from __future__ import annotations

import pytest

from nas4textgen.matched import (
    COST_KEYS,
    FRONT_BANDS,
    SEARCH_CORPUS,
    NotEnoughRuns,
    analyse,
    compare,
    gather,
)


def manifest(genes, corpus, perplexity, seed=0):
    return {"corpus": corpus, "token_budget": 100_000_000,
            "genotype": {"values": list(genes)},
            "config": {"seed": seed},
            "objectives": {"perplexity": perplexity}}


class TestWhatItBands:
    def test_too_few_architectures_refuses_rather_than_banding(self):
        """Five bands over a handful of points puts one architecture in each extreme, and
        a mean of one is not a comparison."""
        rows = gather([manifest((2, 32, 4, 4, 1), "babylm", 10.0)], "babylm")
        with pytest.raises(NotEnoughRuns):
            compare(rows, "babylm", None, "parameters")

    def test_both_definitions_of_cost_are_reported(self):
        """Parameter count and cache size rank the space differently, so matching on one
        is not matching on the other and a conclusion can hold under just one."""
        assert set(COST_KEYS) == {"parameters", "kv_cache_bytes"}

    def test_seeds_of_one_architecture_are_averaged_not_counted_twice(self):
        ms = [manifest((2, 32, 4, 4, 1), "babylm", p, seed=s)
              for s, p in enumerate((10.0, 12.0, 14.0))]
        rows = gather(ms, "babylm")
        assert len(rows) == 1
        assert rows[0]["perplexity"] == pytest.approx(12.0)

    def test_a_seed_filter_selects_one_draw(self):
        ms = [manifest((2, 32, 4, 4, 1), "babylm", p, seed=s)
              for s, p in enumerate((10.0, 12.0, 14.0))]
        assert gather(ms, "babylm", seed=2)[0]["perplexity"] == pytest.approx(14.0)


class TestTheControl:
    def test_the_search_corpus_is_also_run_on_the_front_alone(self):
        """The row that decides the reading. Off the search corpus only the front was
        retrained, and a front is selected for the very trade-off under test; without the
        same restriction applied to the search corpus the drop reads as a corpus effect
        when it is a selection effect.
        """
        ms = []
        for h in range(1, 31):
            genes = (h if h <= 16 else 16, 32, (h % 12) + 1, 4, 1)
            ms.append(manifest(genes, SEARCH_CORPUS, 100.0 - h))
            if h <= 20:
                ms.append(manifest(genes, "babylm", 90.0 - h))
        report = analyse(ms, bands=2)
        corpora = {c.corpus for c in report.comparisons}
        assert f"{SEARCH_CORPUS} (front only)" in corpora

    def test_the_control_runs_over_what_the_other_corpora_ran(self):
        """It is only a control if it is the same architectures."""
        ms = []
        for h in range(1, 31):
            genes = (h if h <= 16 else 16, 32, (h % 12) + 1, 4, 1)
            ms.append(manifest(genes, SEARCH_CORPUS, 100.0 - h))
            if h <= 20:
                ms.append(manifest(genes, "babylm", 90.0 - h))
        report = analyse(ms, bands=2)
        control = next(c for c in report.comparisons
                       if c.corpus.endswith("(front only)")
                       and c.cost == "parameters")
        other = next(c for c in report.comparisons
                     if c.corpus == "babylm" and c.seed is None
                     and c.cost == "parameters")
        assert control.architectures == other.architectures


class TestWhatItReports:
    def test_bands_won_counts_only_where_width_was_ahead(self):
        from nas4textgen.matched import BandedComparison

        c = BandedComparison(corpus="x", seed=None, cost="parameters", bands=3,
                             architectures=90,
                             deep_means=(10.0, 10.0, 10.0),
                             wide_means=(9.0, 11.0, 9.5))
        assert c.bands_won_by_width == 2
        assert c.unanimous is False

    def test_unanimity_requires_every_band(self):
        from nas4textgen.matched import BandedComparison

        c = BandedComparison(corpus="x", seed=None, cost="parameters", bands=2,
                             architectures=60,
                             deep_means=(10.0, 10.0), wide_means=(9.0, 9.0))
        assert c.unanimous is True

    def test_the_band_count_is_declared(self):
        assert FRONT_BANDS == 5
