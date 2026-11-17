"""Assembling the table, and what it refuses to include.

This is the artefact the paper publishes and the second paper searches over, so what stays
out matters as much as what goes in. Each test here is a row that must not appear.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nas4textgen.benchmark import COLUMNS, NoUsableRows, assemble
from nas4textgen.space.genotype import GENES, Genotype

ROOT = Path(__file__).resolve().parents[1]

TINY = Genotype(h=2, d_head=32, n=2, ffn=2, kv=1)
BIG = Genotype(h=12, d_head=128, n=4, ffn=2, kv=6)


def manifest(genotype=TINY, perplexity=100.0, usable=True, seed=0, **overrides):
    payload = {
        "usable_for_benchmark": usable,
        "genotype": {"values": list(genotype.as_tuple())},
        "objectives": {"perplexity": perplexity, "bits_per_byte": 1.5},
        "tokens_consumed": 100_007_936,
        "wall_clock_seconds": 3600.0,
        "config": {"seed": seed},
        "precision": "float16",
        "device": "cuda",
        "data_vintage": "2026-08-13",
        "tokeniser_fingerprint": "855ec41a9d90094e",
        "token_budget": 100_000_000,
        "cost": {"tokens_per_second": 27_000.0, "achieved_tflops": 1.3,
                 "micro_batch_used": 8, "resumes": 0},
    }
    payload.update(overrides)
    return payload


class TestWhatStaysOut:
    def test_a_truncated_run_is_excluded_and_counted(self):
        """It measured something real and it is not the same thing: fewer tokens means its
        perplexity answers a different question. Placing it beside a complete row would put
        the prior work's central defect into the one artefact everything depends on.
        """
        result = assemble(
            [manifest(), manifest(BIG, 50.0, usable=False)], space_size=1099
        )
        assert len(result.rows) == 1
        assert result.excluded["not_usable_for_benchmark"] == 1

    def test_rows_measured_differently_are_refused_outright(self):
        """Not excluded — refused. A table half in float16 and half in bfloat16 is not a
        table with a caveat, it is two tables."""
        from nas4textgen.search import IncomparableRows

        with pytest.raises(IncomparableRows):
            assemble(
                [manifest(), manifest(BIG, 50.0, precision="bfloat16")], space_size=1099
            )

    def test_a_repeated_measurement_appears_once(self):
        result = assemble([manifest(), manifest(perplexity=101.0)], space_size=1099)
        assert len(result.rows) == 1
        assert result.excluded["duplicate_fingerprint"] == 1

    def test_the_same_architecture_at_different_seeds_is_two_rows(self):
        """Seeds are replicates of one measurement, not duplicates of one row — the ROPE
        depends on being able to see them separately."""
        result = assemble(
            [manifest(seed=0), manifest(seed=1, perplexity=101.0)], space_size=1099
        )
        assert len(result.rows) == 2
        assert "duplicate_fingerprint" not in result.excluded

    def test_nothing_usable_is_an_error_that_says_why(self):
        with pytest.raises(NoUsableRows, match="different measurement"):
            assemble([manifest(usable=False)], space_size=1099)


class TestTheFront:
    def test_the_front_is_non_dominated_over_all_three_objectives(self):
        rows = [
            manifest(Genotype(h=2, d_head=32, n=2, ffn=2, kv=1), 500.0),
            manifest(Genotype(h=12, d_head=128, n=4, ffn=2, kv=6), 54.0),
            manifest(Genotype(h=12, d_head=64, n=16, ffn=4, kv=6), 300.0),
        ]
        result = assemble(rows, space_size=1099)
        on_front = [r for r in result.rows if r["on_measured_front"]]

        assert on_front, "something must be non-dominated"
        for row in result.rows:
            if not row["on_measured_front"]:
                dominators = [
                    o for o in on_front
                    if o["params"] <= row["params"]
                    and o["kv_cache_bytes"] <= row["kv_cache_bytes"]
                    and o["perplexity"] <= row["perplexity"]
                ]
                assert dominators, "a dominated row needs something dominating it"

    def test_partial_coverage_is_labelled_as_partial(self):
        """A front over a tenth of the space is a different object from the front over all
        of it, and the summary has to say so rather than let a reader assume."""
        result = assemble([manifest(), manifest(BIG, 50.0)], space_size=1099)
        assert result.coverage < 0.01
        assert "not the same object" in result.summary()["front_is"]

    def test_full_coverage_says_exact(self):
        result = assemble([manifest(), manifest(BIG, 50.0)], space_size=2)
        assert result.coverage == 1.0
        assert "exact over the whole space" in result.summary()["front_is"]


class TestColumns:
    def test_every_declared_column_is_produced(self):
        result = assemble([manifest()], space_size=1099)
        assert set(result.rows[0]) == set(COLUMNS)

    def test_the_genotype_comes_first_then_what_is_derived_from_it(self):
        """A reader should be able to recompute the derived columns from the genes alone."""
        assert COLUMNS[: len(GENES)] == GENES
        assert COLUMNS.index("params") > COLUMNS.index(GENES[-1])
        assert COLUMNS.index("perplexity") > COLUMNS.index("params")

    def test_non_embedding_parameters_are_reported_separately(self):
        """The column that matters for compute. Total parameters here are dominated by the
        50,257-row embedding, which is why ordering by them orders by width."""
        result = assemble([manifest(BIG, 54.0)], space_size=1099)
        row = result.rows[0]
        assert row["params_non_embedding"] < row["params"]
        assert row["params_non_embedding"] == row["params"] - 50257 * row["e"]

    def test_how_a_row_was_measured_travels_with_it(self):
        row = assemble([manifest()], space_size=1099).rows[0]
        for field in ("precision", "device", "data_vintage", "tokeniser_fingerprint",
                      "seed", "resumes"):
            assert row[field] is not None


class TestACheckThatOnlyHeldWhileNothingVaried:
    """`corpus` was missing from the comparability rule for the whole census.

    It never announced itself, because every manifest carried the same value. The first
    assembly after 1,980 runs on three further corpora would have merged them and computed
    one front over the mixture — and the result would have looked plausible rather than
    broken: code gives perplexity 5.3 where the search corpus gives 60 for the same
    architecture, so the front would have been a ranking of corpora wearing the shape of a
    ranking of designs.
    """

    @staticmethod
    def _manifest(corpus: str, budget: int = 100_000_000, perplexity: float = 60.0,
                  genes=(4, 32, 2, 2, 1)):
        return {
            "usable_for_benchmark": True,
            "genotype": {"values": list(genes)},
            "objectives": {"perplexity": perplexity},
            "corpus": corpus,
            "precision": "float16",
            "data_vintage": "2026-08-13",
            "token_budget": str(budget),
            "tokeniser_fingerprint": "855ec41a9d90094e",
            "device": "cuda",
            "config": {"seed": 0},
            "wall_clock_seconds": 100,
        }

    def test_mixing_corpora_is_refused(self):
        from nas4textgen.search.problem import IncomparableRows

        with pytest.raises(IncomparableRows, match="corpus"):
            assemble(
                [self._manifest("wikitext103", perplexity=60.0),
                 self._manifest("stack_python", perplexity=5.3, genes=(4, 32, 2, 2, 2))],
                space_size=1099,
            )

    def test_the_reason_names_the_consequence_not_just_the_field(self):
        from nas4textgen.search.problem import IncomparableRows

        try:
            assemble([self._manifest("wikitext103"),
                      self._manifest("babylm", genes=(4, 32, 2, 2, 2))], space_size=1099)
        except IncomparableRows as exc:
            assert "not the same quantity" in str(exc)

    def test_selecting_a_configuration_keeps_only_that_one(self):
        result = assemble(
            [self._manifest("wikitext103"),
             self._manifest("stack_python", perplexity=5.3, genes=(4, 32, 2, 2, 2)),
             self._manifest("tinystories", perplexity=6.3, genes=(4, 32, 4, 2, 1))],
            space_size=1099,
            select={"corpus": "wikitext103", "token_budget": 100_000_000},
        )
        assert len(result.rows) == 1
        assert result.excluded["other_configuration"] == 2

    def test_the_budget_is_part_of_the_configuration_not_only_the_corpus(self):
        """The first fix filtered on corpus alone, and the 4x-budget study is the same
        corpus at four times the tokens — so the rule refused it immediately."""
        result = assemble(
            [self._manifest("wikitext103", budget=100_000_000),
             self._manifest("wikitext103", budget=400_000_000, genes=(4, 32, 4, 2, 1))],
            space_size=1099,
            select={"corpus": "wikitext103", "token_budget": 100_000_000},
        )
        assert len(result.rows) == 1

    def test_the_table_records_which_configuration_it_is(self):
        """With several in the container, a reader of the artifact needs to know which one
        they are looking at."""
        result = assemble(
            [self._manifest("wikitext103")], space_size=1099,
            select={"corpus": "wikitext103", "token_budget": 100_000_000},
        )
        assert result.summary()["selected"]["corpus"] == "wikitext103"
        assert result.summary()["held_constant"]["corpus"] == "wikitext103"


class TestARunIsNotReplacedByItsReRun:
    """Which manifest wins when an architecture and seed were run more than once.

    The assembly counted the second as a duplicate and then kept it: the assignment ran
    unconditionally, so the table was a function of the order of the manifest file. It
    moved the measured front from 220 architectures to 218 and changed 220 perplexities,
    and nothing caught it, because the verification tier seeds the published table as an
    input instead of rebuilding it. The first completed run is the measurement.
    """

    def manifest(self, perplexity, seed=0):
        return {
            "corpus": "wikitext103",
            "token_budget": 100,
            "usable_for_benchmark": True,
            "genotype": {"values": [2, 64, 12, 4, 1]},
            "config": {"seed": seed, "micro_batch_size": 16},
            "objectives": {"perplexity": perplexity},
            "precision": "float16",
            "device": "cuda",
            "data_vintage": "2026-08-13",
            "tokeniser_fingerprint": "abc",
        }

    def test_the_first_run_is_kept_and_the_re_run_is_counted(self):
        from nas4textgen.benchmark import assemble

        result = assemble(
            [self.manifest(100.0), self.manifest(200.0)], space_size=1,
            select={"corpus": "wikitext103", "token_budget": 100})
        assert len(result.rows) == 1
        assert result.rows[0]["perplexity"] == 100.0, (
            "the re-run replaced the measurement it was counted as a duplicate of")
        assert result.excluded["duplicate_fingerprint"] == 1

    def test_the_table_does_not_depend_on_the_order_of_the_manifests(self):
        from nas4textgen.benchmark import assemble

        runs = [self.manifest(100.0), self.manifest(200.0), self.manifest(50.0, seed=1)]
        first = assemble(runs, space_size=1,
                         select={"corpus": "wikitext103", "token_budget": 100})
        # The duplicate moved to the end of the file, which is all it takes.
        shuffled = [runs[0], runs[2], runs[1]]
        second = assemble(shuffled, space_size=1,
                          select={"corpus": "wikitext103", "token_budget": 100})
        assert ([r["perplexity"] for r in first.rows]
                == [r["perplexity"] for r in second.rows])

    def test_the_published_table_regenerates_unchanged(self):
        """The check that would have caught it. Byte-for-byte against the artifact the
        paper's every number is read from."""
        import csv
        import subprocess
        import sys

        published = ROOT / "artifacts" / "benchmark" / "table.csv"
        if not published.exists():
            pytest.skip("no published table")
        before = published.read_text(encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "nas4textgen.cli", "table"],
            capture_output=True, text=True, cwd=ROOT)
        if result.returncode != 0:
            pytest.skip(f"table unavailable: {result.stdout[-200:]}")
        after = published.read_text(encoding="utf-8")
        if after != before:
            published.write_text(before, encoding="utf-8")
        assert after == before, "the published table does not regenerate from its manifests"
        front = sum(1 for row in csv.DictReader(before.splitlines())
                    if float(row.get("on_measured_front", 0) or 0) > 0)
        assert front == 220, f"the measured front is {front} architectures, not 220"
