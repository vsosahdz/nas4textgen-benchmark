"""Two algorithms pymoo does not carry, and the budget accounting they exposed.

The slate's first run said random search beat NSGA-II by five times. It did not — the
budget was counted in queries, and the population methods spend most of theirs
re-evaluating architectures they had already seen.
"""

from __future__ import annotations

import numpy as np
import pytest

from nas4textgen.search.local import neighbours, parego, pareto_local_search
from nas4textgen.space.genotype import GENES, LEVELS, Genotype


@pytest.fixture(scope="module")
def table():
    from nas4textgen.search import BenchmarkTable
    from nas4textgen.space.cost import kv_cache_bytes, parameter_count
    from nas4textgen.space.enumerate import enumerate_with_ceiling

    rng = np.random.default_rng(0)
    rows = {}
    for genotype in enumerate_with_ceiling().genotypes:
        params = parameter_count(genotype)
        rows[genotype.as_tuple()] = np.array([
            float(params), float(kv_cache_bytes(genotype)),
            5000.0 / np.log10(max(10, params)) + rng.normal(0, 15),
        ])
    return BenchmarkTable(rows, ("params", "kv_bytes", "perplexity"),
                          (True, True, True), source="synthetic")


class TestNeighbourhood:
    def test_a_neighbour_differs_in_one_gene_by_one_level(self):
        """Ordinal steps, so a move is a small architectural change rather than a jump."""
        start = Genotype(h=8, d_head=64, n=8, ffn=3, kv=2)
        for other in neighbours(start):
            differing = [g for g in GENES if getattr(other, g) != getattr(start, g)]
            assert len(differing) == 1
            gene = differing[0]
            levels = LEVELS[gene]
            assert abs(levels.index(getattr(other, gene))
                       - levels.index(getattr(start, gene))) == 1

    def test_infeasible_neighbours_are_never_offered(self):
        """On a grid the neighbourhood is cheap to enumerate, so a search should not spend
        queries on points it can see are invalid."""
        for start in (Genotype(h=4, d_head=32, n=2, ffn=2, kv=4),
                      Genotype(h=12, d_head=64, n=6, ffn=3, kv=6)):
            assert all(n.is_valid() for n in neighbours(start))

    def test_the_extremes_have_fewer_neighbours_than_the_middle(self):
        smallest = Genotype(h=2, d_head=32, n=2, ffn=2, kv=1)
        middle = Genotype(h=8, d_head=64, n=8, ffn=3, kv=2)
        assert len(neighbours(smallest)) < len(neighbours(middle))


class TestTermination:
    def test_pareto_local_search_stops_at_its_budget(self, table):
        """The first version hung: a neighbour already in the archive was re-queued, and
        because a repeat lookup does not advance the budget the loop never ended."""
        search = pareto_local_search(table, budget=80, seed=0)
        assert search.spent <= 80
        assert search.spent > 10, "it must actually make progress"

    def test_parego_stops_at_its_budget(self, table):
        search = parego(table, budget=40, seed=0)
        assert search.spent <= 40

    def test_a_repeat_lookup_does_not_consume_budget(self, table):
        """Re-asking about an architecture already measured is free in a campaign too, and
        charging for it would penalise algorithms that keep an archive."""
        from nas4textgen.search.local import TableSearch

        search = TableSearch(table)
        genotype = table.genotypes()[0]
        search.evaluate(genotype)
        search.evaluate(genotype)
        assert search.spent == 1

    def test_an_architecture_outside_the_table_returns_nothing(self, table):
        from nas4textgen.search.local import TableSearch

        search = TableSearch(table)
        assert search.evaluate(Genotype(h=16, d_head=128, n=24, ffn=4, kv=16)) is None
        assert search.spent == 0


class TestDistinctBudget:
    def test_the_budget_axis_is_distinct_architectures(self, table):
        """A budget in queries compares methods on different numbers of real trainings. On
        the interim table NSGA-II reached 23 distinct points in 158 queries while random
        search reached 115 in 149, and scoring both at 120 would have compared 115
        trainings against 23 — which is what made random search look five times better.
        """
        from nas4textgen.search.scoring import score_trace

        repeated = np.vstack([table.matrix()[0]] * 20 + list(table.matrix()[1:11]))
        trace = score_trace(repeated, table.true_front(),
                            table.matrix().min(axis=0), table.matrix().max(axis=0))

        assert trace.evaluations[-1] == 30, "thirty queries"
        assert trace.distinct[-1] == 11, "eleven distinct architectures"
        # Asking for a budget of eleven must not be satisfied by the first eleven queries,
        # ten of which are the same point.
        assert trace.at_budget(11) == pytest.approx(trace.igd_plus[-1])

    def test_the_docstring_records_why(self):
        import inspect

        from nas4textgen.search.scoring import QueryTrace

        doc = inspect.getdoc(QueryTrace.at_budget)
        assert "23 distinct" in doc and "115" in doc


class TestSlateIntegrity:
    def test_moead_is_excluded_with_its_reason(self):
        """Giving it a repaired problem would make it the only entry solving a different
        one."""
        from nas4textgen.search.run import SLATE, UnsupportedAlgorithm, build_algorithm

        entry = next(a for a in SLATE if a.name == "moead")
        assert "no constraints" in entry.role
        with pytest.raises(UnsupportedAlgorithm, match="repairing that silently"):
            build_algorithm("moead", "normalised vector")

    def test_the_numpy_bridge_restores_only_exact_renames(self):
        """A name numpy moved can be restored; a function whose behaviour changed cannot,
        and shimming the second kind would make results a property of the shim table."""
        import inspect

        from nas4textgen.search.run import build_algorithm

        source = inspect.getsource(build_algorithm)
        assert 'setattr(_np, removed, replacement)' in source, (
            "written as a plain assignment, ruff rewrites the target and the shim becomes "
            "a no-op that lints clean"
        )
        assert "exact rename" in source

    def test_the_slate_covers_the_declared_families(self):
        from nas4textgen.search.run import SLATE

        names = {a.name for a in SLATE}
        assert {"random", "sms-emoa", "nsga2", "age-moea2",
                "pareto-local-search", "parego", "moead"} <= names
