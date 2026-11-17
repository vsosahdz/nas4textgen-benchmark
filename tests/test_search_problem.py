"""The benchmark table as a search problem, and what the two representations cost.

The comparison these tests protect is the second paper's central one, and its validity
rests on the algorithms being scored against the exact front rather than against each
other's best effort. Everything here uses a constructed table so the right answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymoo")

from nas4textgen.search import (  # noqa: E402
    BenchmarkTable,
    UnmeasuredArchitecture,
    build_choice_problem,
    build_vector_problem,
    decode_vector,
)
from nas4textgen.space.cost import parameter_count  # noqa: E402
from nas4textgen.space.enumerate import enumerate_with_ceiling  # noqa: E402
from nas4textgen.space.genotype import GENES, LEVELS, Genotype  # noqa: E402


@pytest.fixture(scope="module")
def table() -> BenchmarkTable:
    """Real cost objectives, a synthetic quality that genuinely trades off against them."""
    rng = np.random.default_rng(0)
    rows = {}
    for genotype in enumerate_with_ceiling().genotypes:
        params = parameter_count(genotype)
        quality = 5000.0 / np.log10(max(10, params)) + rng.normal(0, 20)
        rows[genotype.as_tuple()] = np.array([float(params), quality])
    return BenchmarkTable(rows, ("params", "perplexity"), (True, True), source="synthetic")


class TestTable:
    def test_the_true_front_is_known_because_everything_was_measured(self, table):
        """The reason the benchmark exists. Scoring against the best front any algorithm
        found rewards whoever ran longest and leaves the distance to the answer unknown."""
        front = table.true_front()
        assert 1 < len(front) < len(table)

        # Nothing in the table dominates any point on it.
        points = table.matrix()
        for row in front:
            dominated = np.all(points <= row, axis=1) & np.any(points < row, axis=1)
            assert not dominated.any()

    def test_an_unmeasured_architecture_is_an_error_not_a_default(self, table):
        """A search that reaches an unmeasured point is searching a different space than
        the one that was measured."""
        off_grid = Genotype(h=16, d_head=128, n=24, ffn=4, kv=16)
        assert off_grid.as_tuple() not in table.rows
        with pytest.raises(UnmeasuredArchitecture, match="different space"):
            table.lookup(off_grid)

    def test_an_empty_table_cannot_be_searched(self):
        with pytest.raises(ValueError, match="empty table"):
            BenchmarkTable({}, ("f",), (True,))

    def test_every_objective_needs_a_declared_direction(self):
        with pytest.raises(ValueError, match="declared direction"):
            BenchmarkTable({(1,): np.array([1.0, 2.0])}, ("a", "b"), (True,))


class TestVectorRepresentation:
    def test_decoding_covers_every_level_of_every_gene(self):
        """A mapping that could never produce some level would silently shrink the space."""
        for position, gene in enumerate(GENES):
            reached = set()
            for u in np.linspace(0, 1, 2001):
                row = np.full(len(GENES), 0.5)
                row[position] = u
                reached.add(getattr(decode_vector(row), gene))
            assert reached == set(LEVELS[gene]), f"{gene} is not fully reachable"

    def test_the_upper_bound_maps_inside_the_grid(self):
        """A vector of exactly 1.0 must not index past the last level."""
        genotype = decode_vector(np.ones(len(GENES)))
        for gene in GENES:
            assert getattr(genotype, gene) == LEVELS[gene][-1]


class TestQueryAccounting:
    def test_queries_are_counted_and_duplicates_are_visible(self, table):
        """Final hypervolume rewards whoever ran longest; query efficiency does not, and
        an algorithm that re-evaluates a point has spent budget without buying anything."""
        problem = build_vector_problem(table)
        lookup = problem.lookup

        # A vector that decodes to a feasible point. The midpoint of the space does not:
        # 0.5 everywhere gives h=8 with kv=6, which is the representation's problem in
        # miniature and is covered separately below.
        row = np.zeros((1, len(GENES)))
        assert decode_vector(row[0]).is_valid()

        problem.evaluate(row)
        problem.evaluate(row)

        assert lookup.n_evaluations == 2
        assert lookup.n_distinct == 1, "the repeat is visible as waste"

    def test_an_invalid_genotype_costs_a_query_and_reports_a_violation(self, table):
        """Repairing silently would credit an algorithm with candidates it did not
        produce; staying inside a constrained grid is part of what is being compared."""
        problem = build_choice_problem(table)
        before = problem.lookup.infeasible_queries

        out: dict = {}
        problem._evaluate({"h": 2, "d_head": 32, "n": 2, "ffn": 2, "kv": 12}, out)

        assert out["G"][0] > 0, "kv=12 does not divide h=2"
        assert problem.lookup.infeasible_queries == before + 1

    def test_the_violation_has_a_gradient_rather_than_being_a_wall(self, table):
        """A constant penalty gives an algorithm nothing to follow back to feasibility.

        The obvious formula, h % kv / h, is exactly that wall: whenever kv exceeds h the
        remainder is h, so kv=12 and kv=16 at h=6 both score 1.0 and an algorithm cannot
        tell it is getting closer. Distance to the nearest feasible kv has a gradient
        everywhere and points at the repair.
        """
        from nas4textgen.search.problem import constraint_violation

        at_12 = constraint_violation(Genotype(h=6, d_head=32, n=2, ffn=2, kv=12))
        at_16 = constraint_violation(Genotype(h=6, d_head=32, n=2, ffn=2, kv=16))
        assert 0 < at_12 < at_16, "closer to a divisor of 6 must score lower"

        naive = [g.h % g.kv / g.h for g in (
            Genotype(h=6, d_head=32, n=2, ffn=2, kv=12),
            Genotype(h=6, d_head=32, n=2, ffn=2, kv=16),
        )]
        assert naive[0] == naive[1] == 1.0, "the formula this replaces had no gradient here"

        assert constraint_violation(Genotype(h=6, d_head=32, n=2, ffn=2, kv=6)) == 0.0

    def test_reset_clears_the_trace(self, table):
        problem = build_vector_problem(table)
        problem.evaluate(np.full((3, len(GENES)), 0.5))
        problem.lookup.reset()
        assert problem.lookup.n_evaluations == 0
        assert problem.lookup.infeasible_queries == 0


class TestBothRepresentationsRun:
    """Each representation distorts the problem differently, and neither is free.

    The vector encoding rounds, which makes the landscape piecewise-constant: a crossover
    interpolating between two parents frequently returns a child identical to one of them,
    so the run spends much of its budget re-querying points it already has. The categorical
    encoding samples kv independently of h, and most of those combinations violate the
    divisibility constraint.

    An algorithm that wins under only one of them has told you about the encoding rather
    than about itself, which is why both are reported.
    """

    def _run(self, problem, generations=10):
        from pymoo.optimize import minimize

        if problem.representation.startswith("categorical"):
            from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
            from pymoo.core.mixed import MixedVariableGA

            algorithm = MixedVariableGA(pop_size=30, survival=RankAndCrowdingSurvival())
        else:
            from pymoo.algorithms.moo.nsga2 import NSGA2

            algorithm = NSGA2(pop_size=30)

        minimize(problem, algorithm, ("n_gen", generations), seed=1, verbose=False)
        return problem.lookup

    def test_the_vector_representation_wastes_budget_on_duplicates(self, table):
        lookup = self._run(build_vector_problem(table))
        assert lookup.n_evaluations > 0
        duplicate_share = 1 - lookup.n_distinct / lookup.n_evaluations
        assert duplicate_share > 0.3, (
            "rounding a real vector onto a coarse grid should produce many repeats; "
            f"saw {duplicate_share:.0%}"
        )

    def test_the_categorical_representation_explores_more_distinct_points(self, table):
        lookup = self._run(build_choice_problem(table))
        duplicate_share = 1 - lookup.n_distinct / max(1, lookup.n_evaluations)
        assert duplicate_share < 0.3, f"saw {duplicate_share:.0%} repeats"

    def test_the_categorical_representation_pays_in_infeasible_candidates(self, table):
        """Sampling kv independently of h means most draws violate divisibility. That is
        the cost of the encoding that avoids the rounding problem, and it belongs in the
        results rather than being repaired away."""
        lookup = self._run(build_choice_problem(table))
        total = lookup.n_evaluations + lookup.infeasible_queries
        assert lookup.infeasible_queries / total > 0.3


class TestComparability:
    """A row is comparable to another only if everything but the architecture was held.

    Some of what must be held is not part of what a job asks for, which is how the canary
    produced three architectures in float16 and one in emulated bfloat16. All four
    completed their budget, all four wrote valid manifests, and nothing about the
    perplexities announced that one came from different arithmetic.
    """

    def row(self, **overrides):
        base = {
            "precision": "float16",
            "data_vintage": "2026-08-13",
            "token_budget": 100_000_000,
            "tokeniser_fingerprint": "855ec41a9d90094e",
        }
        base.update(overrides)
        return base

    def test_rows_measured_alike_are_accepted(self):
        from nas4textgen.search import require_comparable

        agreed = require_comparable([self.row(), self.row(), self.row()])
        assert agreed["precision"] == "float16"
        assert agreed["token_budget"] == "100000000"

    def test_mixed_precision_is_refused(self):
        """The case that actually happened."""
        from nas4textgen.search import IncomparableRows, require_comparable

        with pytest.raises(IncomparableRows, match="not the same measurement"):
            require_comparable([self.row(), self.row(precision="bfloat16")])

    def test_a_moved_corpus_is_refused(self):
        from nas4textgen.search import IncomparableRows, require_comparable

        with pytest.raises(IncomparableRows, match="different text"):
            require_comparable([self.row(), self.row(data_vintage="2026-09-01")])

    def test_unequal_training_is_refused_by_name(self):
        """The prior work's defect, made unrepresentable rather than merely discouraged."""
        from nas4textgen.search import IncomparableRows, require_comparable

        with pytest.raises(IncomparableRows, match="defect this project exists to correct"):
            require_comparable([self.row(), self.row(token_budget=50_000_000)])

    def test_a_missing_field_counts_as_a_disagreement(self):
        """A manifest that never recorded the precision cannot be asserted comparable."""
        from nas4textgen.search import IncomparableRows, require_comparable

        incomplete = self.row()
        del incomplete["precision"]
        with pytest.raises(IncomparableRows, match="missing"):
            require_comparable([self.row(), incomplete])

    def test_things_that_do_not_change_the_number_are_not_checked(self):
        """Wall clock and micro-batch differ legitimately across a sweep; refusing on them
        would block every table that was ever assembled."""
        from nas4textgen.search import require_comparable

        require_comparable([
            self.row(wall_clock_seconds=100, micro_batch_used=32),
            self.row(wall_clock_seconds=9000, micro_batch_used=4),
        ])
