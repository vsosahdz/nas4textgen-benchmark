"""The benchmark table as a pymoo problem, so search algorithms query it instead of training.

Once every architecture has been measured, comparing search algorithms no longer needs a
GPU: an "evaluation" is a lookup. That is what makes a slate of seven algorithms over many
seeds affordable, and it is also what makes the comparison exact — the true Pareto front is
known, so a run can be scored against the answer rather than against the best thing anyone
happened to find.

**Two representations, because the choice of representation is itself a result.** The
genotype is five categorical or ordinal genes, and most multi-objective algorithms assume a
real vector. The usual move is to encode the genes as normalised reals and round on
evaluation, which is convenient and quietly changes the problem: rounding makes the
landscape piecewise-constant, so a crossover that interpolates between two parents can
produce a child identical to one of them, and neighbourhood structure in MOEA/D is measured
in a space where distance does not correspond to architectural similarity. The categorical
representation avoids that but excludes algorithms that require a real vector.

Reporting both is the honest arrangement: an algorithm that wins only under one of them has
told you something about the encoding rather than about itself.

**Infeasible points exist and are not silently repaired.** The grid contains genotypes where
kv does not divide h, and architectures above the parameter ceiling. Repairing them
invisibly would let an algorithm spend its budget generating invalid candidates and be
credited with the repaired ones. Here an invalid candidate costs a query and returns the
constraint violation, which is what an algorithm should be judged on handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..space.genotype import GENES, LEVELS, Genotype


class UnmeasuredArchitecture(KeyError):
    """Raised when the table has no row for a genotype the search asked for."""


class IncomparableRows(RuntimeError):
    """Raised when rows in a benchmark table were not measured under one protocol."""


def require_comparable(manifests: list[dict]) -> dict[str, str]:
    """Refuse a table whose rows were measured under different conditions.

    A benchmark row is only comparable to another if everything except the architecture was
    held constant, and some of what must be held constant is not part of what a job asks
    for. The canary produced exactly this: three architectures measured in float16 and one
    in emulated bfloat16, because the image changed between them. All four completed their
    budget, all four wrote valid manifests, and nothing about the perplexities announces
    that one of them came from different arithmetic.

    So the check happens where the table is assembled, not where a job is submitted. The
    fields below are the ones a difference in would make the numbers answer different
    questions; a difference in wall clock or micro-batch would not, and is not checked.

    ``corpus`` was missing from this list until the multi-corpus study ran, and its absence
    was invisible for the whole census because every manifest carried the same value. The
    first assembly after 1,980 runs on three further corpora would have merged them all into
    one table and computed a front over it. A check that only holds while the thing it
    checks never varies is not a check, and this one survived that way for a month.
    """
    required = {
        "corpus": (
            "perplexity on different text is not the same quantity — the transfer study "
            "measures 5.3 on code where the search corpus gives 60 for the same "
            "architecture, so a front over the mixture would be a ranking of corpora "
            "wearing the shape of a ranking of designs"
        ),
        "precision": "arithmetic differs, so the perplexities are not the same measurement",
        "data_vintage": "the corpus moved, so the models saw different text",
        "token_budget": "unequal training is the defect this project exists to correct",
        "tokeniser_fingerprint": "a different vocabulary is a different objective",
    }

    if not manifests:
        raise IncomparableRows("no manifests to assemble a table from")

    agreed: dict[str, str] = {}
    for held, why in required.items():
        values = {str(m.get(held, "<missing>")) for m in manifests}
        if len(values) > 1:
            raise IncomparableRows(
                f"{len(manifests)} rows disagree on {held!r}: {sorted(values)}.\n"
                f"Why it matters: {why}.\n"
                "Re-run the minority under the majority's conditions, or split the table "
                "and say which rows belong to which protocol. Mixing them silently is how "
                "a benchmark reports a comparison it did not make."
            )
        agreed[held] = values.pop()
    return agreed


@dataclass
class BenchmarkTable:
    """Measured objectives for every architecture, keyed by genotype.

    The ground truth for every algorithm comparison. Built from the sweep's manifests, so
    a row exists only where a run met its budget — a truncated run measured something, but
    not the same thing, and admitting it would put an incomparable point on the front.
    """

    #: genotype tuple -> objective vector, in the declared objective order.
    rows: dict[tuple[int, ...], np.ndarray]
    objective_names: tuple[str, ...]
    #: True where lower is better. Objectives are internally minimised, so anything
    #: maximised is negated on the way in and the sign is recorded here rather than
    #: applied twice by accident.
    minimise: tuple[bool, ...]
    source: str = "measured sweep"

    def __post_init__(self) -> None:
        if len(self.objective_names) != len(self.minimise):
            raise ValueError("every objective needs a declared direction")
        if not self.rows:
            raise ValueError("an empty table cannot be searched")
        width = len(self.objective_names)
        for key, value in self.rows.items():
            if len(value) != width:
                raise ValueError(
                    f"row {key} has {len(value)} objectives against {width} declared"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def lookup(self, genotype: Genotype) -> np.ndarray:
        key = genotype.as_tuple()
        try:
            return self.rows[key]
        except KeyError as exc:
            raise UnmeasuredArchitecture(
                f"{genotype} is not in the table ({len(self.rows)} measured "
                f"architectures from {self.source}). A search that reaches an unmeasured "
                "point is searching a different space than the one that was measured."
            ) from exc

    def matrix(self) -> np.ndarray:
        """Every measured objective vector, in insertion order."""
        return np.vstack(list(self.rows.values()))

    def genotypes(self) -> list[Genotype]:
        return [Genotype(**dict(zip(GENES, key, strict=True))) for key in self.rows]

    def true_front(self) -> np.ndarray:
        """The exact non-dominated set. Known because everything was measured.

        This is the reason the benchmark exists. Scoring against the best front any
        algorithm found rewards the algorithm that ran longest, and the distance to the
        real answer stays unknown.
        """
        from ..space.enumerate import non_dominated

        points = self.matrix()
        return points[non_dominated(points)]


@dataclass
class LookupProblem:
    """Shared machinery for both representations: query counting and constraint handling."""

    table: BenchmarkTable
    #: Every genotype the search asked for, in order. The query trace is the measurement
    #: that matters — final hypervolume rewards whoever ran longest, query efficiency does
    #: not.
    queries: list[tuple[int, ...]] = field(default_factory=list)
    infeasible_queries: int = 0

    @property
    def n_evaluations(self) -> int:
        return len(self.queries)

    @property
    def n_distinct(self) -> int:
        """Distinct architectures queried. An algorithm that re-evaluates the same point
        has spent budget without buying information, and the gap between this and
        n_evaluations is exactly that waste."""
        return len(set(self.queries))

    def reset(self) -> None:
        self.queries.clear()
        self.infeasible_queries = 0

    def _evaluate_one(self, genotype: Genotype) -> tuple[np.ndarray, float]:
        """Objectives and constraint violation for one candidate.

        A violation is returned rather than repaired. Repairing silently would credit an
        algorithm with candidates it did not produce, and the ability to stay inside a
        constrained grid is part of what is being compared.
        """
        if not genotype.is_valid():
            self.infeasible_queries += 1
            worst = self.table.matrix().max(axis=0)
            return worst, constraint_violation(genotype)

        key = genotype.as_tuple()
        if key not in self.table.rows:
            self.infeasible_queries += 1
            worst = self.table.matrix().max(axis=0)
            return worst, 1.0

        # Repeats are recorded for the waste metric but do not consume budget.
        #
        # A budget counted in queries compares methods on different numbers of real
        # architectures: on this table NSGA-II reaches 23 distinct points in 158 queries
        # while random search reaches 115 in 149, so scoring both "at 120 evaluations"
        # compares 115 trainings against 23. In a campaign, re-asking about an architecture
        # already trained is free — the answer is already on disk — so distinct evaluations
        # is what a budget actually buys.
        self.queries.append(key)
        return self.table.rows[key], 0.0

    def distinct_prefix(self, budget: int) -> list[tuple[int, ...]]:
        """The queries up to the point where ``budget`` distinct architectures were seen."""
        seen: set[tuple[int, ...]] = set()
        prefix: list[tuple[int, ...]] = []
        for key in self.queries:
            prefix.append(key)
            seen.add(key)
            if len(seen) >= budget:
                break
        return prefix


def _levels_for(gene: str) -> tuple[int, ...]:
    return LEVELS[gene]


def constraint_violation(genotype: Genotype) -> float:
    """How far this genotype is from satisfying kv | h, on a scale an algorithm can follow.

    The obvious formula, ``h % kv / h``, has no gradient where one is most needed: whenever
    kv exceeds h the remainder is h itself, so every such candidate scores exactly 1.0 and
    an algorithm at kv=16, h=6 gets the same signal as one at kv=12, h=6. It cannot tell
    that it is getting closer.

    Distance to the nearest feasible kv does have a gradient everywhere, and it points at a
    repair rather than merely reporting failure.
    """
    if genotype.is_valid():
        return 0.0

    levels = _levels_for("kv")
    feasible = [level for level in levels if genotype.h % level == 0]
    if not feasible:
        # h has no divisor among the kv levels at all, which the grid does not contain,
        # but a search over the vector encoding can construct it.
        return 1.0

    nearest = min(abs(genotype.kv - level) for level in feasible)
    span = max(levels) - min(levels)
    return max(nearest / span, 1e-6)


def build_vector_problem(table: BenchmarkTable):
    """The normalised-real representation, for algorithms that need a continuous vector.

    Each gene is a real in [0, 1) mapped to its level by flooring. Convenient and standard,
    and it changes the problem: the landscape becomes piecewise-constant, so a crossover
    interpolating between parents often returns a child identical to one of them, and
    MOEA/D's neighbourhoods are measured in a space where distance is not architectural
    similarity. Reported alongside the categorical version rather than instead of it.
    """
    from pymoo.core.problem import Problem

    lookup = LookupProblem(table)
    n_obj = len(table.objective_names)

    class VectorLookup(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=len(GENES),
                n_obj=n_obj,
                n_ieq_constr=1,
                xl=np.zeros(len(GENES)),
                xu=np.ones(len(GENES)),
            )
            self.lookup = lookup
            self.representation = "normalised vector"

        def _evaluate(self, x, out, *_, **__):
            objectives = np.empty((len(x), n_obj))
            violations = np.empty((len(x), 1))
            for i, row in enumerate(x):
                genotype = decode_vector(row)
                values, violation = lookup._evaluate_one(genotype)
                objectives[i] = values
                violations[i] = violation
            out["F"] = objectives
            out["G"] = violations

    return VectorLookup()


def decode_vector(row: np.ndarray) -> Genotype:
    """Map a vector in [0, 1]^5 to a genotype by flooring into each gene's levels."""
    values = {}
    for position, gene in enumerate(GENES):
        levels = _levels_for(gene)
        index = int(np.clip(row[position], 0.0, 1.0 - 1e-12) * len(levels))
        values[gene] = levels[index]
    return Genotype(**values)


def build_choice_problem(table: BenchmarkTable):
    """The categorical representation, using pymoo's mixed-variable ``Choice``.

    The genes are what they are: unordered or ordinal levels, not points on a line.
    Operators here recombine by choosing levels rather than by interpolating between them,
    which is the operation the space actually admits.
    """
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.variable import Choice

    lookup = LookupProblem(table)
    n_obj = len(table.objective_names)

    class ChoiceLookup(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                vars={gene: Choice(options=list(_levels_for(gene))) for gene in GENES},
                n_obj=n_obj,
                n_ieq_constr=1,
            )
            self.lookup = lookup
            self.representation = "categorical choice"

        def _evaluate(self, x, out, *_, **__):
            genotype = Genotype(**{gene: int(x[gene]) for gene in GENES})
            values, violation = lookup._evaluate_one(genotype)
            out["F"] = np.asarray(values, dtype=float)
            out["G"] = np.asarray([violation], dtype=float)

    return ChoiceLookup()
