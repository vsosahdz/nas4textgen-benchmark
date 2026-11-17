"""Run one search algorithm over the benchmark table and record what it did.

SMS-EMOA is the reference algorithm rather than NSGA-II, and the reason is the property
being studied. SMS-EMOA is steady-state: it generates one offspring and removes whichever
individual contributes least to the population's hypervolume. That has to be asked for —
pymoo generates a full generation of offspring by default, and the guarantee does not
survive it. That makes population
hypervolume non-decreasing by construction — the operator cannot discard hypervolume it
already has.

That matters for reading a convergence curve. Under NSGA-II, truncation by crowding
distance carries no such guarantee, so its population hypervolume can fall, and a fall
means nothing in particular: it might be the algorithm exploring, or a defect in the
harness, and the curve cannot distinguish them. Under SMS-EMOA a fall can only be a defect,
which turns the curve into an instrument rather than an observation.

NSGA-II stays in the slate as a comparator — it is what most NAS papers use, so excluding
it would make these results incomparable with the literature. It is simply not the
algorithm the convergence analysis is built on.

The population is snapshotted at every evaluation, because without it the recorded
hypervolume is the archive's, which rises monotonically for every algorithm including
random search and therefore says nothing about the search at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import BenchmarkTable, build_choice_problem, build_vector_problem
from .scoring import QueryTrace, score_trace


class UnsupportedAlgorithm(RuntimeError):
    """Raised when an algorithm cannot run this problem as declared."""


@dataclass(frozen=True)
class AlgorithmSpec:
    """One entry in the slate, with why it is there."""

    name: str
    monotone_population: bool
    role: str
    needs_real_vector: bool = False


#: The slate. Every entry states what it is for, so a reader can see the comparison was
#: designed rather than assembled from whatever was available.
SLATE: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        "random", False,
        "the floor. An algorithm that does not beat uniform sampling of the grid has "
        "bought nothing, and on a space of 1,099 points that floor is not low.",
    ),
    AlgorithmSpec(
        "sms-emoa", True,
        "the reference. Steady-state, removing the least hypervolume contribution, so its "
        "population hypervolume cannot decrease and a fall in the curve is a defect rather "
        "than an observation.",
    ),
    AlgorithmSpec(
        "nsga2", False,
        "the comparator. What most NAS papers use, so excluding it would make these "
        "results incomparable with the literature.",
    ),
    AlgorithmSpec(
        "age-moea2", False,
        "adapts to front geometry rather than assuming it, which matters here because the "
        "measured front's shape is not known in advance.",
        needs_real_vector=True,
    ),
    AlgorithmSpec(
        "pareto-local-search", False,
        "the method the space actually admits. Five ordinal genes on a grid make a "
        "neighbourhood exact and small, so it has no operators to tune and no "
        "representation to argue about — which is what makes it the cleanest baseline here, "
        "and it is absent from most NAS comparisons because they reach for population "
        "methods designed for continuous spaces.",
    ),
    AlgorithmSpec(
        "parego", False,
        "the surrogate-assisted entry. The budget curve already found a model needs about a "
        "hundred labels here before it outranks sorting by parameter count, so its early "
        "iterations should be poor and where it overtakes is the interesting part.",
    ),
    AlgorithmSpec(
        "moead", False,
        "decomposition rather than dominance, and the entry most exposed to the "
        "representation question because its neighbourhoods are metric. Excluded in "
        "practice: pymoo's implementation asserts it supports no constraints, and this "
        "problem declares one. Giving it a repaired problem would make it the only entry "
        "solving a different one.",
        needs_real_vector=True,
    ),
)


def build_algorithm(name: str, representation: str, pop_size: int = 30):
    """Construct one algorithm for one representation, or say why it cannot be."""
    categorical = representation.startswith("categorical")

    if name == "sms-emoa":
        if categorical:
            from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
            from pymoo.core.mixed import MixedVariableGA

            # pymoo's mixed-variable machinery does not carry SMS-EMOA's survival, so the
            # categorical arm loses the monotonicity guarantee. Saying so is better than
            # labelling a run SMS-EMOA when its survival operator is not SMS-EMOA's.
            return MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival()), False
        from pymoo.algorithms.moo.sms import SMSEMOA

        # ``n_offsprings=1`` is what makes this SMS-EMOA rather than something named after
        # it. The published algorithm is steady-state: one offspring per iteration, and the
        # individual contributing least hypervolume is removed — which is why the
        # population's hypervolume cannot decrease, and why it was chosen as the instrument
        # for reading convergence.
        #
        # pymoo defaults ``n_offsprings`` to the population size, making it generational.
        # Removing mu individuals from 2*mu by greedy least-contribution does not keep the
        # best mu-subset, so the guarantee is lost. Measured rather than reasoned about:
        # under the defaults the population hypervolume fell in all three seeds tried
        # (worst steps -1.0e-4, -7.6e-5, -5.4e-5); with one offspring it fell in none, the
        # worst step being exactly zero. Setting ``normalize=False`` as well changed
        # nothing, so the offspring count is the whole cause.
        #
        # The assertion downstream caught this. It exists because a monotone algorithm is
        # only an instrument while it is monotone, and a fall that means "defect" is worth
        # more than a fall that might mean "exploration".
        return SMSEMOA(pop_size=pop_size, n_offsprings=1), True

    if name == "nsga2":
        if categorical:
            from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
            from pymoo.core.mixed import MixedVariableGA

            return MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival()), False
        from pymoo.algorithms.moo.nsga2 import NSGA2

        return NSGA2(pop_size=pop_size), False

    if name == "random":
        from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch

        return RandomSearch(), False

    if name == "age-moea2":
        # pymoo 0.6.2 calls np.row_stack, which numpy 2 removed in favour of np.vstack.
        # Shimmed rather than dropped: the alternative is excluding an algorithm the design
        # asked for because of a two-line incompatibility between two pinned dependencies,
        # which would be a fact about the versions rather than about the method.
        import numpy as _np

        # A bridge between two pinned dependencies, not a fix to either. pymoo 0.6.2
        # calls numpy names that numpy 2 removed, and this restores them by their new
        # equivalents.
        #
        # setattr with a string literal, deliberately: written as plain assignments, ruff's
        # numpy-2 deprecation rule rewrites the assignment *target*, turning the shim into
        # `np.vstack = np.vstack` — a no-op that lints clean and leaves the algorithm
        # broken. An autofix that edits the thing being defined is hard to see afterwards.
        # Every entry is an exact rename, which is the line this bridge does not cross: a
        # name numpy merely moved can be restored, a function whose behaviour changed
        # cannot, and shimming the second kind would make the algorithm's results a
        # property of this table rather than of the method.
        for removed, replacement in (
            ("row_stack", _np.vstack),
            ("trapz", _np.trapezoid),
            ("in1d", _np.isin),
        ):
            if not hasattr(_np, removed):
                setattr(_np, removed, replacement)
        from pymoo.algorithms.moo.age2 import AGEMOEA2

        return AGEMOEA2(pop_size=pop_size), False

    if name == "moead":
        raise UnsupportedAlgorithm(
            "pymoo's MOEA/D asserts it supports no constraints, and this problem declares "
            "one: kv must divide h, and repairing that silently would credit the algorithm "
            "with candidates it did not produce. Excluded and recorded rather than given a "
            "quietly different problem from every other entry."
        )

    raise ValueError(f"unknown algorithm {name!r}; the slate is {[a.name for a in SLATE]}")


def run_algorithm(
    table: BenchmarkTable,
    name: str,
    representation: str = "vector",
    seed: int = 0,
    generations: int = 30,
    pop_size: int = 30,
) -> QueryTrace:
    """Run one algorithm once and return its scored trace.

    The population is captured after every generation through a callback, so the returned
    trace carries both curves: what the algorithm evaluated, and what it was holding.
    """
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize

    problem = (
        build_choice_problem(table) if representation.startswith("categorical")
        else build_vector_problem(table)
    )
    algorithm, monotone_expected = build_algorithm(name, problem.representation, pop_size)

    snapshots: list[np.ndarray] = []
    query_counts: list[int] = []

    class Snapshot(Callback):
        def notify(self, algo):
            objectives = algo.pop.get("F")
            snapshots.append(np.asarray(objectives, dtype=float))
            query_counts.append(problem.lookup.n_evaluations)

    minimize(
        problem, algorithm, ("n_gen", generations),
        seed=seed, verbose=False, callback=Snapshot(),
    )

    queried = np.vstack([table.rows[key] for key in problem.lookup.queries]) if (
        problem.lookup.queries
    ) else np.empty((0, len(table.objective_names)))

    # Expand generation snapshots to one per evaluation, so the population curve lines up
    # with the archive curve on a shared x-axis of true evaluations.
    per_evaluation: list[np.ndarray] = []
    for index in range(len(queried)):
        position = np.searchsorted(query_counts, index + 1, side="left")
        per_evaluation.append(snapshots[min(position, len(snapshots) - 1)])

    points = table.matrix()
    trace = score_trace(
        queried,
        table.true_front(),
        lower=points.min(axis=0),
        upper=points.max(axis=0),
        population_snapshots=per_evaluation if snapshots else None,
        algorithm=name,
        representation=problem.representation,
        seed=seed,
        infeasible_queries=problem.lookup.infeasible_queries,
        duplicate_queries=problem.lookup.n_evaluations - problem.lookup.n_distinct,
    )

    if monotone_expected:
        from .scoring import MONOTONICITY_DRIFT_FRACTION, MONOTONICITY_TOLERANCE

        if not trace.population_is_monotone:
            raise AssertionError(
                f"{name} guarantees a non-decreasing population hypervolume and this run "
                f"fell by more than the measured drift tolerance of "
                f"{MONOTONICITY_TOLERANCE:.2e}. That is a defect in the harness, not a "
                "property of the search — which is exactly why a monotone algorithm is the "
                "instrument for this measurement."
            )
        if trace.population_drift > MONOTONICITY_DRIFT_FRACTION:
            raise AssertionError(
                f"{name} gave back {trace.population_drift:.2%} of the hypervolume it "
                f"gained, against a ceiling of {MONOTONICITY_DRIFT_FRACTION:.0%}. Each "
                "step may sit inside the drift tolerance while the run as a whole does "
                "not, and it is the run that a convergence curve shows."
            )
    return trace
