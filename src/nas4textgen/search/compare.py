"""Every algorithm on the slate, over the complete table, against the exact front.

This is what the census bought. Because all 1,099 architectures are measured, the Pareto
front is known exactly rather than estimated, so the distance from an algorithm's result to
the front is a measurement instead of a comparison against whichever run happened to be
best. That is the difference between "our method found a better front than theirs" and "at
this budget this algorithm is this far from the true answer" — and only the second is
reusable by anyone else.

**The budget axis is distinct architectures, not evaluations.** A campaign pays to train an
architecture once; asking about one already measured costs nothing and is answered from the
table. Charging for repeats penalises algorithms that keep an archive over ones that forget,
and an earlier version of this comparison did exactly that: it reported random search
beating NSGA-II fivefold, which vanished once NSGA-II's re-queries stopped counting against
its budget — 23 distinct architectures in 158 evaluations against random's 115 in 149.

**The ROPE is pooled seed noise, not a chosen threshold.** Two algorithms differing by less
than the spread a single algorithm shows across its own seeds are not distinguishable at
this sample size, and the width comes from that spread rather than from a number that
looked reasonable. It is measured over every algorithm rather than one, because a ROPE
resting on one algorithm's variance would be a claim that its noise is everyone's.

**Blocks are seeds, treatments are algorithms.** Every algorithm sees the same seeds, so
the comparison controls for the run-to-run variation that a single-seed table would confuse
with a difference between methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Seeds every algorithm runs, and the number is derived rather than picked.
#:
#: Five was the first choice and it was too few. Friedman's statistic cannot exceed
#: blocks x (treatments - 1), so with six algorithms three blocks is the minimum at which
#: rejection is arithmetically possible at all — a bar that only says a perfectly
#: consistent result could reach significance. Detecting a moderate agreement across seeds
#: (Kendall's W around 0.3, which is what a real but not overwhelming difference between
#: search algorithms looks like) needs eight. Twenty leaves room for seeds that fail and
#: keeps the comparison powered when an algorithm drops out of some of them.
#:
#: Affordable because these searches query a table rather than train anything: seconds
#: each, no accelerator. The five-seed limit was inherited from the sweep, where a seed
#: costs an hour of GPU time, and carrying it here was thoughtless rather than reasoned.
SEEDS = tuple(range(20))

#: Where each algorithm's trace is read. The measurement is the gap to the exact front at a
#: fixed budget of distinct architectures, which is the quantity a reader of the benchmark
#: wants and the only one the census makes exact.
BUDGETS = (25, 50, 100, 200)


class NotEnoughRuns(RuntimeError):
    """Raised when the comparison has too few runs to derive its own noise floor."""


@dataclass
class RunResult:
    """One algorithm, one seed: what it reached at each budget."""

    algorithm: str
    seed: int
    #: budget -> IGD+ to the exact front, lower is better.
    igd_plus: dict[int, float] = field(default_factory=dict)
    #: budget -> hypervolume of the archive in the objectives' own units.
    hypervolume: dict[int, float] = field(default_factory=dict)
    distinct_reached: int = 0
    population_is_monotone: bool | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "igd_plus": {str(k): round(v, 8) for k, v in self.igd_plus.items()},
            "hypervolume_unnormalised": {
                str(k): round(v, 4) for k, v in self.hypervolume.items()
            },
            "distinct_reached": self.distinct_reached,
            "population_is_monotone": self.population_is_monotone,
        }


def at_budget(trace, budget: int) -> tuple[float, float] | None:
    """The trace's IGD+ and raw hypervolume once it had seen ``budget`` architectures.

    Indexed by distinct architectures rather than by position, because position is
    evaluations and the two diverge exactly where an algorithm re-queries. Returns None
    when the run never reached the budget, which is reported rather than extrapolated.
    """
    distinct = np.asarray(trace.distinct)
    reached = np.searchsorted(distinct, budget, side="left")
    if reached >= len(distinct):
        return None
    return float(trace.igd_plus[reached]), float(trace.hypervolume_unnormalised[reached])


def run_slate(
    table,
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...] = SEEDS,
    generations: int = 30,
    pop_size: int = 30,
    representation: str = "vector",
) -> tuple[list[RunResult], dict[str, str]]:
    """Run every algorithm at every seed. Returns the results and what could not run."""
    from .run import UnsupportedAlgorithm, run_algorithm

    results: list[RunResult] = []
    excluded: dict[str, str] = {}
    failed: dict[str, list[str]] = {}

    for name in algorithms:
        for seed in seeds:
            try:
                if name == "pareto-local-search":
                    from .local import pareto_local_search
                    from .scoring import score_trace

                    search = pareto_local_search(table, budget=max(BUDGETS) * 2, seed=seed)
                    points = table.matrix()
                    trace = score_trace(
                        search.objectives_so_far(), table.true_front(),
                        lower=points.min(axis=0), upper=points.max(axis=0),
                        algorithm=name, representation="table", seed=seed,
                    )
                elif name == "parego":
                    from .local import parego
                    from .scoring import score_trace

                    search = parego(table, budget=max(BUDGETS), seed=seed)
                    points = table.matrix()
                    trace = score_trace(
                        search.objectives_so_far(), table.true_front(),
                        lower=points.min(axis=0), upper=points.max(axis=0),
                        algorithm=name, representation="table", seed=seed,
                    )
                else:
                    # Generations are set so every algorithm can reach the largest budget,
                    # because a comparison at a fixed number of distinct architectures is
                    # only fair if each entry can get there. A steady-state algorithm
                    # evaluates one point per generation and a generational one evaluates
                    # a population, so a shared generation count is a budget of 30 for
                    # SMS-EMOA against 900 for NSGA-II — and the first run of this
                    # comparison reported SMS-EMOA at only the smallest budget for exactly
                    # that reason, which reads as the algorithm failing rather than as the
                    # harness starving it.
                    steady_state = name == "sms-emoa"
                    needed = (pop_size + max(BUDGETS) * 2 if steady_state
                              else generations)
                    trace = run_algorithm(
                        table, name, representation=representation, seed=seed,
                        generations=needed, pop_size=pop_size,
                    )
            except UnsupportedAlgorithm as exc:
                # The problem itself rules this algorithm out, so no seed will work.
                excluded[name] = str(exc)
                break
            except Exception as exc:  # noqa: BLE001
                # One seed, not the algorithm. Breaking here abandoned SMS-EMOA after its
                # third seed tripped the drift ceiling, and because blocks must be complete
                # that pulled every other algorithm down to two seeds — a Friedman test
                # with two blocks and six treatments cannot reject anything, and its
                # p-value of 0.81 reads exactly like evidence of no difference.
                failed.setdefault(name, []).append(
                    f"seed {seed}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                )
                continue

            result = RunResult(
                algorithm=name, seed=seed,
                distinct_reached=int(np.asarray(trace.distinct)[-1])
                if len(trace.distinct) else 0,
                population_is_monotone=trace.population_is_monotone,
            )
            for budget in BUDGETS:
                reading = at_budget(trace, budget)
                if reading is not None:
                    result.igd_plus[budget], result.hypervolume[budget] = reading
            results.append(result)

    for name, reasons in failed.items():
        excluded[f"{name} (some seeds)"] = "; ".join(reasons)
    return results, excluded


def measured_rope(results: list[RunResult], budget: int):
    """The noise floor, from how much each algorithm varies across its own seeds.

    Pooled across algorithms rather than taken from one, because a ROPE resting on a single
    algorithm's variance would assert that its noise is everyone's — the same argument the
    perplexity ROPE makes for pooling across architectures.
    """
    from ..statistics.rope import from_seed_matrix

    by_algorithm: dict[str, list[float]] = {}
    for result in results:
        if budget in result.igd_plus:
            by_algorithm.setdefault(result.algorithm, []).append(result.igd_plus[budget])

    usable = [values for values in by_algorithm.values() if len(values) >= 2]
    if len(usable) < 2:
        raise NotEnoughRuns(
            f"{len(usable)} algorithm(s) have two or more seeds at a budget of {budget} "
            "distinct architectures, which is too few to pool a noise estimate. Reporting "
            "the comparison as undecided is the honest outcome; inventing a width is not."
        )

    width = min(len(values) for values in usable)
    return from_seed_matrix(
        [values[:width] for values in usable],
        metric=f"IGD+ to the exact front at {budget} distinct architectures",
        source=(
            f"{len(usable)} algorithms x {width} seeds on the complete 1,099-architecture "
            "table, pooled within algorithm so real differences between them are removed"
        ),
    )


def compare(results: list[RunResult], budget: int):
    """Friedman with Holm, plus the Bayesian ranking, over seeds as blocks."""
    from ..statistics import Comparison, Declaration, plackett_luce
    from ..statistics.frequentist import friedman, posthoc

    algorithms = sorted({r.algorithm for r in results if budget in r.igd_plus})
    seeds = sorted({r.seed for r in results if budget in r.igd_plus})
    if len(algorithms) < 2 or len(seeds) < 2:
        raise NotEnoughRuns(
            f"{len(algorithms)} algorithm(s) over {len(seeds)} seed(s) at budget {budget}"
        )

    lookup = {(r.algorithm, r.seed): r.igd_plus[budget]
              for r in results if budget in r.igd_plus}
    # Only seeds every algorithm reached, so blocks are complete. A block missing one
    # treatment is not a block, and filling it would invent a measurement.
    complete = [s for s in seeds if all((a, s) in lookup for a in algorithms)]
    if len(complete) < 2:
        raise NotEnoughRuns(
            f"only {len(complete)} seed(s) were reached by every algorithm at budget "
            f"{budget}; a block missing a treatment is not a block"
        )

    rope = measured_rope(results, budget)
    declaration = Declaration(
        question=(
            f"At a budget of {budget} distinct architectures, how far from the exact "
            "Pareto front does each algorithm get?"
        ),
        treatments=tuple(algorithms),
        blocks=tuple(f"seed{s}" for s in complete),
        metric=f"IGD+ to the exact front at {budget} distinct architectures",
        higher_is_better=False,
        rope=rope.width,
        rope_source=rope.source,
        notes=(
            "The front is exact rather than estimated, because every architecture under "
            "the ceiling was trained and measured. The budget axis is distinct "
            "architectures rather than evaluations: a campaign pays to train an "
            "architecture once, and charging for repeats penalises algorithms that keep an "
            "archive over ones that forget."
        ),
    )
    comparison = Comparison(
        declaration,
        np.array([[lookup[(a, s)] for a in algorithms] for s in complete], dtype=float),
    )

    omnibus = friedman(comparison)

    # Blocks needed before a negative result means anything. Friedman's statistic is
    # compared against chi-square with k-1 degrees of freedom, and with b blocks the
    # largest attainable statistic is b*(k-1) — so below a threshold in b, no arrangement
    # of ranks can reach significance and the test cannot reject whatever the data say.
    # Reported because 15.6 asks for power by name, and because its absence let a
    # two-block comparison return p = 0.81 that read like evidence of equivalence.
    from scipy.stats import chi2

    k = len(algorithms)
    critical = float(chi2.ppf(0.95, k - 1))
    max_statistic = len(complete) * (k - 1)
    blocks_needed = int(np.ceil(critical / (k - 1)))

    # A negative omnibus licenses no pairwise test. Searching pairs after one has an error
    # rate nobody can state, and the module refuses rather than leaving that to discipline
    # — so the refusal is honoured here instead of being worked around. What it means is
    # reported instead: the Bayesian analysis separates "equivalent within the measured
    # ROPE" from "too few runs to tell", which a p-value cannot.
    pairwise: list[dict] | None = None
    if max_statistic < critical:
        # Not merely negative — incapable. Saying so is the difference between reporting a
        # result and reporting the absence of one.
        return {
            "declaration": declaration.as_dict(),
            "rope": rope.as_dict(),
            "underpowered": (
                f"{len(complete)} blocks against {k} treatments cannot reject at alpha "
                f"0.05 whatever the data show; {blocks_needed} are needed. No conclusion "
                "about these algorithms follows from this comparison, and the p-value "
                "below must not be read as evidence that they agree."
            ),
            "friedman": omnibus.as_dict(),
            "power": {
                "blocks": len(complete), "treatments": k,
                "blocks_needed_to_reject": blocks_needed, "can_reject_at_all": False,
            },
            "seeds_used": complete,
        }
    if omnibus.p_value < 0.05:
        pairwise = [entry.as_dict() for entry in posthoc(comparison, omnibus)]

    return {
        "declaration": declaration.as_dict(),
        "rope": rope.as_dict(),
        "mean_ranks": comparison.mean_ranks(),
        "friedman": omnibus.as_dict(),
        "power": {
            "blocks": len(complete),
            "treatments": k,
            "blocks_needed_to_reject": blocks_needed,
            "can_reject_at_all": max_statistic >= critical,
            "why": (
                "Friedman's statistic cannot exceed blocks x (treatments - 1), so below "
                f"{blocks_needed} blocks no arrangement of ranks reaches significance and "
                "a high p-value means the test had no power rather than that the "
                "treatments agree."
            ),
        },
        "posthoc": pairwise,
        "posthoc_withheld": None if pairwise is not None else (
            f"the omnibus found no difference (p = {omnibus.p_value:.4g}), so no pairwise "
            "comparison is licensed. The treatments are indistinguishable at this budget; "
            "whether that means equivalent or merely undecided is what the Bayesian "
            "ranking below answers."
        ),
        "probability_best": plackett_luce(comparison).probability_best,
        "seeds_used": complete,
    }
