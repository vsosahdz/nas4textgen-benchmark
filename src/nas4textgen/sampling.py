"""Which sampling design to spend a training budget on, answered against the whole space.

The practical question a reader of this benchmark faces is not which algorithm to search
with — it is blunter. *I can afford a hundred and fifty architectures. How should I pick
them?* Before the census that question could only be answered by argument, because the
front the sample was being compared against was itself a sample. Now the front is exact,
so a design's loss is the gap between the front its subset recovers and the front that
exists.

**Uniform random is the floor and is reported as the floor.** The campaign used a maximin
ordering, so a comparison that showed maximin winning without saying what random achieves
would be an author grading their own choice. If random matches maximin, the campaign's
design decision did not matter and that is worth knowing; the benchmark is not diminished
by it, and the reader with a hundred and fifty runs to spend is better served.

**Two losses, because they answer different questions.** IGD+ says how close the recovered
front is to the real one in objective space — the quantity that matters if the goal is a
good trade-off curve. Front recall says what share of the exactly-known Pareto set the
subset actually contains — the quantity that matters if the goal is to find particular
architectures. A design can do well on one and badly on the other.

**Every design is stochastic here, including maximin.** Its greedy traversal starts from a
seeded choice, so it varies run to run, and a single draw of any of the three would be a
statement about that draw. The comparison blocks on seeds for the same reason the algorithm
comparison does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Budgets compared, matching the sweep's own wave sizes so the numbers answer the question
#: the campaign actually faced rather than a rounder one.
BUDGETS = (150, 300, 600)

#: Seeds per design per budget. Enough blocks that a moderate difference is detectable —
#: the same power argument as the algorithm comparison, and cheap for the same reason:
#: selecting a subset of a table costs no training.
SEEDS = tuple(range(20))


@dataclass(frozen=True)
class DesignResult:
    """One design, one budget, one seed."""

    design: str
    budget: int
    seed: int
    #: Gap from the subset's front to the exact front, in normalised objective space.
    igd_plus: float
    #: Share of the exact Pareto set the subset contains.
    front_recall: float
    #: Worst-case distance from any architecture to the nearest sampled one. What a
    #: space-filling design minimises, and the property maximin is chosen for — reported so
    #: a design that fills the space well but recovers the front badly is visible as such.
    fill_distance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "design": self.design,
            "budget": self.budget,
            "seed": self.seed,
            "igd_plus": round(self.igd_plus, 8),
            "front_recall": round(self.front_recall, 6),
            "fill_distance": round(self.fill_distance, 6),
        }


@dataclass
class SamplingComparison:
    """Every design at every budget."""

    results: list[DesignResult] = field(default_factory=list)
    space_size: int = 0
    front_size: int = 0

    def medians(self, measure: str) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for budget in sorted({r.budget for r in self.results}):
            out[budget] = {}
            for design in sorted({r.design for r in self.results}):
                values = [getattr(r, measure) for r in self.results
                          if r.budget == budget and r.design == design]
                if values:
                    out[budget][design] = float(np.median(values))
        return out


def evaluate_design(
    indices: np.ndarray,
    objectives: np.ndarray,
    exact_front: set[int],
    genotypes: list,
) -> tuple[float, float, float]:
    """IGD+, front recall and fill distance for one sampled subset."""
    from .search.scoring import igd_plus, normalise
    from .space.designs import fill_distance as _fill
    from .space.enumerate import non_dominated

    lower, upper = objectives.min(axis=0), objectives.max(axis=0)
    subset = objectives[indices]
    local_front = indices[non_dominated(subset)]

    gap = float(igd_plus(
        normalise(objectives[local_front], lower, upper),
        normalise(objectives[sorted(exact_front)], lower, upper),
    ))
    recall = len(set(local_front.tolist()) & exact_front) / len(exact_front)
    return gap, recall, float(_fill(genotypes, indices))


def compare_designs(
    genotypes: list,
    objectives: np.ndarray,
    budgets: tuple[int, ...] = BUDGETS,
    seeds: tuple[int, ...] = SEEDS,
) -> SamplingComparison:
    """Run every design at every budget and seed against the exact front."""
    from .space.designs import (
        latin_hypercube_design,
        maximin_order,
        uniform_random_design,
    )
    from .space.enumerate import non_dominated

    exact_front = set(non_dominated(objectives).tolist())
    comparison = SamplingComparison(
        space_size=len(genotypes), front_size=len(exact_front)
    )

    for seed in seeds:
        # The maximin order is computed once per seed and sliced, because every prefix of
        # it is itself a maximin design — that extensibility is the property the campaign
        # chose it for, and recomputing per budget would measure a different design.
        order = maximin_order(genotypes, seed=seed)
        for budget in budgets:
            for design, indices in (
                ("maximin", order[:budget]),
                ("latin_hypercube", latin_hypercube_design(genotypes, budget, seed=seed)[0]),
                ("uniform_random", uniform_random_design(genotypes, budget, seed=seed)),
            ):
                gap, recall, fill = evaluate_design(
                    np.asarray(indices), objectives, exact_front, genotypes
                )
                comparison.results.append(DesignResult(
                    design=design, budget=budget, seed=seed,
                    igd_plus=gap, front_recall=recall, fill_distance=fill,
                ))
    return comparison


def test_designs(comparison: SamplingComparison, budget: int, measure: str) -> dict:
    """Friedman with Holm over seeds, plus the Bayesian ranking, for one budget.

    ``measure`` is ``igd_plus`` or ``front_recall``, and the direction differs between them
    — lower is better for a distance and higher for a share — which is exactly the sort of
    thing a test cannot infer and a declaration has to state.
    """
    from .statistics import Comparison, Declaration, plackett_luce
    from .statistics.frequentist import friedman, posthoc
    from .statistics.rope import from_seed_matrix

    designs = sorted({r.design for r in comparison.results})
    seeds = sorted({r.seed for r in comparison.results if r.budget == budget})
    lookup = {
        (r.design, r.seed): getattr(r, measure)
        for r in comparison.results if r.budget == budget
    }
    complete = [s for s in seeds if all((d, s) in lookup for d in designs)]

    rope = from_seed_matrix(
        [[lookup[(d, s)] for s in complete] for d in designs],
        metric=measure,
        source=(
            f"{len(designs)} designs x {len(complete)} seeds at a budget of {budget} "
            "architectures, pooled within design so real differences between them are "
            "removed"
        ),
    )
    declaration = Declaration(
        question=(
            f"With {budget} architectures to train, which sampling design recovers the "
            f"exact Pareto front best, measured by {measure}?"
        ),
        treatments=tuple(designs),
        blocks=tuple(f"seed{s}" for s in complete),
        metric=measure,
        higher_is_better=measure == "front_recall",
        rope=rope.width,
        rope_source=rope.source,
        notes=(
            "The front is exact because every architecture under the ceiling was trained, "
            "so this is a loss against ground truth rather than a comparison between "
            "samples. Uniform random is the floor: the campaign used maximin, and a "
            "comparison that omitted the floor would be an author grading their own choice."
        ),
    )
    matrix = np.array([[lookup[(d, s)] for d in designs] for s in complete], dtype=float)
    result = Comparison(declaration, matrix)

    omnibus = friedman(result)
    from scipy.stats import chi2

    k = len(designs)
    critical = float(chi2.ppf(0.95, k - 1))
    if len(complete) * (k - 1) < critical:
        return {
            "declaration": declaration.as_dict(),
            "underpowered": (
                f"{len(complete)} blocks against {k} designs cannot reject at alpha 0.05 "
                "whatever the data show"
            ),
            "friedman": omnibus.as_dict(),
        }

    return {
        "declaration": declaration.as_dict(),
        "rope": rope.as_dict(),
        "mean_ranks": result.mean_ranks(),
        "friedman": omnibus.as_dict(),
        "power": {"blocks": len(complete), "treatments": k,
                  "blocks_needed_to_reject": int(np.ceil(critical / (k - 1)))},
        "posthoc": ([e.as_dict() for e in posthoc(result, omnibus)]
                    if omnibus.p_value < 0.05 else None),
        "posthoc_withheld": None if omnibus.p_value < 0.05 else (
            f"the omnibus found no difference (p = {omnibus.p_value:.4g}), so no pairwise "
            "comparison is licensed"
        ),
        "probability_best": plackett_luce(result).probability_best,
    }
