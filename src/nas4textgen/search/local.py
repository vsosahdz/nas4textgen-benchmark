"""Two algorithms pymoo does not carry, implemented against the lookup table.

**Pareto local search.** The natural method for a discrete grid and absent from most NAS
comparisons, which reach for population methods designed for continuous spaces. It keeps an
archive, repeatedly picks an unexplored member, evaluates its neighbours — the architectures
one gene level away — and adds any that are not dominated. On 1,099 points with five ordinal
genes the neighbourhood is small and exact, so the method has no operators to tune and no
representation to argue about, which makes it the cleanest baseline in the slate.

**ParEGO.** The surrogate-assisted entry, and the one the campaign's own budget curve makes
interesting. It scalarises the objectives with a random augmented Tchebycheff weight each
iteration, fits a model to the scalarised values, and evaluates whichever unevaluated
architecture the model rates best. Whether that beats spending the same evaluations on a
population method is precisely the question a benchmark with a known front can answer, and
the surrogate study already found that a model needs about a hundred labels here before it
outranks sorting by parameter count — so ParEGO's early iterations are expected to be poor
and the interesting part is where it overtakes.

Both work directly on the enumerated table rather than through pymoo's operators. That is
not a shortcut: the table *is* the search space, every neighbour is a real architecture with
a measured value, and going through a continuous encoding would reintroduce the rounding
artefact the representation study measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..space.genotype import GENES, LEVELS, Genotype

#: Weight vectors ParEGO draws from. Fixed rather than continuous so a run is reproducible
#: from its seed alone, and enough of them that the front is covered.
PAREGO_WEIGHTS = 21

#: The augmented Tchebycheff term. Standard, and it is what stops the scalarisation from
#: accepting weakly dominated points that the pure Tchebycheff form admits.
PAREGO_RHO = 0.05


@dataclass
class TableSearch:
    """Shared bookkeeping: evaluate by lookup, count every query, never repeat one."""

    table: object  # BenchmarkTable
    queries: list[tuple[int, ...]] = field(default_factory=list)
    _seen: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)

    def evaluate(self, genotype: Genotype) -> np.ndarray | None:
        """Objectives for one architecture, or None if it is not in the table.

        A repeat costs nothing and is not counted: on a lookup benchmark, re-asking about an
        architecture already measured is free in reality too, and charging for it would
        penalise algorithms that keep an archive over those that forget.
        """
        key = genotype.as_tuple()
        if key in self._seen:
            return self._seen[key]
        if key not in self.table.rows:
            return None
        value = self.table.rows[key]
        self._seen[key] = value
        self.queries.append(key)
        return value

    @property
    def spent(self) -> int:
        return len(self.queries)

    def objectives_so_far(self) -> np.ndarray:
        return (
            np.vstack([self._seen[k] for k in self.queries])
            if self.queries else np.empty((0, len(self.table.objective_names)))
        )


def neighbours(genotype: Genotype) -> list[Genotype]:
    """Architectures one gene level away, in either direction.

    Ordinal steps rather than arbitrary substitutions: the gene levels are ordered, so a
    neighbour is the next level up or down, and a move is a small architectural change
    rather than a jump across the space. Infeasible results — kv not dividing h — are
    dropped here rather than evaluated, because on a discrete grid the neighbourhood is
    cheap to enumerate and a search should not spend queries on points it can see are
    invalid.
    """
    found = []
    for gene in GENES:
        levels = LEVELS[gene]
        current = levels.index(getattr(genotype, gene))
        for step in (-1, 1):
            position = current + step
            if not 0 <= position < len(levels):
                continue
            candidate = Genotype(**{
                **{g: getattr(genotype, g) for g in GENES},
                gene: levels[position],
            })
            if candidate.is_valid():
                found.append(candidate)
    return found


def pareto_local_search(table, budget: int, seed: int = 0) -> TableSearch:
    """Explore neighbourhoods of non-dominated points until the budget runs out.

    Restarts from a fresh random point when the archive is fully explored, which is what
    stops it converging on one basin and then spending the rest of the budget confirming it.
    """
    from ..space.enumerate import non_dominated

    rng = np.random.default_rng(seed)
    search = TableSearch(table)
    space = table.genotypes()

    archive: dict[tuple[int, ...], np.ndarray] = {}
    unexplored: list[Genotype] = []
    # Every point whose neighbourhood has been enumerated. Without this the search
    # re-queues architectures already in the archive, and because a repeat lookup does not
    # advance the budget the loop never terminates — the first version hung on exactly that.
    explored: set[tuple[int, ...]] = set()
    stalled = 0

    while search.spent < budget:
        if not unexplored:
            start = space[rng.integers(len(space))]
            if start.as_tuple() in explored:
                # Restarting onto an exhausted point buys nothing. Give up after enough
                # consecutive misses rather than spinning: on a grid this small the archive
                # can genuinely cover everything reachable.
                stalled += 1
                if stalled > 50:
                    break
                continue
            stalled = 0
            if search.evaluate(start) is None:
                continue
            archive[start.as_tuple()] = search._seen[start.as_tuple()]
            unexplored.append(start)

        current = unexplored.pop()
        if current.as_tuple() in explored:
            continue
        explored.add(current.as_tuple())

        for neighbour in neighbours(current):
            if search.spent >= budget:
                break
            value = search.evaluate(neighbour)
            if value is None or neighbour.as_tuple() in explored:
                continue

            candidates = np.vstack([*archive.values(), value])
            keys = [*archive.keys(), neighbour.as_tuple()]
            keep = set(non_dominated(candidates).tolist())
            if len(candidates) - 1 in keep:
                archive = {keys[i]: candidates[i] for i in sorted(keep)}
                unexplored.append(neighbour)

    return search


def parego(table, budget: int, seed: int = 0, initial: int = 10) -> TableSearch:
    """Scalarise with a random weight, fit a model, evaluate its best guess.

    One evaluation per iteration, which is what makes it a surrogate-assisted method rather
    than a population one: the model is refitted from everything measured so far and asked
    for a single next architecture. Expensive per iteration and cheap in evaluations, which
    is the trade a real NAS campaign wants.
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    search = TableSearch(table)
    space = table.genotypes()
    features = np.array(
        [[getattr(g, gene) for gene in GENES] for g in space], dtype=float
    )

    # A Latin-ish start: random architectures, because a surrogate fitted on nothing has no
    # opinion and the first evaluations have to come from somewhere.
    for index in rng.permutation(len(space))[:initial]:
        if search.spent >= budget:
            return search
        search.evaluate(space[index])

    weights = np.linspace(0, 1, PAREGO_WEIGHTS)
    while search.spent < budget:
        measured = search.objectives_so_far()
        lower, upper = measured.min(axis=0), measured.max(axis=0)
        span = np.where(upper - lower == 0, 1.0, upper - lower)
        normalised = (measured - lower) / span

        # Augmented Tchebycheff, with a fresh weight each iteration so successive fits chase
        # different parts of the front rather than converging on one region.
        w = np.array([weights[rng.integers(len(weights))]])
        weight = np.concatenate([w, 1.0 - w])
        if measured.shape[1] > 2:
            weight = rng.dirichlet(np.ones(measured.shape[1]))
        scalar = (normalised * weight).max(axis=1) + PAREGO_RHO * (normalised * weight).sum(axis=1)

        evaluated = {k for k in search.queries}
        pending = [i for i, g in enumerate(space) if g.as_tuple() not in evaluated]
        if not pending:
            break

        # Small and shallow on purpose. The model is refitted once per evaluation, so a
        # forest sized for accuracy would make the algorithm's wall clock a property of
        # this implementation rather than of the method — and what is being compared is
        # evaluations spent, not seconds.
        model = RandomForestRegressor(
            30, max_depth=8, random_state=seed, n_jobs=-1
        )
        model.fit(
            np.array([[getattr(space_g, gene) for gene in GENES]
                      for space_g in (Genotype(**dict(zip(GENES, k, strict=True)))
                                      for k in search.queries)], dtype=float),
            scalar,
        )
        predicted = model.predict(features[pending])
        search.evaluate(space[pending[int(np.argmin(predicted))]])

    return search
