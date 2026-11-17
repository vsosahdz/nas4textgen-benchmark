"""Sampling designs: the sweep's execution order, and the designs it is compared to.

The sweep runs in nested waves whose union is eventually the whole space, ordered so
that every completed prefix is space-filling. Two consequences: an interrupted sweep
yields a valid sample rather than an arbitrary subset, and the waves double as the
nested subsets the surrogate-budget study needs — so that study costs nothing extra.

**Why maximin and not Latin hypercube.** Latin hypercube sampling is built for
continuous boxes. Here the genes are categorical or ordinal with as few as three
levels, so stratifying a dimension into more bins than it has levels is vacuous. More
decisively, standard LHS is not extensible: an LHS of 300 does not contain an LHS of
150, so the waves would not nest. A sequential maximin design over the enumerated grid
nests by construction and needs no approximation.

LHS and uniform random are kept as **comparison designs**, not discarded. With the
complete table as ground truth, surrogate quality per unit of sampling budget becomes
measurable across designs — which tests the sampling assumption the prior work made
rather than inheriting it.
"""

from __future__ import annotations

import numpy as np

from .genotype import GENES, LEVELS, Genotype, encode


def _normalised(genotypes: list[Genotype]) -> np.ndarray:
    """Genotypes as points in the unit hypercube, one row each."""
    return np.array([encode(g) for g in genotypes], dtype=np.float64)


def maximin_order(genotypes: list[Genotype], seed: int = 0) -> np.ndarray:
    """A nested, space-filling ordering of the whole space.

    Greedy farthest-point traversal: start from the point closest to the centre of the
    unit hypercube, then repeatedly take whichever remaining point is farthest from
    everything already chosen. Every prefix of the result is a maximin design, so the
    ordering is extensible by construction — the property Latin hypercube sampling
    lacks.

    Deterministic given ``seed``, which only breaks ties.
    """
    points = _normalised(genotypes)
    n = len(points)
    rng = np.random.default_rng(seed)

    centre = np.full(points.shape[1], 0.5)
    order = [int(np.argmin(((points - centre) ** 2).sum(axis=1)))]
    remaining = np.ones(n, dtype=bool)
    remaining[order[0]] = False

    # Distance from every remaining point to the nearest already-selected point.
    nearest = np.linalg.norm(points - points[order[0]], axis=1)

    while remaining.any():
        candidates = np.flatnonzero(remaining)
        distances = nearest[candidates]
        best = distances.max()
        tied = candidates[distances >= best - 1e-12]
        chosen = int(tied[0]) if len(tied) == 1 else int(rng.choice(tied))
        order.append(chosen)
        remaining[chosen] = False
        nearest = np.minimum(nearest, np.linalg.norm(points - points[chosen], axis=1))

    return np.array(order, dtype=np.int64)


def waves(order: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    """Split an ordering into cumulative waves.

    Wave ``k + 1`` contains every architecture of wave ``k``, which is what makes the
    waves usable as the nested subsets of the surrogate-budget study.
    """
    out = []
    for size in sizes:
        out.append(order[: min(size, len(order))])
    return out


def latin_hypercube_design(
    genotypes: list[Genotype], size: int, seed: int = 0
) -> tuple[np.ndarray, dict[str, str]]:
    """A Latin hypercube design mapped onto the discrete grid, for comparison.

    Returns the selected indices and a record of how stratification was resolved,
    because the honest answer differs per gene: a gene with three levels cannot be
    stratified into more than three strata, so its strata are recycled and the design
    degenerates towards uniform sampling in that dimension. Recording this is required
    by the benchmark-artifact spec — the limitation belongs in the artifact, not in a
    footnote nobody reads.
    """
    rng = np.random.default_rng(seed)
    notes: dict[str, str] = {}

    columns = []
    for gene in GENES:
        levels = len(LEVELS[gene])
        if levels >= size:
            picks = rng.permutation(levels)[:size]
            notes[gene] = f"{levels} levels >= {size} samples: proper stratification"
        else:
            repeats = int(np.ceil(size / levels))
            picks = np.concatenate([rng.permutation(levels) for _ in range(repeats)])[:size]
            notes[gene] = (
                f"{levels} levels < {size} samples: strata recycled {repeats}x, "
                "so this dimension degenerates towards uniform sampling"
            )
        columns.append(picks / max(levels - 1, 1))

    requested = np.column_stack(columns)

    # Map each requested point to the nearest architecture that actually exists. LHS
    # proposes points in a continuous box; the space is a grid, so the design can only
    # be realised approximately.
    points = _normalised(genotypes)
    chosen: list[int] = []
    for row in requested:
        distances = np.linalg.norm(points - row, axis=1)
        for candidate in np.argsort(distances):
            if int(candidate) not in chosen:
                chosen.append(int(candidate))
                break
    notes["_realisation"] = (
        "each LHS point mapped to the nearest existing architecture, without "
        "replacement; the space is a grid, so the design is approximate"
    )
    return np.array(chosen, dtype=np.int64), notes


def uniform_random_design(
    genotypes: list[Genotype], size: int, seed: int = 0
) -> np.ndarray:
    """Uniform sampling without replacement — the floor every design must beat."""
    rng = np.random.default_rng(seed)
    return rng.permutation(len(genotypes))[:size].astype(np.int64)


def fill_distance(genotypes: list[Genotype], selected: np.ndarray) -> float:
    """Worst-case distance from any architecture to the nearest selected one.

    The quantity a space-filling design minimises, and the one that makes "this prefix
    is space-filling" a measurement rather than an assertion.
    """
    points = _normalised(genotypes)
    if len(selected) == 0:
        return float("inf")
    chosen = points[selected]
    return float(
        max(float(np.linalg.norm(chosen - p, axis=1).min()) for p in points)
    )
