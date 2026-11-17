"""Enumeration, the parameter ceiling, and the exact cost front.

The space is small enough to enumerate, and that is the design's central asset
rather than a limitation (design.md D19). Enumerating gives the **exact** cost
front, against which every predicted front and every search algorithm is later
measured. Nothing here needs a GPU, and all of it is bit-exact on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cost import kv_cache_bytes, parameter_count
from .genotype import REFERENCE_CONTEXT, Genotype, enumerate_space

#: Architectures above this parameter count are excluded from the sweep. The value
#: follows the constrained-deployment framing — roughly twice the anchor — and the
#: excluded fraction is recorded rather than passed over in silence.
PARAMETER_CEILING = 150_000_000


@dataclass(frozen=True)
class Enumeration:
    """The enumerated space, with the ceiling applied and the exclusion recorded."""

    genotypes: list[Genotype]
    objectives: np.ndarray  # shape (n, 2): params, kv_bytes — both minimised
    ceiling: int
    total_before_ceiling: int
    context: int

    @property
    def excluded(self) -> int:
        return self.total_before_ceiling - len(self.genotypes)

    @property
    def excluded_fraction(self) -> float:
        return self.excluded / self.total_before_ceiling

    def __len__(self) -> int:
        return len(self.genotypes)


def enumerate_with_ceiling(
    ceiling: int | None = PARAMETER_CEILING,
    context: int = REFERENCE_CONTEXT,
) -> Enumeration:
    """Enumerate the space and apply the parameter ceiling.

    Order is deterministic, inherited from :func:`enumerate_space`, so the result is
    reproducible bit for bit.
    """
    every = list(enumerate_space())
    params = np.array([parameter_count(g) for g in every], dtype=np.int64)

    keep = np.ones(len(every), dtype=bool) if ceiling is None else params <= ceiling
    kept = [g for g, k in zip(every, keep, strict=True) if k]

    objectives = np.column_stack(
        [
            np.array([parameter_count(g) for g in kept], dtype=np.int64),
            np.array([kv_cache_bytes(g, context=context) for g in kept], dtype=np.int64),
        ]
    )
    return Enumeration(
        genotypes=kept,
        objectives=objectives,
        ceiling=ceiling if ceiling is not None else int(params.max()),
        total_before_ceiling=len(every),
        context=context,
    )


def non_dominated(objectives: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated set, all objectives minimised.

    A point is dominated when another is no worse in every objective and strictly
    better in at least one. Ties are kept: two architectures with identical
    objectives are both on the front, because discarding one would silently drop a
    distinct architecture from the benchmark.
    """
    n = len(objectives)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        no_worse = np.all(objectives <= objectives[i], axis=1)
        strictly_better = np.any(objectives < objectives[i], axis=1)
        if np.any(no_worse & strictly_better):
            keep[i] = False
    return np.flatnonzero(keep)


def reference_point(objectives: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Hypervolume reference derived from the measured range, never fixed by hand.

    The prior work used ``[1e9, 0.0]`` and asserted it bounded the space; 12.3% of
    cost values exceeded it, so those individuals' hypervolume contributions were
    invalid. Deriving the point from the enumerated maxima makes that impossible.
    """
    return objectives.max(axis=0) * (1.0 + margin)


def reference_span(objectives: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fraction of each reference axis the space actually spans.

    An axis spanning almost none of its range contributes almost nothing to the
    hypervolume and cannot discriminate — which is what happened when a parameter
    axis bounded at 1e9 was compared against architectures of a few hundred thousand.
    """
    lo = objectives.min(axis=0)
    return (objectives.max(axis=0) - lo) / (reference - lo)
