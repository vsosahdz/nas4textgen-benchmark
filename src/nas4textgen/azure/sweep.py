"""The sweep, submitted in nested waves rather than all at once.

The canary asked whether the platform works and the pilot asked what it costs. The sweep is
the measurement, and the one thing it must not do is spend eleven days discovering something
a hundred and fifty runs would have shown.

**Waves nest, so wave one is a usable benchmark on its own.** The execution order is the
maximin sequence already computed and verified: each prefix is a space-filling sample of the
grid, and wave two contains wave one. That property is what makes stopping early cheap
rather than wasteful — a hundred and fifty completed architectures spread over the whole
space support a surrogate study and an interim front, while the first hundred and fifty in
enumeration order would all be small and support neither.

**Each wave is a decision point.** Between waves the completion rate, the failure reasons
and the cost to date are all known, and the next wave is submitted or it is not. Submitting
1,099 jobs at once removes that, and the failure it removes it for is the expensive kind:
an architecture class that reliably runs out of memory, a corpus path that is wrong for one
split, a limit that is too tight at the large end. Every one of those was found this way
already, at canary scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..space.genotype import GENES, Genotype

#: Nested wave sizes. Each is a usable sample; each contains the one before.
WAVE_SIZES = (150, 300, 600, 1000, 1099)


class ExecutionOrderMissing(RuntimeError):
    """Raised when the maximin order has not been computed and recorded."""


class OrderNotNested(RuntimeError):
    """Raised when the recorded waves do not contain one another."""


@dataclass(frozen=True)
class Wave:
    number: int
    size: int
    genotypes: tuple[Genotype, ...]
    #: Architectures this wave adds beyond the previous one. What actually gets submitted.
    new: tuple[Genotype, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "wave": self.number,
            "cumulative_size": self.size,
            "new_this_wave": len(self.new),
            "genotypes": [
                {gene: getattr(g, gene) for gene in GENES} for g in self.new
            ],
        }


def load_order(path: Path | None = None) -> list[int]:
    """The recorded maximin order, refusing to invent one.

    Read rather than recomputed. The order defines which architectures are measured first,
    so recomputing it at submission time would let a change in the sampling code silently
    reorder a campaign that is already half done — and the waves would stop nesting with
    the ones already run.
    """
    from ..artifacts import artifacts_root

    path = path or artifacts_root() / "space" / "sampling_designs.json"
    if not path.exists():
        raise ExecutionOrderMissing(
            f"no execution order at {path}. Run `nas4textgen designs` first: the order "
            "defines which architectures are measured first and is recorded so that a "
            "later change to the sampling code cannot reorder a campaign in flight."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["maximin_order"])


def waves(order: list[int], space: list[Genotype],
          sizes: tuple[int, ...] = WAVE_SIZES) -> list[Wave]:
    """Split the order into nested waves, checking that they really do nest."""
    if len(order) != len(space):
        raise ExecutionOrderMissing(
            f"the order names {len(order)} architectures and the space holds {len(space)}. "
            "One of them is from a different revision of the search space."
        )

    built: list[Wave] = []
    previous: set[int] = set()
    for number, size in enumerate(sizes, start=1):
        indices = order[:size]
        current = set(indices)
        if not previous <= current:
            raise OrderNotNested(
                f"wave {number} does not contain wave {number - 1}. Nesting is what makes "
                "an interrupted sweep a usable sample rather than a partial one."
            )
        built.append(Wave(
            number=number,
            size=size,
            genotypes=tuple(space[i] for i in indices),
            new=tuple(space[i] for i in indices if i not in previous),
        ))
        previous = current
    return built


@dataclass(frozen=True)
class WaveCost:
    wave: int
    new_runs: int
    node_hours: float
    usd: float
    days_at_nodes: float

    def as_dict(self) -> dict[str, object]:
        return {
            "wave": self.wave,
            "new_runs": self.new_runs,
            "node_hours": round(self.node_hours, 1),
            "usd": round(self.usd, 2),
            "days": round(self.days_at_nodes, 2),
        }


#: Measured price, from the Azure retail prices API rather than assumed.
LOW_PRIORITY_USD_PER_HOUR = 0.1260


def cost(wave: Wave, model, token_budget: int, nodes: int) -> WaveCost:
    """What this wave's new architectures will cost, from the fitted throughput model."""
    seconds = np.array([model.seconds_for(g, token_budget) for g in wave.new])
    hours = float(seconds.sum() / 3600)
    return WaveCost(
        wave=wave.number,
        new_runs=len(wave.new),
        node_hours=hours,
        usd=hours * LOW_PRIORITY_USD_PER_HOUR,
        days_at_nodes=hours / max(nodes, 1) / 24,
    )
