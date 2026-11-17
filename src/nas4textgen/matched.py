"""Does the shape conclusion survive the corpus and the seed it was measured on.

Section 4.3 reports that within bands of equal parameter count the widest third of the space
reaches lower perplexity than the deepest third, on every band, on WikiText-103, at one seed.
That is a claim about shape rather than size — banding is what removes the confound, since
depth adds only block parameters while the embedding dominates the total — but it is still
one corpus and one draw.

This module repeats the comparison on every corpus the front was retrained on and at every
seed, and bands by the second cost objective as well as the first. A conclusion that holds on
one corpus at one seed under one definition of cost is weaker than it sounds, and saying so
costs nothing now that the runs exist.

**What this cannot do.** Only the 220 front architectures were retrained elsewhere, so the
comparison off the search corpus runs inside a set selected for being Pareto-optimal. A front
is not a sample of the space: it is biased toward the shapes that win, which is precisely the
quantity under test. The result below is therefore about whether the ordering holds *among
good architectures* on other corpora, and is reported that way rather than as a repeat of the
census result.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

#: The corpus the census covers in full; everywhere else only the front was retrained.
SEARCH_CORPUS = "wikitext103"

#: Fewer bands than the census analysis uses. Ten bands over 220 architectures leaves 22 per
#: band and seven per extreme third, which is too thin to read; five leaves 44 and fourteen.
FRONT_BANDS = 5

#: The two definitions of cost this paper searches under. Reported separately because they
#: disagree — the structural gate accepts the space only because their rank correlation is
#: +0.615 — so "matched cost" is two different matchings and a conclusion can hold under one.
COST_KEYS = ("parameters", "kv_cache_bytes")


class NotEnoughRuns(RuntimeError):
    """Raised when a corpus and seed have too few architectures to band."""


@dataclass(frozen=True)
class BandedComparison:
    """One corpus, one seed, one definition of cost."""

    corpus: str
    seed: int | None
    cost: str
    bands: int
    architectures: int
    deep_means: tuple[float, ...]
    wide_means: tuple[float, ...]

    @property
    def deltas(self) -> tuple[float, ...]:
        return tuple(d - w for d, w in zip(self.deep_means, self.wide_means, strict=True))

    @property
    def bands_won_by_width(self) -> int:
        """Bands where the widest third reached lower perplexity than the deepest third."""
        return sum(1 for delta in self.deltas if delta > 0)

    @property
    def unanimous(self) -> bool:
        return self.bands_won_by_width == self.bands

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "seed": self.seed,
            "cost": self.cost,
            "bands": self.bands,
            "architectures": self.architectures,
            "deep_mean_perplexity": [round(v, 3) for v in self.deep_means],
            "wide_mean_perplexity": [round(v, 3) for v in self.wide_means],
            "delta_deep_minus_wide": [round(v, 3) for v in self.deltas],
            "bands_won_by_width": self.bands_won_by_width,
            "unanimous": self.unanimous,
        }


@dataclass(frozen=True)
class MatchedReport:
    """Every comparison, and what they agree on."""

    comparisons: tuple[BandedComparison, ...] = field(default_factory=tuple)

    @property
    def unanimous_everywhere(self) -> bool:
        return all(c.unanimous for c in self.comparisons)

    def as_dict(self) -> dict[str, object]:
        by_cost: dict[str, list[int]] = {}
        for comparison in self.comparisons:
            by_cost.setdefault(comparison.cost, []).append(comparison.bands_won_by_width)
        return {
            "bands": FRONT_BANDS,
            "cost_definitions": list(COST_KEYS),
            "comparisons": [c.as_dict() for c in self.comparisons],
            "bands_won_by_width": {
                cost: {"total": sum(values),
                       "of": sum(c.bands for c in self.comparisons if c.cost == cost)}
                for cost, values in by_cost.items()
            },
            "unanimous_everywhere": self.unanimous_everywhere,
            "what_this_runs_over": (
                "The 220 architectures of the measured front, which is the only set "
                "retrained off the search corpus. A front is selected for being "
                "Pareto-optimal, so this asks whether the ordering holds among good "
                "architectures elsewhere, not whether a second census would reproduce it."
            ),
            "why_two_definitions_of_cost": (
                "Parameter count and key-value cache size rank the space differently — the "
                "structural gate accepts the space only because their rank correlation is "
                "+0.615 — so matching on one is not matching on the other, and a shape "
                "conclusion can hold under one definition and fail under the other."
            ),
        }


def _band(values: Sequence[float], perplexity: Sequence[float], shape: Sequence[float],
          bands: int, extreme: float = 1 / 3) -> tuple[list[float], list[float]]:
    """Mean perplexity of the deepest and widest extremes inside each band of equal cost."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    per_band = max(1, len(order) // bands)
    deep_means, wide_means = [], []
    for index in range(bands):
        start = index * per_band
        stop = len(order) if index == bands - 1 else (index + 1) * per_band
        members = order[start:stop]
        if len(members) < 3:
            continue
        by_shape = sorted(members, key=lambda i: shape[i])
        cut = max(1, int(len(by_shape) * extreme))
        # Low ratio is wide, high ratio is deep: the shape key is depth over width.
        wide_means.append(statistics.fmean(perplexity[i] for i in by_shape[:cut]))
        deep_means.append(statistics.fmean(perplexity[i] for i in by_shape[-cut:]))
    return deep_means, wide_means


def compare(rows: Sequence[dict], corpus: str, seed: int | None,
            cost: str, bands: int = FRONT_BANDS) -> BandedComparison:
    """One banded comparison. ``rows`` carry ``parameters``, ``kv_cache_bytes``,
    ``perplexity`` and ``shape``."""
    if len(rows) < bands * 3:
        raise NotEnoughRuns(
            f"{corpus} seed {seed}: {len(rows)} architectures cannot fill {bands} bands")
    deep, wide = _band([r[cost] for r in rows], [r["perplexity"] for r in rows],
                       [r["shape"] for r in rows], bands)
    return BandedComparison(
        corpus=corpus, seed=seed, cost=cost, bands=len(deep),
        architectures=len(rows),
        deep_means=tuple(deep), wide_means=tuple(wide))


def gather(manifests: Sequence[dict], corpus: str,
           seed: int | None = None) -> list[dict]:
    """Rows for one corpus, optionally one seed, with both costs and the shape ratio."""
    from .space.cost import kv_cache_bytes, parameter_count
    from .space.genotype import Genotype

    by_architecture: dict[tuple[int, ...], list[float]] = {}
    for manifest in manifests:
        if manifest.get("corpus") != corpus:
            continue
        if seed is not None and manifest.get("config", {}).get("seed") != seed:
            continue
        perplexity = (manifest.get("objectives") or {}).get("perplexity")
        if perplexity is None:
            continue
        genes = tuple(manifest["genotype"]["values"])
        by_architecture.setdefault(genes, []).append(float(perplexity))

    rows = []
    for genes, values in by_architecture.items():
        genotype = Genotype(*genes)
        rows.append({
            "genotype": genes,
            "parameters": float(parameter_count(genotype)),
            "kv_cache_bytes": float(kv_cache_bytes(genotype)),
            # Depth over width: the ratio that orders architectures by shape at a size.
            "shape": genotype.n / genotype.e,
            "perplexity": statistics.fmean(values),
        })
    return rows


def analyse(manifests: Sequence[dict], corpora: Sequence[str] | None = None,
            bands: int = FRONT_BANDS) -> MatchedReport:
    """Every corpus, every seed it was run at, under both definitions of cost."""
    if corpora is None:
        corpora = sorted({m.get("corpus") for m in manifests if m.get("corpus")})

    comparisons = []

    # The control, and it is not optional. Off the search corpus only the front was
    # retrained, and a front is selected for Pareto-optimality in perplexity against
    # parameters — so within it the parameter-quality trade-off is largely spent and shape
    # has little left to explain. Running the search corpus restricted to the same 220
    # separates that selection effect from a corpus effect. Without this row the drop from
    # five bands to two reads as "the conclusion does not transfer", which is false.
    front = {tuple(m["genotype"]["values"]) for m in manifests
             if m.get("corpus") and m.get("corpus") != SEARCH_CORPUS}
    if front:
        rows = [r for r in gather(manifests, SEARCH_CORPUS) if r["genotype"] in front]
        for cost in COST_KEYS:
            try:
                comparisons.append(
                    replace(compare(rows, SEARCH_CORPUS, None, cost, bands),
                            corpus=f"{SEARCH_CORPUS} (front only)"))
            except NotEnoughRuns:
                continue

    for corpus in corpora:
        seeds = sorted({m.get("config", {}).get("seed")
                        for m in manifests if m.get("corpus") == corpus})
        # The pooled row first, then each seed, so a conclusion that only holds on average
        # is visible beside the draws it was averaged from.
        for seed in [None, *seeds]:
            rows = gather(manifests, corpus, seed)
            for cost in COST_KEYS:
                try:
                    comparisons.append(compare(rows, corpus, seed, cost, bands))
                except NotEnoughRuns:
                    continue
    return MatchedReport(comparisons=tuple(comparisons))
