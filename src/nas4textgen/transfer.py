"""Does the measured front hold outside the corpus it was measured on.

The benchmark is a census of one corpus. Every claim it supports — which architectures are on
the exact front, which design rules survive, whether depth beats width at matched size — is a
claim about WikiText-103 until something else is measured, and a paper that generalised from
one corpus would be making the error it was written to correct.

The transfer design answers a narrower question than a second census would, and says so. The
220 architectures of the measured front were retrained on three further corpora at three
seeds each; the rest of the space was not. So this module cannot report another corpus's
exact front — that needs all 1,099 architectures on that corpus — and does not pretend to.
What it can establish is whether the front *survives*: whether architectures selected on
WikiText-103 remain non-dominated elsewhere, whether their ordering is preserved, and whether
the rules read off the front still hold. A front that collapses on a different corpus is a
finding about WikiText-103; one that survives is a finding about the space.

**Perplexity is not comparable across corpora and is never compared across them here.**
Different text has different entropy, so a lower number on TinyStories than on WikiText-103
says nothing about either model. Every quantity below is computed within a corpus — a rank, a
dominance relation, a share — and only those within-corpus quantities are compared.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

#: The corpus the census and the search were run on.
SEARCH_CORPUS = "wikitext103"

#: The budget every comparable run was trained to.
TOKEN_BUDGET = 100_000_000

#: Design rules read off the search corpus, restated here to be tested elsewhere. Each maps
#: a genotype to whether it satisfies the rule.
DESIGN_RULES = {
    "kv <= 1": lambda g: g[4] <= 1,
    "n <= 6": lambda g: g[2] <= 6,
    "h <= 2": lambda g: g[0] <= 2,
}

#: Rules are tested against the quantity they were derived from: membership of the front.
#: An earlier version of this module scored them against the top quintile by perplexity
#: alone, which guarantees that a cost-efficiency rule fails — the architectures such a rule
#: selects are the small ones, and small models have worse perplexity by construction. The
#: result looked like a finding (`h <= 2` scoring a lift of exactly zero on all three
#: corpora) and was an artefact of asking a question the rules were never answers to.


class NotEnoughRuns(RuntimeError):
    """Raised when a corpus has too few measured architectures to say anything."""


@dataclass(frozen=True)
class Measured:
    """One architecture on one corpus, across its seeds."""

    genotype: tuple[int, ...]
    parameters: int
    perplexities: tuple[float, ...]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.perplexities)

    @property
    def spread(self) -> float:
        """Seed-to-seed standard deviation, or zero where there is only one seed."""
        return (statistics.stdev(self.perplexities)
                if len(self.perplexities) > 1 else 0.0)


@dataclass(frozen=True)
class RuleOutcome:
    """Whether a rule read off the search corpus still selects for quality elsewhere."""

    rule: str
    base_rate: float
    precision: float
    lift: float

    @property
    def holds(self) -> bool:
        return self.lift > 1.0

    def as_dict(self) -> dict[str, object]:
        return {"rule": self.rule, "base_rate": round(self.base_rate, 4),
                "precision": round(self.precision, 4), "lift": round(self.lift, 4),
                "holds": self.holds}


@dataclass(frozen=True)
class CorpusTransfer:
    """What became of the search corpus's front on one other corpus."""

    corpus: str
    architectures: int
    seeds: int
    rope: float
    kendall_tau: float
    kendall_p: float
    tau_top_decile: float
    retained: int
    top10_overlap: int
    rules: tuple[RuleOutcome, ...] = ()

    @property
    def retained_share(self) -> float:
        return self.retained / self.architectures if self.architectures else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "architectures": self.architectures,
            "seeds": self.seeds,
            "rope": round(self.rope, 6),
            "kendall_tau": round(self.kendall_tau, 4),
            "kendall_p": self.kendall_p,
            "kendall_tau_top_decile": round(self.tau_top_decile, 4),
            "still_non_dominated": self.retained,
            "still_non_dominated_share": round(self.retained_share, 4),
            "top10_overlap": self.top10_overlap,
            "rules": [r.as_dict() for r in self.rules],
        }


@dataclass(frozen=True)
class TransferReport:
    """Every corpus the front was retrained on, against the corpus it came from."""

    search_corpus: str
    front_size: int
    corpora: tuple[CorpusTransfer, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "search_corpus": self.search_corpus,
            "front_size": self.front_size,
            "token_budget": TOKEN_BUDGET,
            "corpora": [c.as_dict() for c in self.corpora],
            "what_this_can_and_cannot_say": (
                "Only the front was retrained, so no other corpus's exact front is known "
                "and none is claimed. Retention is the share of the search corpus's front "
                "that remains non-dominated among those same architectures on the new "
                "corpus; it would be 100% by construction if the ordering were identical, "
                "and it falls as the ordering diverges."
            ),
            "why_no_perplexity_is_compared_across_corpora": (
                "Different text has different entropy. Every quantity here is a rank, a "
                "dominance relation or a share, computed inside one corpus."
            ),
        }


def collect(manifests: Sequence[dict], corpus: str,
            token_budget: int = TOKEN_BUDGET) -> dict[tuple[int, ...], Measured]:
    """Group one corpus's runs by architecture, gathering their seeds."""
    from .space.cost import parameter_count
    from .space.genotype import Genotype

    grouped: dict[tuple[int, ...], list[float]] = {}
    for manifest in manifests:
        if manifest.get("corpus") != corpus:
            continue
        if manifest.get("token_budget") != token_budget:
            continue
        if not manifest.get("usable_for_benchmark", True):
            continue
        objectives = manifest.get("objectives") or {}
        perplexity = objectives.get("perplexity")
        if perplexity is None:
            continue
        genes = tuple(manifest["genotype"]["values"])
        grouped.setdefault(genes, []).append(float(perplexity))

    measured: dict[tuple[int, ...], Measured] = {}
    for genes, values in grouped.items():
        measured[genes] = Measured(
            genotype=genes,
            parameters=parameter_count(Genotype(*genes)),
            perplexities=tuple(sorted(values)),
        )
    return measured


def measured_rope(measured: Sequence[Measured]) -> float:
    """A region of practical equivalence, from seed replicates rather than chosen.

    Half the standard deviation of the replicates, pooled within architecture so that real
    differences between architectures are removed and what remains is the noise of running
    the same architecture again. Choosing this threshold after seeing the differences is how
    any gap is made to look meaningful, which is the practice this project refuses.
    """
    deviations = [m.spread for m in measured if len(m.perplexities) > 1]
    return 0.5 * statistics.fmean(deviations) if deviations else 0.0


def _non_dominated(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated points, minimising both coordinates."""
    keep = []
    for i, (a, b) in enumerate(points):
        if not any(c <= a and d <= b and (c < a or d < b)
                   for j, (c, d) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def compare(search: dict[tuple[int, ...], Measured],
            other: dict[tuple[int, ...], Measured],
            corpus: str) -> CorpusTransfer:
    """One corpus against the corpus the front was selected on."""
    from scipy.stats import kendalltau

    shared = sorted(set(search) & set(other))
    if len(shared) < 3:
        raise NotEnoughRuns(
            f"{corpus} shares {len(shared)} architectures with {SEARCH_CORPUS}; "
            "nothing can be said about transfer from that"
        )

    here = [other[g] for g in shared]
    baseline = [search[g].mean for g in shared]
    candidate = [m.mean for m in here]

    tau, p_value = kendalltau(baseline, candidate)

    # The top decile of the search corpus, which is the part a practitioner selects from.
    # A front that reorders only its poor members still transfers for every practical use.
    decile = max(3, len(shared) // 10)
    best = sorted(range(len(shared)), key=lambda i: baseline[i])[:decile]
    tau_decile, _ = kendalltau([baseline[i] for i in best], [candidate[i] for i in best])

    surviving = _non_dominated([(m.mean, float(m.parameters)) for m in here])
    retained = len(surviving)

    top10_search = {shared[i] for i in sorted(range(len(shared)),
                                              key=lambda i: baseline[i])[:10]}
    top10_other = {shared[i] for i in sorted(range(len(shared)),
                                             key=lambda i: candidate[i])[:10]}

    return CorpusTransfer(
        corpus=corpus,
        architectures=len(shared),
        seeds=max(len(m.perplexities) for m in here),
        rope=measured_rope(here),
        kendall_tau=float(tau),
        kendall_p=float(p_value),
        tau_top_decile=float(tau_decile) if tau_decile == tau_decile else 0.0,
        retained=retained,
        top10_overlap=len(top10_search & top10_other),
        rules=tuple(evaluate_rules(here, [here[i] for i in surviving])),
    )


def evaluate_rules(measured: Sequence[Measured],
                   retained: Sequence[Measured]) -> list[RuleOutcome]:
    """Whether each design rule still describes the architectures that stay on the front.

    The rules were derived as statements about front membership, so they are tested against
    front membership here: how much more often a retained architecture satisfies the rule
    than an architecture of the original front does. A lift near one means the rule has no
    power to say which members survive the change of corpus, which is a weaker claim than
    the rule being wrong and the one the data supports.
    """
    if not measured or not retained:
        return []

    outcomes = []
    for rule, satisfies in DESIGN_RULES.items():
        base = sum(satisfies(m.genotype) for m in measured) / len(measured)
        precision = sum(satisfies(m.genotype) for m in retained) / len(retained)
        outcomes.append(RuleOutcome(
            rule=rule,
            base_rate=base,
            precision=precision,
            lift=precision / base if base else 0.0,
        ))
    return outcomes


def analyse(manifests: Sequence[dict],
            corpora: Sequence[str] | None = None) -> TransferReport:
    """The whole transfer study, from the cached run manifests."""
    search = collect(manifests, SEARCH_CORPUS)
    if not search:
        raise NotEnoughRuns(f"no runs for {SEARCH_CORPUS}; sync the manifest cache")

    if corpora is None:
        corpora = sorted({m.get("corpus") for m in manifests
                          if m.get("corpus") and m.get("corpus") != SEARCH_CORPUS})

    results = []
    for corpus in corpora:
        other = collect(manifests, corpus)
        if not other:
            continue
        results.append(compare(search, other, corpus))

    front = max((r.architectures for r in results), default=0)
    return TransferReport(search_corpus=SEARCH_CORPUS, front_size=front,
                          corpora=tuple(results))
