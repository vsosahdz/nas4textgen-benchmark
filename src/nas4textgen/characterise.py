"""What the measured table says about which architectures are good, and why.

Four questions, each one a claim someone has made or would make.

**Does depth beat width at a fixed parameter budget?** The prior work says deep and thin
wins. That is a testable claim and it is tested here rather than repeated: within bands of
equal parameter count, the deepest third against the widest third, through the declared
statistical protocol with the measured ROPE.

**What does grouped-query attention cost?** The KV cache is the second objective, and the
question a practitioner asks is how much perplexity a unit of saved memory costs. That is a
slope, not a ranking, and it is only meaningful at matched capacity.

**Which genes matter?** Sobol indices over measured quality, so the answer comes from what
the models did rather than from what the cost model implies. The structural gate already
measured the genes' effect on *cost*; this is their effect on *quality*, and the two need
not agree.

**Are the parameter-optimal and memory-optimal fronts the same set?** If they were, the
second objective would be decoration and the problem would be single-objective. The
project's premise is that they are not, and a premise is worth checking.

Everything here reads the assembled table and computes; nothing fits a model or touches the
cluster. Partial coverage is carried into every result, because a design rule extracted from
a quarter of the space is a hypothesis about the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Bands the iso-parameter comparison uses. Ten rather than five, because each band is one
#: paired observation and a signed-rank test over five pairs can say almost nothing.
DEFAULT_BANDS = 10

#: Share of each band taken as the deep and the wide extreme. A third each leaves a middle
#: that is neither, which is what makes the comparison about the extremes rather than about
#: an arbitrary split of a continuum.
EXTREME_FRACTION = 1 / 3


@dataclass
class IsoParameterResult:
    """Depth against width at matched parameter count."""

    bands: int
    deep_means: np.ndarray = field(repr=False)
    wide_means: np.ndarray = field(repr=False)
    band_edges: list[tuple[float, float]] = field(default_factory=list)
    band_counts: list[int] = field(default_factory=list)

    @property
    def deltas(self) -> np.ndarray:
        """Deep minus wide. Positive means wide is better, since lower perplexity wins."""
        return self.deep_means - self.wide_means

    @property
    def bands_won_by_width(self) -> int:
        return int((self.deltas > 0).sum())

    def as_dict(self) -> dict[str, object]:
        return {
            "bands": self.bands,
            "band_edges_params": [[int(a), int(b)] for a, b in self.band_edges],
            "band_counts": self.band_counts,
            "deep_mean_perplexity": [round(float(v), 3) for v in self.deep_means],
            "wide_mean_perplexity": [round(float(v), 3) for v in self.wide_means],
            "delta_deep_minus_wide": [round(float(v), 3) for v in self.deltas],
            "bands_won_by_width": self.bands_won_by_width,
            "claim_under_test": (
                "The prior work reports that deep-and-thin architectures win. Within bands "
                "of equal parameter count, positive deltas mean the widest third reached "
                "lower perplexity than the deepest third."
            ),
        }


def iso_parameter_depth_vs_width(
    params: np.ndarray,
    perplexity: np.ndarray,
    depth: np.ndarray,
    width: np.ndarray,
    bands: int = DEFAULT_BANDS,
    extreme: float = EXTREME_FRACTION,
) -> IsoParameterResult:
    """Compare the deepest and widest thirds within bands of equal parameter count.

    Banding by parameter count is what makes this a comparison about shape rather than about
    size. Without it, deep architectures in this space are also smaller on average — the
    embedding dominates the parameter count and depth adds only block parameters — so any
    raw comparison of deep against wide is mostly a comparison of small against large.

    The ratio depth/width orders architectures by shape at a given size. Thirds rather than
    a median split, so the middle is excluded and the contrast is between the extremes the
    claim is about.
    """
    if not (0 < extreme < 0.5):
        raise ValueError("the extreme fraction must leave a middle")

    edges = np.percentile(params, np.linspace(0, 100, bands + 1))
    deep_means, wide_means, kept_edges, counts = [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        inside = (params >= lo) & (params <= hi)
        if inside.sum() < 6:
            continue

        shape = depth[inside] / width[inside]
        deep = perplexity[inside][shape >= np.quantile(shape, 1 - extreme)]
        wide = perplexity[inside][shape <= np.quantile(shape, extreme)]
        if len(deep) < 2 or len(wide) < 2:
            continue

        deep_means.append(float(deep.mean()))
        wide_means.append(float(wide.mean()))
        kept_edges.append((float(lo), float(hi)))
        counts.append(int(inside.sum()))

    if len(deep_means) < 2:
        raise ValueError(
            f"only {len(deep_means)} usable band(s). The comparison needs several, since "
            "each band is one paired observation."
        )

    return IsoParameterResult(
        bands=len(deep_means),
        deep_means=np.array(deep_means),
        wide_means=np.array(wide_means),
        band_edges=kept_edges,
        band_counts=counts,
    )


def test_iso_parameter(result: IsoParameterResult, rope: float,
                       rope_source: str) -> dict[str, object]:
    """Put the depth-versus-width comparison through the declared protocol.

    Blocks are parameter bands, treatments are the two shapes, and the metric is perplexity
    where lower wins. The Bayesian statement is the one that matters: a frequentist rejection
    over ten bands would say the two differ, and what a reader needs to know is whether the
    difference exceeds the noise floor the pilot measured.
    """
    from .statistics import Comparison, Declaration, plackett_luce, signed_rank

    declaration = Declaration(
        question="At a fixed parameter budget, does depth or width reach lower perplexity?",
        treatments=("deep", "wide"),
        blocks=tuple(f"band{i + 1}" for i in range(result.bands)),
        metric="perplexity",
        higher_is_better=False,
        rope=rope,
        rope_source=rope_source,
        notes=(
            "Banded by parameter count so the contrast is shape rather than size. In this "
            "space depth adds only block parameters while the embedding dominates the "
            "total, so an unbanded comparison of deep against wide is mostly small against "
            "large."
        ),
    )
    comparison = Comparison(
        declaration,
        np.column_stack([result.deep_means, result.wide_means]),
    )

    verdict = signed_rank(comparison, "deep", "wide")
    ranking = plackett_luce(comparison)

    return {
        "declaration": declaration.as_dict(),
        "mean_ranks": comparison.mean_ranks(),
        "bayesian": verdict.as_dict(),
        "probability_best": ranking.probability_best,
        "iso_parameter": result.as_dict(),
    }


@dataclass
class FrontComparison:
    """Whether the two cost objectives pick out the same architectures."""

    params_front: set[int]
    memory_front: set[int]
    joint_front: set[int]
    total: int

    @property
    def jaccard(self) -> float:
        union = self.params_front | self.memory_front
        return len(self.params_front & self.memory_front) / len(union) if union else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "architectures": self.total,
            "params_vs_quality_front": len(self.params_front),
            "memory_vs_quality_front": len(self.memory_front),
            "in_both": len(self.params_front & self.memory_front),
            "jaccard": round(self.jaccard, 4),
            "joint_front": len(self.joint_front),
            "only_params_optimal": len(self.params_front - self.memory_front),
            "only_memory_optimal": len(self.memory_front - self.params_front),
            "why_it_matters": (
                "If the two fronts were the same set, the KV-cache objective would be "
                "decoration and the problem would be single-objective with a relabelled "
                "axis. The project's premise is that they differ, and a premise is worth "
                "checking rather than asserting."
            ),
        }


def fronts_differ(params: np.ndarray, kv_bytes: np.ndarray,
                  perplexity: np.ndarray) -> FrontComparison:
    """The parameter-optimal front against the memory-optimal front.

    Two bi-objective fronts and the joint tri-objective one. An architecture can be
    parameter-efficient and memory-wasteful — grouped-query attention is precisely a knob
    that trades one for the other at almost no parameter cost — so the two need not agree,
    and how much they disagree is the size of the multi-objective problem.
    """
    from .space.enumerate import non_dominated

    params_front = set(non_dominated(np.column_stack([params, perplexity])).tolist())
    memory_front = set(non_dominated(np.column_stack([kv_bytes, perplexity])).tolist())
    joint = set(non_dominated(np.column_stack([params, kv_bytes, perplexity])).tolist())

    return FrontComparison(params_front, memory_front, joint, len(perplexity))


def gene_effects_on_quality(genes: np.ndarray, perplexity: np.ndarray,
                            names: tuple[str, ...]) -> dict[str, object]:
    """How much of the variation in measured quality each gene accounts for.

    Sobol first-order indices by a variance decomposition over the measured rows, which is
    what the data supports — a full Sobol analysis needs a designed sample the sweep does not
    provide, so this is the conditional-variance estimate and it is labelled as such.

    Reported separately from the structural gate's numbers on purpose. The gate measured each
    gene's effect on *cost*, which is closed-form; this is its effect on *quality*, which is
    measured, and the two need not agree. A gene that barely moves cost can dominate quality.
    """
    total_variance = float(perplexity.var())
    if total_variance == 0:
        raise ValueError("perplexity does not vary; there is nothing to decompose")

    effects = {}
    for index, name in enumerate(names):
        levels = np.unique(genes[:, index])
        # Variance of the conditional means: the part of the spread explained by knowing
        # this gene alone.
        weights, means = [], []
        for level in levels:
            mask = genes[:, index] == level
            if mask.sum() < 2:
                continue
            weights.append(mask.sum())
            means.append(perplexity[mask].mean())
        if len(means) < 2:
            continue
        weights = np.array(weights, float) / sum(weights)
        grand = float(np.sum(weights * np.array(means)))
        between = float(np.sum(weights * (np.array(means) - grand) ** 2))
        effects[name] = round(between / total_variance, 4)

    return {
        "first_order_indices": dict(sorted(effects.items(), key=lambda kv: -kv[1])),
        "sum_of_first_order": round(sum(effects.values()), 4),
        "estimator": (
            "Variance of the conditional means over measured rows, normalised by the total "
            "variance. A designed Sobol sample would give interaction terms too; the sweep "
            "is a space-filling sample rather than a Sobol design, so only first-order "
            "effects are estimable and the shortfall from one is where interactions live."
        ),
        "distinct_from_the_gate": (
            "The structural gate measured each gene's effect on cost, which is closed-form. "
            "This is the effect on measured quality. A gene can be nearly inert in cost and "
            "dominant in quality, and the gate's ffn finding is exactly that case."
        ),
    }


@dataclass
class GroupedQueryFrontier:
    """What sharing key-value heads costs, per byte of cache saved."""

    groups: list[dict] = field(default_factory=list)
    slope_per_mb: float = float("nan")
    slope_r2: float = float("nan")
    pairs: int = 0
    #: Per-pair elasticity: relative perplexity change per relative cache reduction.
    elasticities: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "matched_groups": len(self.groups),
            "paired_observations": self.pairs,
            "perplexity_per_mb_saved": round(self.slope_per_mb, 5),
            "fit_r2": round(self.slope_r2, 4),
            "elasticity": {
                "median": round(float(np.median(self.elasticities)), 4)
                if self.elasticities.size else None,
                "quartiles": [
                    round(float(np.percentile(self.elasticities, q)), 4)
                    for q in (25, 75)
                ] if self.elasticities.size else None,
                "share_costing_quality": round(
                    float((self.elasticities > 0).mean()), 4
                ) if self.elasticities.size else None,
                "definition": (
                    "Relative perplexity increase per relative cache reduction, per matched "
                    "pair. Reported as a distribution rather than a pooled slope because a "
                    "pooled slope in perplexity-per-megabyte is dominated by the largest "
                    "architectures: least squares weights by the square of the saving, and "
                    "the same six perplexity points are one per cent of 509 and ten per "
                    "cent of 60. The absolute slope is kept above for continuity and should "
                    "not be read as the answer."
                ),
            },
            "groups": self.groups,
            "what_the_slope_means": (
                "Perplexity given up for each megabyte of key-value cache removed, at "
                "matched head count and depth and width. Positive means the saving costs "
                "quality; near zero means grouped-query attention is close to free, which "
                "is the claim the technique is usually adopted on."
            ),
            "why_matched": (
                "kv is not independent of the rest: it must divide h, and architectures "
                "with many heads can reach cache reductions that few-headed ones cannot. "
                "Comparing across unmatched architectures measures the correlation between "
                "kv and everything kv is constrained by. Matching is on every gene but kv — "
                "leaving the feed-forward ratio free was enough to bury the effect entirely."
            ),
        }


def grouped_query_frontier(
    heads: np.ndarray,
    depth: np.ndarray,
    width: np.ndarray,
    ffn: np.ndarray,
    kv: np.ndarray,
    kv_bytes: np.ndarray,
    perplexity: np.ndarray,
) -> GroupedQueryFrontier:
    """Quality lost per megabyte of key-value cache saved, at matched architecture.

    Grouped-query attention shares key and value projections across query heads, so the kv
    gene is a knob that trades cache for quality at almost no parameter cost. How steep that
    trade is, is the question the second objective exists to answer.

    Measured within groups identical in *every* gene but kv. The first version grouped on
    heads, depth and width only, which left the feed-forward ratio free — so a group like
    h=2, n=2, e=64 mixed three ffn values whose perplexities differ by more than the kv
    effect being measured, and the fit returned a slope indistinguishable from zero with an
    R-squared of 0.002. Matching means matching on all of them.

    Slopes are pooled across groups by least squares through the origin on the paired
    differences, so a group with two kv values contributes one observation and one with four
    contributes six.
    """
    keys = {}
    for index in range(len(perplexity)):
        signature = (
            int(heads[index]), int(depth[index]), int(width[index]), int(ffn[index])
        )
        keys.setdefault(signature, []).append(index)

    deltas_mb, deltas_ppl, elasticities, groups = [], [], [], []
    for signature, members in sorted(keys.items()):
        if len(members) < 2:
            continue
        distinct_kv = {int(kv[i]) for i in members}
        if len(distinct_kv) < 2:
            continue

        pairs_here = 0
        for a in members:
            for b in members:
                if kv[a] <= kv[b]:
                    continue
                # a has more kv heads: a larger cache and, by hypothesis, better quality.
                saved_mb = (kv_bytes[a] - kv_bytes[b]) / 1e6
                if saved_mb <= 0:
                    continue
                deltas_mb.append(saved_mb)
                deltas_ppl.append(perplexity[b] - perplexity[a])
                # Both sides relative, so a pair of tiny models and a pair of large ones
                # contribute on the same scale.
                elasticities.append(
                    ((perplexity[b] - perplexity[a]) / perplexity[a])
                    / ((kv_bytes[a] - kv_bytes[b]) / kv_bytes[a])
                )
                pairs_here += 1

        if pairs_here:
            h, n, e, f = signature
            groups.append({
                "h": h, "n": n, "e": e, "ffn": f,
                "kv_values": sorted(distinct_kv),
                "pairs": pairs_here,
                "perplexity_range": [
                    round(float(min(perplexity[i] for i in members)), 3),
                    round(float(max(perplexity[i] for i in members)), 3),
                ],
            })

    if len(deltas_mb) < 3:
        raise ValueError(
            f"only {len(deltas_mb)} matched pair(s) differing in kv. The frontier needs "
            "architectures identical in heads, depth and width and differing only in how "
            "many key-value heads they share."
        )

    x = np.array(deltas_mb)
    y = np.array(deltas_ppl)
    # Through the origin: saving nothing costs nothing, and forcing an intercept would let
    # the fit absorb a constant offset that has no physical meaning here.
    slope = float((x @ y) / (x @ x))
    residual = y - slope * x
    r2 = 1.0 - float(residual @ residual) / float(y @ y) if float(y @ y) > 0 else float("nan")

    return GroupedQueryFrontier(
        groups=groups, slope_per_mb=slope, slope_r2=r2, pairs=len(deltas_mb),
        elasticities=np.array(elasticities),
    )


#: A rule must beat this much of the front over what the rule would catch by chance before
#: it is a rule rather than a description. Lift is the front's rate under the rule divided
#: by the front's rate overall: 1.0 means the rule says nothing.
MINIMUM_LIFT = 1.25

#: And it must cover enough of the front to be worth stating. A rule matching four
#: architectures can have enormous lift and describe nothing.
MINIMUM_COVERAGE = 0.15


@dataclass(frozen=True)
class DesignRule:
    """One candidate rule about Pareto-optimal architectures, and whether it survives.

    The trap this guards against is the one every "design rule" in the NAS literature walks
    into: 92 per cent of front architectures have fewer than 12 layers, which sounds like a
    finding until you notice 89 per cent of the whole space does too. A rule is only a rule
    where the front's rate under it exceeds the front's rate overall.
    """

    gene: str
    predicate: str
    #: Architectures satisfying the rule, and how many of those are on the front.
    matched: int
    matched_on_front: int
    #: The same over the whole table, for the baseline the rule has to beat.
    total: int
    total_on_front: int

    @property
    def precision(self) -> float:
        """Of the architectures the rule picks out, the share on the front."""
        return self.matched_on_front / self.matched if self.matched else 0.0

    @property
    def base_rate(self) -> float:
        """The share on the front without any rule at all."""
        return self.total_on_front / self.total if self.total else 0.0

    @property
    def coverage(self) -> float:
        """Of the front, the share the rule accounts for."""
        return self.matched_on_front / self.total_on_front if self.total_on_front else 0.0

    @property
    def lift(self) -> float:
        return self.precision / self.base_rate if self.base_rate else 0.0

    @property
    def holds(self) -> bool:
        return self.lift >= MINIMUM_LIFT and self.coverage >= MINIMUM_COVERAGE

    @property
    def verdict(self) -> str:
        if self.lift < MINIMUM_LIFT:
            return (
                f"describes rather than predicts: {self.precision:.1%} of matches are on "
                f"the front against a base rate of {self.base_rate:.1%}, a lift of "
                f"{self.lift:.2f}"
            )
        if self.coverage < MINIMUM_COVERAGE:
            return (
                f"too narrow to state: accounts for {self.coverage:.1%} of the front"
            )
        return (
            f"holds: {self.precision:.1%} of matches on the front against {self.base_rate:.1%} "
            f"overall, lift {self.lift:.2f}, covering {self.coverage:.1%} of the front"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "gene": self.gene,
            "rule": self.predicate,
            "matched": self.matched,
            "matched_on_front": self.matched_on_front,
            "precision": round(self.precision, 6),
            "base_rate": round(self.base_rate, 6),
            "lift": round(self.lift, 4),
            "coverage": round(self.coverage, 6),
            "holds": self.holds,
            "verdict": self.verdict,
        }


def design_rules(genes: np.ndarray, on_front: np.ndarray,
                 names: tuple[str, ...]) -> list[DesignRule]:
    """Every single-gene threshold rule, scored against the base rate it must beat.

    Thresholds only, and one gene at a time. Conjunctions over five genes would give
    hundreds of candidates over 1,099 architectures, and the best of hundreds looks
    impressive by construction — the multiplicity problem that makes most mined design
    rules unreproducible. A rule a reader can hold in their head is also the only kind
    worth putting in a paper.
    """
    genes = np.asarray(genes)
    on_front = np.asarray(on_front, dtype=bool)
    total, total_on_front = len(on_front), int(on_front.sum())

    rules: list[DesignRule] = []
    for index, name in enumerate(names):
        column = genes[:, index]
        for level in sorted(set(column.tolist()))[:-1]:
            for symbol, mask in (
                ("<=", column <= level),
                (">", column > level),
            ):
                matched = int(mask.sum())
                if not matched:
                    continue
                rules.append(DesignRule(
                    gene=name,
                    predicate=f"{name} {symbol} {int(level)}",
                    matched=matched,
                    matched_on_front=int((mask & on_front).sum()),
                    total=total,
                    total_on_front=total_on_front,
                ))
    return sorted(rules, key=lambda r: (-r.lift, -r.coverage))
