"""Does the ranking at 100M tokens predict the ranking at 400M?

The sweep trains every architecture on a fixed token budget, which is what makes it an
architecture comparison rather than a repeat of the prior work's defect. It also means every
architecture is trained at a different distance from its own compute-optimal point: with
parameters spanning 0.1 to 91 million non-embedding, a fixed 100M tokens is forty times more
than the smallest wants and a twentieth of what the largest does.

That is a systematic bias, not noise, and it has a specific consequence for what the sweep
already found. Width beats depth in every parameter band measured — but depth generally needs
more tokens to pay off, so "width wins" may be a statement about the budget rather than about
the shape. Nothing in the table can separate those.

So a stratified subset is retrained at four times the budget and the two rankings are
compared. Either answer is a result. A high rank correlation says the sweep's budget is
sufficient and the shape finding stands, which is a sentence that disarms the obvious
objection before it is made. A low one says multi-objective NAS rankings for text generation
are not stable under proxy budgets, which is a finding about the literature rather than about
this space.

**Four times and not more.** WikiText-103 holds 117.9 million tokens, so 400M is 3.4 epochs.
Repeated data stops behaving like fresh data somewhere around four passes, so this is the
largest multiple defensible without changing corpus — and changing corpus would confound the
budget question with a data question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..space.cost import forward_flops_per_token, parameter_count
from ..space.genotype import GENES, Genotype

#: Multiple of the sweep's budget. Set by the repeat-data ceiling rather than by taste.
BUDGET_MULTIPLE = 4

#: Architectures retrained. Enough for a stable rank correlation, few enough that the study
#: costs a fifth of the sweep rather than four times it.
DEFAULT_SUBSET = 50


@dataclass(frozen=True)
class BudgetStudyMember:
    genotype: Genotype
    stratum: int
    flops_per_token: int
    params: int

    def as_dict(self) -> dict[str, object]:
        return {
            "genotype": {gene: getattr(self.genotype, gene) for gene in GENES},
            "e": self.genotype.e,
            "stratum": self.stratum,
            "flops_per_token": self.flops_per_token,
            "params": self.params,
        }


def select(measured: list[Genotype], size: int = DEFAULT_SUBSET) -> list[BudgetStudyMember]:
    """A subset spread evenly over compute, drawn from what the sweep has measured.

    Stratified rather than random, and over compute rather than over the space. The question
    is whether the budget's distance from compute-optimal changes the ranking, and that
    distance is a function of size — so a sample that missed the expensive end would be
    unable to see the effect it exists to look for.

    Drawn only from architectures already measured at the sweep's budget, because a rank
    correlation needs both rankings over the same set.
    """
    if len(measured) < size:
        raise ValueError(
            f"{len(measured)} measured architectures cannot supply a subset of {size}. "
            "The comparison needs both budgets over the same architectures, so the subset "
            "comes from what the sweep has already done."
        )

    ordered = sorted(measured, key=forward_flops_per_token)
    # One from each equal-count stratum, taken at its midpoint, so the sample spans the
    # compute range without over-representing the dense cheap end.
    edges = np.linspace(0, len(ordered), size + 1).astype(int)
    chosen: list[BudgetStudyMember] = []
    for stratum, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False), start=1):
        if hi <= lo:
            continue
        genotype = ordered[(lo + hi) // 2]
        chosen.append(BudgetStudyMember(
            genotype=genotype,
            stratum=stratum,
            flops_per_token=forward_flops_per_token(genotype),
            params=parameter_count(genotype),
        ))
    return chosen


def plan(measured: list[Genotype], sweep_budget: int,
         size: int = DEFAULT_SUBSET) -> dict[str, object]:
    """What the study will run, and what each outcome would mean."""
    members = select(measured, size)
    flops = [m.flops_per_token for m in members]

    return {
        "architectures": len(members),
        "short_budget": sweep_budget,
        "long_budget": sweep_budget * BUDGET_MULTIPLE,
        "multiple": BUDGET_MULTIPLE,
        "epochs_at_long_budget": round(
            sweep_budget * BUDGET_MULTIPLE / 117_920_140, 2
        ),
        "compute_span": {
            "min_flops_per_token": min(flops),
            "max_flops_per_token": max(flops),
            "decades": round(float(np.log10(max(flops) / min(flops))), 2),
        },
        "why_four": (
            "WikiText-103 holds 117.9 million tokens, so four times the sweep's budget is "
            "3.4 epochs. Repeated data stops behaving like fresh data around four passes, "
            "making this the largest multiple defensible without changing corpus — and "
            "changing corpus would confound the budget question with a data question."
        ),
        "what_each_outcome_means": {
            "high_rank_correlation": (
                "the sweep's budget is sufficient and the shape finding stands on its own. "
                "Reported as a verification rather than as a discovery."
            ),
            "low_rank_correlation": (
                "multi-objective NAS rankings for text generation are not stable under "
                "proxy budgets. A finding about the literature rather than about this "
                "space, and a more interesting result than the first."
            ),
        },
        "members": [m.as_dict() for m in members],
    }


def compare(short: dict[tuple[int, ...], float],
            long: dict[tuple[int, ...], float]) -> dict[str, object]:
    """Rank correlation between the two budgets, over the architectures both measured.

    Reported with the overlap count, because a correlation over eight architectures and one
    over fifty are different claims and the number is what tells them apart.
    """
    from scipy.stats import kendalltau, spearmanr

    common = sorted(set(short) & set(long))
    if len(common) < 8:
        raise ValueError(
            f"only {len(common)} architecture(s) measured at both budgets; a rank "
            "correlation over fewer is a line through noise"
        )

    a = np.array([short[k] for k in common])
    b = np.array([long[k] for k in common])

    # How much each architecture improved, relative to itself. The spread of this is what
    # says whether the extra tokens helped everyone equally — if they did, the ranking
    # cannot have changed, and a high correlation would be uninformative.
    improvement = (a - b) / a

    return {
        "architectures_at_both_budgets": len(common),
        "kendall_tau": round(float(kendalltau(a, b).statistic), 4),
        "spearman": round(float(spearmanr(a, b).statistic), 4),
        "relative_improvement": {
            "median": round(float(np.median(improvement)), 4),
            "min": round(float(improvement.min()), 4),
            "max": round(float(improvement.max()), 4),
        },
        "interpretation": (
            "A high rank correlation means the sweep's budget orders architectures the same "
            "way four times the budget does. The spread of relative improvement is reported "
            "beside it because a uniform improvement cannot change a ranking, so a high "
            "correlation would then be a property of the design rather than evidence about "
            "the budget."
        ),
    }
