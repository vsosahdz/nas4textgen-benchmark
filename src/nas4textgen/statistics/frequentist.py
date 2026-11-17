"""Friedman, then Holm. An omnibus test before any pairwise claim.

The order is the point. Running every pairwise comparison and reporting whichever came out
below 0.05 inflates the error rate exactly as many times as there are pairs — with seven
algorithms that is twenty-one chances to find a difference that is not there. The omnibus
test asks one question first: is anything different at all? Only if it says yes do the
pairwise comparisons mean anything, and even then each one is corrected for how many were
made.

Holm rather than Bonferroni: it controls the same family-wise error rate and is uniformly
more powerful, so preferring Bonferroni means discarding real differences for no gain.

Nothing here averages a p-value. The module that would have made that easy raises instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comparison import Comparison


@dataclass(frozen=True)
class OmnibusResult:
    statistic: float
    p_value: float
    n_treatments: int
    n_blocks: int
    mean_ranks: dict[str, float]
    #: Kendall's W. A p-value says whether the treatments differ; this says by how much
    #: the blocks agree about the ordering, which is the more useful number when the
    #: sample is large enough to make almost anything significant.
    concordance: float

    @property
    def differs(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict[str, object]:
        return {
            "test": "Friedman",
            "statistic": round(self.statistic, 6),
            "p_value": self.p_value,
            "n_treatments": self.n_treatments,
            "n_blocks": self.n_blocks,
            "mean_ranks": {k: round(v, 4) for k, v in self.mean_ranks.items()},
            "kendalls_w": round(self.concordance, 6),
            "differs_at_0.05": self.differs,
        }


def friedman(comparison: Comparison) -> OmnibusResult:
    """Friedman's test over the declared block-by-treatment matrix."""
    from scipy.stats import friedmanchisquare

    if comparison.n_blocks < 2:
        raise ValueError(
            f"Friedman needs at least two blocks; the declaration has "
            f"{comparison.n_blocks}. With one block there is nothing to rank across."
        )

    columns = [comparison.values[:, j] for j in range(comparison.n_treatments)]
    statistic, p_value = friedmanchisquare(*columns)

    # Kendall's W from the Friedman statistic: W = chi2 / (n_blocks * (k - 1)).
    k, n = comparison.n_treatments, comparison.n_blocks
    concordance = float(statistic) / (n * (k - 1)) if n * (k - 1) else 0.0

    return OmnibusResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_treatments=k,
        n_blocks=n,
        mean_ranks=comparison.mean_ranks(),
        concordance=concordance,
    )


@dataclass(frozen=True)
class PairwiseResult:
    a: str
    b: str
    statistic: float
    p_raw: float
    p_adjusted: float
    rejected: bool
    median_difference: float

    def as_dict(self) -> dict[str, object]:
        return {
            "a": self.a,
            "b": self.b,
            "statistic": round(self.statistic, 6),
            "p_raw": self.p_raw,
            "p_adjusted": self.p_adjusted,
            "rejected_at_0.05": self.rejected,
            "median_difference": round(self.median_difference, 6),
        }


def holm(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm–Bonferroni step-down. Returns adjusted p-values and rejection flags.

    Adjusted values are made monotone so a comparison can never end up with a smaller
    adjusted p-value than one that was more significant before correction — without that
    step the sorted output can invert and read as a contradiction.
    """
    m = len(p_values)
    if m == 0:
        return [], []

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, min(1.0, candidate))
        adjusted[index] = running

    return adjusted, [p <= alpha for p in adjusted]


def posthoc(comparison: Comparison, omnibus: OmnibusResult | None = None,
            alpha: float = 0.05) -> list[PairwiseResult]:
    """Every pair, by Wilcoxon signed-rank, with Holm correction across the family.

    Refuses to run when the omnibus test found nothing. That refusal is the guardrail: a
    pairwise search after a negative omnibus is exactly the procedure whose error rate
    nobody can state, and it is where a result that will not replicate comes from.
    """
    from itertools import combinations

    from scipy.stats import wilcoxon

    if omnibus is not None and not omnibus.differs:
        raise ValueError(
            f"the omnibus test found no difference (p = {omnibus.p_value:.4f}), so no "
            "pairwise comparison is licensed. Searching pairs after a negative omnibus "
            "has an error rate nobody can state.\n"
            "Report that the treatments are indistinguishable, and use the Bayesian "
            "analysis to say whether that means equivalent or merely undecided."
        )

    names = comparison.declaration.treatments
    pairs = list(combinations(names, 2))
    raw: list[float] = []
    statistics: list[float] = []
    medians: list[float] = []

    for a, b in pairs:
        differences = comparison.pairwise_differences(a, b)
        if np.allclose(differences, 0):
            # Wilcoxon is undefined when every difference is zero; that is not an error,
            # it is the strongest possible statement of no difference.
            statistics.append(0.0)
            raw.append(1.0)
        else:
            statistic, p = wilcoxon(differences)
            statistics.append(float(statistic))
            raw.append(float(p))
        medians.append(float(np.median(differences)))

    adjusted, rejected = holm(raw, alpha=alpha)
    return [
        PairwiseResult(a, b, statistics[i], raw[i], adjusted[i], rejected[i], medians[i])
        for i, (a, b) in enumerate(pairs)
    ]
