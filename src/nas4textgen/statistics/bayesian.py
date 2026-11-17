"""The analysis that can say "equivalent" rather than only "not distinguishable".

A null result from a frequentist test is silent about which of two very different
situations produced it: the treatments are the same, or the experiment could not tell.
Reporting the first when only the second is supported is one of the ways a benchmark
overclaims, and this project exists partly because of an overclaim.

Two procedures, both standard in the empirical-comparison literature:

**Bayesian signed-rank** for a pair. It returns three probabilities that sum to one —
that A is practically better, that they are practically equivalent, and that B is
practically better — where "practically" means the difference exceeds a ROPE derived from
measured noise. A high middle probability is a positive statement of equivalence, which no
p-value can make.

**Plackett–Luce** for a slate. Given several algorithms over many blocks it returns the
posterior probability that each is the best, which is what a reader actually wants from a
comparison of seven optimisers and cannot get from twenty-one pairwise tests.

Both are computed by simulation. The seed is an argument rather than a global, so a
reported probability can be regenerated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comparison import Comparison, MalformedComparison

#: Draws for the posterior. Large enough that the third decimal of a reported probability
#: is stable, which is as far as any of these numbers are quoted.
DEFAULT_DRAWS = 50_000


@dataclass(frozen=True)
class PairedVerdict:
    a: str
    b: str
    rope: float
    probability_a: float
    probability_equivalent: float
    probability_b: float
    draws: int
    seed: int

    @property
    def verdict(self) -> str:
        """The claim this supports, in words, at the usual 0.95 threshold."""
        if self.probability_equivalent >= 0.95:
            return f"{self.a} and {self.b} are practically equivalent"
        if self.probability_a >= 0.95:
            return f"{self.a} is practically better"
        if self.probability_b >= 0.95:
            return f"{self.b} is practically better"
        return "undecided — the data do not support any of the three claims"

    def as_dict(self) -> dict[str, object]:
        return {
            "a": self.a,
            "b": self.b,
            "rope": self.rope,
            f"p({self.a} better)": round(self.probability_a, 4),
            "p(equivalent)": round(self.probability_equivalent, 4),
            f"p({self.b} better)": round(self.probability_b, 4),
            "verdict": self.verdict,
            "draws": self.draws,
            "seed": self.seed,
        }


def signed_rank(
    comparison: Comparison,
    a: str,
    b: str,
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260101,
) -> PairedVerdict:
    """Bayesian signed-rank test with a ROPE, by Dirichlet-weighted resampling.

    The posterior over the differences is a Dirichlet process centred on the observed
    values; drawing weights from it and taking the weighted median gives the posterior of
    the median difference. Comparing that to the ROPE partitions the posterior into the
    three claims.
    """
    declaration = comparison.declaration
    if declaration.rope is None:
        raise MalformedComparison(
            f"{declaration.question!r} declares no ROPE, so 'practically equivalent' has "
            "no meaning here. Derive one from measured noise — see statistics/rope.py — "
            "and record the measurement it came from."
        )

    differences = comparison.pairwise_differences(a, b)
    rng = np.random.default_rng(seed)
    n = len(differences)

    # Dirichlet weights over the observed differences. The pseudo-observation at zero is
    # the prior: it keeps a unanimous sample from producing a probability of exactly one,
    # which would claim more certainty than n observations can carry.
    support = np.concatenate([[0.0], differences])
    weights = rng.dirichlet(np.ones(n + 1), size=draws)

    order = np.argsort(support)
    sorted_support = support[order]
    cumulative = np.cumsum(weights[:, order], axis=1)
    # Weighted median: first point where the cumulative weight reaches one half.
    indices = np.argmax(cumulative >= 0.5, axis=1)
    medians = sorted_support[indices]

    rope = declaration.rope
    p_a = float(np.mean(medians > rope))
    p_b = float(np.mean(medians < -rope))
    p_eq = float(np.mean(np.abs(medians) <= rope))

    return PairedVerdict(a, b, rope, p_a, p_eq, p_b, draws, seed)


@dataclass(frozen=True)
class SlateRanking:
    treatments: tuple[str, ...]
    probability_best: dict[str, float]
    expected_rank: dict[str, float]
    credible_rank_interval: dict[str, tuple[int, int]]
    draws: int
    seed: int

    @property
    def winner(self) -> str:
        return max(self.probability_best, key=self.probability_best.__getitem__)

    def as_dict(self) -> dict[str, object]:
        return {
            "model": "Bayesian Plackett-Luce over within-block rankings",
            "probability_best": {k: round(v, 4) for k, v in self.probability_best.items()},
            "expected_rank": {k: round(v, 3) for k, v in self.expected_rank.items()},
            "credible_rank_interval_95": {
                k: list(v) for k, v in self.credible_rank_interval.items()
            },
            "winner": self.winner,
            "draws": self.draws,
            "seed": self.seed,
            "why": (
                "A reader comparing seven algorithms wants the probability each is best, "
                "which twenty-one pairwise tests cannot give. The rank interval is "
                "reported beside it because a treatment can be most probably best and "
                "still have a wide interval, and reporting only the winner hides that."
            ),
        }


def plackett_luce(
    comparison: Comparison,
    draws: int = DEFAULT_DRAWS,
    seed: int = 20260101,
) -> SlateRanking:
    """Posterior probability that each treatment is best, over the whole slate.

    Fitted on within-block rankings rather than raw values, so a corpus whose perplexities
    are numerically large cannot dominate one whose are small. Blocks are resampled with a
    Dirichlet weighting, which propagates the uncertainty that comes from having measured
    on finitely many corpora and seeds — the uncertainty a point estimate of mean rank
    throws away.
    """
    ranks = comparison.ranks()
    names = comparison.declaration.treatments
    n_blocks, k = ranks.shape
    rng = np.random.default_rng(seed)

    weights = rng.dirichlet(np.ones(n_blocks), size=draws)
    # Weighted mean rank per draw: (draws x n_blocks) @ (n_blocks x k).
    weighted = weights @ ranks

    # Lower mean rank is better, since ranks() puts 1 at the best.
    best = np.argmin(weighted, axis=1)
    probability_best = {
        name: float(np.mean(best == j)) for j, name in enumerate(names)
    }

    # Rank of each treatment within each draw, as a reader would read a leaderboard.
    draw_order = np.argsort(np.argsort(weighted, axis=1), axis=1) + 1
    expected_rank = {
        name: float(draw_order[:, j].mean()) for j, name in enumerate(names)
    }
    intervals = {
        name: (
            int(np.percentile(draw_order[:, j], 2.5)),
            int(np.percentile(draw_order[:, j], 97.5)),
        )
        for j, name in enumerate(names)
    }

    return SlateRanking(
        treatments=names,
        probability_best=probability_best,
        expected_rank=expected_rank,
        credible_rank_interval=intervals,
        draws=draws,
        seed=seed,
    )
