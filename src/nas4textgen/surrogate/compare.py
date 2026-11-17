"""The surrogate slate compared as a statistical question rather than a leaderboard.

A table of cross-validated Kendall taus sorted descending invites one reading — the top row
is the best predictor — and that reading is unavailable from the numbers alone. Fourteen
candidates over five folds produce a winner by arithmetic whether or not any candidate is
genuinely better, and the gap between first and second is routinely smaller than the gap
the same candidate shows between its own folds.

So the comparison blocks on the splits. Every candidate sees the same fold on the same
repetition, which is what makes a rank within a block mean something, and the noise floor
comes from how much each candidate varies across those blocks rather than from a threshold.

**Blocks are (repetition, fold) pairs, not folds.** Five folds is five blocks, and five is
too few — though not for the reason the algorithm comparison failed. There, few treatments
made rejection arithmetically impossible. Here fourteen treatments make the maximum
statistic large, so two blocks clear that bar; what five blocks cannot do is detect anything
short of a near-perfect agreement, which would make the p-value's size a fact about the
design. Four repetitions give twenty blocks, past the six a moderate agreement needs.
Repeating the cross-validation multiplies the blocks without pretending there is more data
than there is, and the repetition index is carried into the block label so a reader can see
the design.

**The trivial floors are treatments here, not a footnote.** Sorting by parameter count is
in the comparison as a candidate, so "the best model beats the floor" becomes a ranked
result with a p-value rather than two numbers a reader has to subtract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Folds per repetition, and how many repetitions. Five folds is conventional; five blocks
#: can only reject on a near-perfect agreement at this treatment count, and four
#: repetitions give twenty, past the six a moderate agreement needs.
FOLDS = 5
REPEATS = 4

#: The agreement across blocks a comparison should be able to detect. Kendall's W of 0.3 is
#: a real but not overwhelming consistency — the shape a genuine difference between
#: predictors takes — and the block count is derived from it rather than picked.
MODERATE_AGREEMENT = 0.3


class NotEnoughBlocks(RuntimeError):
    """Raised when the design cannot support a test over this many candidates."""


@dataclass
class FoldScores:
    """Kendall tau per candidate per block, which is what the test consumes."""

    #: candidate -> (repetition, fold) -> tau
    taus: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)

    def blocks(self) -> list[tuple[int, int]]:
        """Blocks every available candidate scored, so no block is missing a treatment."""
        usable = [set(v) for v in self.taus.values() if v]
        if not usable:
            return []
        shared = set.intersection(*usable)
        return sorted(shared)

    def candidates(self) -> list[str]:
        blocks = set(self.blocks())
        return sorted(
            name for name, values in self.taus.items()
            if blocks and blocks <= set(values)
            and all(values[b] == values[b] for b in blocks)  # no nan
        )


def per_fold_taus(
    candidates: list,
    features: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    folds: int = FOLDS,
    repeats: int = REPEATS,
    seed: int = 0,
) -> FoldScores:
    """Cross-validate every candidate, keeping each fold's tau rather than their mean.

    The splits are generated once per repetition and shared across candidates. Generating
    them per candidate would give each its own partition, and the comparison would be
    between partitions as much as between predictors.
    """
    from sklearn.model_selection import KFold

    from .evaluate import _fit_predict, kendall_tau

    scores = FoldScores()
    partitions = [
        list(KFold(folds, shuffle=True, random_state=seed + repetition).split(features))
        for repetition in range(repeats)
    ]

    for candidate in candidates:
        if candidate.build is None and candidate.closed_form is None:
            scores.unavailable[candidate.name] = candidate.why
            continue
        values: dict[tuple[int, int], float] = {}
        try:
            for repetition, splits in enumerate(partitions):
                for fold, (train, test) in enumerate(splits):
                    predicted = _fit_predict(
                        candidate, features, target, train, test, params
                    )
                    values[(repetition, fold)] = kendall_tau(predicted, target[test])
        except Exception as exc:  # noqa: BLE001
            # Reported rather than dropped: a slate that shrinks silently is a slate whose
            # result depends on which machine ran it.
            scores.unavailable[candidate.name] = (
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
            )
            continue
        scores.taus[candidate.name] = values
    return scores


def compare(scores: FoldScores) -> dict:
    """Friedman with Holm over blocks, plus the Bayesian ranking and a measured ROPE."""
    from scipy.stats import chi2

    from ..statistics import Comparison, Declaration, plackett_luce
    from ..statistics.frequentist import friedman, posthoc
    from ..statistics.rope import from_seed_matrix

    names = scores.candidates()
    blocks = scores.blocks()
    if len(names) < 2 or len(blocks) < 2:
        raise NotEnoughBlocks(
            f"{len(names)} candidate(s) over {len(blocks)} block(s); a comparison needs "
            "at least two of each, and a block missing a treatment is not a block"
        )

    k = len(names)
    critical = float(chi2.ppf(0.95, k - 1))

    # Two thresholds, because the arithmetic one is nearly vacuous here and saying so is
    # better than quoting it as reassurance. Friedman's statistic cannot exceed
    # blocks x (k - 1), so the blocks needed for rejection to be *possible* falls as
    # treatments grow — two, at fourteen candidates. What matters is the blocks needed to
    # detect a moderate agreement, since Kendall's W is the statistic divided by
    # blocks x (k - 1): at W = 0.3 and fourteen candidates that is six.
    impossible = int(np.ceil(critical / (k - 1)))
    underpowered = int(np.ceil(critical / (MODERATE_AGREEMENT * (k - 1))))
    if len(blocks) < impossible:
        raise NotEnoughBlocks(
            f"{len(blocks)} blocks against {k} candidates cannot reject at alpha 0.05 "
            f"whatever the data show; {impossible} are needed."
        )
    if len(blocks) < underpowered:
        raise NotEnoughBlocks(
            f"{len(blocks)} blocks against {k} candidates can only reject on a nearly "
            f"perfect agreement across blocks; {underpowered} are needed before a "
            f"moderate one (Kendall's W = {MODERATE_AGREEMENT}) reaches significance. "
            "Raise the repetition count rather than reporting a p-value whose size is a "
            "fact about the design."
        )

    rope = from_seed_matrix(
        [[scores.taus[name][b] for b in blocks] for name in names],
        metric="Kendall tau on held-out perplexity rank",
        source=(
            f"{k} candidates x {len(blocks)} blocks ({REPEATS} repetitions of {FOLDS}-fold "
            "cross-validation), pooled within candidate so real differences between them "
            "are removed"
        ),
    )
    declaration = Declaration(
        question="Which surrogate ranks held-out architectures best by Kendall tau?",
        treatments=tuple(names),
        blocks=tuple(f"r{r}f{f}" for r, f in blocks),
        metric="Kendall tau on held-out perplexity rank",
        higher_is_better=True,
        rope=rope.width,
        rope_source=rope.source,
        notes=(
            "Blocks are repetition-and-fold pairs so every candidate sees the same split. "
            "The trivial floors are treatments rather than a footnote: sorting by parameter "
            "count is in the comparison, so beating it is a ranked result with a p-value "
            "rather than two numbers a reader subtracts."
        ),
    )
    comparison = Comparison(
        declaration,
        np.array([[scores.taus[name][b] for name in names] for b in blocks], dtype=float),
    )

    omnibus = friedman(comparison)
    return {
        "declaration": declaration.as_dict(),
        "rope": rope.as_dict(),
        "mean_ranks": comparison.mean_ranks(),
        "friedman": omnibus.as_dict(),
        "power": {
            "blocks": len(blocks),
            "treatments": k,
            "blocks_for_rejection_to_be_possible": impossible,
            "blocks_to_detect_moderate_agreement": underpowered,
            "design": f"{REPEATS} repetitions x {FOLDS} folds",
            "why_two_thresholds": (
                "The arithmetic threshold falls as treatments grow — two, at fourteen "
                "candidates — so quoting it alone would read as reassurance while saying "
                "almost nothing. The second is the one the design is sized against."
            ),
        },
        "posthoc": ([e.as_dict() for e in posthoc(comparison, omnibus)]
                    if omnibus.p_value < 0.05 else None),
        "posthoc_withheld": None if omnibus.p_value < 0.05 else (
            f"the omnibus found no difference (p = {omnibus.p_value:.4g}), so no pairwise "
            "comparison is licensed"
        ),
        "probability_best": plackett_luce(comparison).probability_best,
        "unavailable": scores.unavailable,
        "median_tau": {
            name: float(np.median([scores.taus[name][b] for b in blocks]))
            for name in names
        },
    }
