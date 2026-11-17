"""Predicting perplexity from a genotype, and whether that is worth doing.

The study's purpose is not to find the best regressor. It is to answer whether a surrogate
earns its place in this campaign at all — and the honest answer may be that it barely does,
because sorting by parameter count is a strong baseline in a space where parameters are
dominated by the embedding matrix.

That is why the trivial floors are first-class members of the slate rather than a footnote,
and why a candidate that outranks neither is disqualified.
"""

from __future__ import annotations

from .budget import DEFAULT_SIZES, RELIABILITY_K, BudgetCurve, BudgetPoint, measure
from .evaluate import (
    PRECISION_AT,
    TOP_DECILE,
    Score,
    kendall_tau,
    precision_at_k,
    report,
    score_slate,
    top_decile_tau,
)
from .slate import EXCLUDED, GENE_ORDER, Candidate, build_slate

__all__ = [
    "DEFAULT_SIZES",
    "EXCLUDED",
    "GENE_ORDER",
    "PRECISION_AT",
    "TOP_DECILE",
    "BudgetCurve",
    "BudgetPoint",
    "Candidate",
    "RELIABILITY_K",
    "measure",
    "Score",
    "build_slate",
    "kendall_tau",
    "precision_at_k",
    "report",
    "score_slate",
    "top_decile_tau",
]
