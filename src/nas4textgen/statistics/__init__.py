"""Statistical comparison, structured so the forbidden shortcuts raise.

The spec asks for three properties, and each is enforced here rather than remembered:

- a comparison declares its treatments, blocks and metric before it runs
  (``comparison.Declaration``);
- multiplicity is handled by an omnibus test followed by a post-hoc correction, never by
  averaging p-values (``frequentist``, and ``comparison.refuse_averaged_pvalues``);
- every region of practical equivalence comes from measured noise and is reported with
  the measurement (``rope``).

The Bayesian side exists because a null frequentist result cannot distinguish "these are
the same" from "this experiment could not tell", and reporting the first when only the
second holds is how a benchmark overclaims.
"""

from __future__ import annotations

from .bayesian import PairedVerdict, SlateRanking, plackett_luce, signed_rank
from .comparison import (
    AggregatedPValue,
    Comparison,
    Declaration,
    MalformedComparison,
    refuse_averaged_pvalues,
)
from .frequentist import OmnibusResult, PairwiseResult, friedman, holm, posthoc
from .rope import InsufficientReplicates, MeasuredRope, from_replicates, from_seed_matrix

__all__ = [
    "AggregatedPValue",
    "Comparison",
    "Declaration",
    "InsufficientReplicates",
    "MalformedComparison",
    "MeasuredRope",
    "OmnibusResult",
    "PairedVerdict",
    "PairwiseResult",
    "SlateRanking",
    "friedman",
    "from_replicates",
    "from_seed_matrix",
    "holm",
    "plackett_luce",
    "posthoc",
    "refuse_averaged_pvalues",
    "signed_rank",
]
