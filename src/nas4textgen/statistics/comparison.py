"""A comparison must declare what it compares before it is allowed to run.

The spec requires treatments, blocks and metric to be named in advance, and this is where
that requirement becomes something a program can enforce rather than something an author
remembers. It is not bureaucracy. The two ways a comparison of this kind goes wrong are
both invisible in the output:

**Mixing structures.** Averaging a p-value from a per-corpus comparison with one from a
per-seed comparison produces a number that answers no question. Both look like p-values.

**Deciding the structure after seeing the data.** If the blocks are chosen once the
results are in, the test is conditioned on the outcome and its error rate is not what it
claims. Declaring first is what makes the claim honest, and a declaration that lives in
code cannot be revised silently afterwards.

So a ``Comparison`` is constructed from a declaration, validated against the data it is
given, and refuses to run if they disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


class MalformedComparison(RuntimeError):
    """Raised when data does not match the structure the comparison declared."""


class AggregatedPValue(RuntimeError):
    """Raised on any attempt to average p-values instead of correcting for multiplicity."""


@dataclass(frozen=True)
class Declaration:
    """What is being compared, over what, on which measurement.

    ``treatments`` are the things being ranked — architectures, algorithms, surrogates.
    ``blocks`` are what every treatment is measured on in common, and what the design
    controls for: corpora, seeds, folds. ``metric`` names the measurement and
    ``higher_is_better`` fixes its direction, which no test can infer.
    """

    question: str
    treatments: tuple[str, ...]
    blocks: tuple[str, ...]
    metric: str
    higher_is_better: bool
    #: The smallest difference in ``metric`` that would matter. Derived from measured
    #: noise, never chosen — see ``rope.py``. Optional because an omnibus test does not
    #: need one; every Bayesian statement here does.
    rope: float | None = None
    rope_source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.treatments) < 2:
            raise MalformedComparison(
                f"{self.question!r} declares {len(self.treatments)} treatment(s); a "
                "comparison needs at least two."
            )
        if len(set(self.treatments)) != len(self.treatments):
            raise MalformedComparison("treatment names repeat, so results cannot be matched")
        if len(set(self.blocks)) != len(self.blocks):
            raise MalformedComparison("block names repeat")
        if self.rope is not None and not self.rope_source:
            raise MalformedComparison(
                "a ROPE without a source is a threshold someone chose. State the "
                "measurement it came from — the spec requires it reported alongside."
            )
        if self.rope is not None and self.rope <= 0:
            raise MalformedComparison("a ROPE must be a positive width")

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "treatments": list(self.treatments),
            "blocks": list(self.blocks),
            "n_treatments": len(self.treatments),
            "n_blocks": len(self.blocks),
            "metric": self.metric,
            "direction": "higher is better" if self.higher_is_better else "lower is better",
            "rope": self.rope,
            "rope_source": self.rope_source,
            "notes": self.notes,
        }


@dataclass
class Comparison:
    """A declaration bound to a complete block-by-treatment matrix."""

    declaration: Declaration
    #: Rows are blocks, columns are treatments, in the declared order.
    values: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        expected = (len(self.declaration.blocks), len(self.declaration.treatments))
        if self.values.shape != expected:
            raise MalformedComparison(
                f"{self.declaration.question!r} declared {expected[0]} blocks and "
                f"{expected[1]} treatments, so it needs a {expected} matrix; got "
                f"{self.values.shape}."
            )
        if not np.isfinite(self.values).all():
            # An unbalanced design silently filled with NaN is the common way a blocked
            # test ends up ranking treatments on different sets of blocks.
            missing = np.argwhere(~np.isfinite(self.values))
            first = missing[0]
            raise MalformedComparison(
                f"{len(missing)} missing value(s), first at block "
                f"{self.declaration.blocks[first[0]]!r} / treatment "
                f"{self.declaration.treatments[first[1]]!r}. A blocked comparison needs "
                "every treatment measured on every block; drop the block or fill the gap, "
                "but do not let the test rank treatments over different block sets."
            )

    @property
    def n_blocks(self) -> int:
        return self.values.shape[0]

    @property
    def n_treatments(self) -> int:
        return self.values.shape[1]

    def ranks(self) -> np.ndarray:
        """Within-block ranks, 1 = best, averaging ties.

        Ranking inside each block is what makes the comparison scale-free, which matters
        because perplexity on WikiText and on TinyStories are not the same quantity and
        averaging them directly would weight whichever corpus happens to have larger
        numbers.
        """
        from scipy.stats import rankdata

        # rankdata puts smallest first; flip when larger is better so 1 is always best.
        oriented = -self.values if self.declaration.higher_is_better else self.values
        return np.vstack([rankdata(row) for row in oriented])

    def mean_ranks(self) -> dict[str, float]:
        means = self.ranks().mean(axis=0)
        return dict(zip(self.declaration.treatments, (float(m) for m in means), strict=True))

    def pairwise_differences(self, a: str, b: str) -> np.ndarray:
        """Per-block differences a − b, oriented so positive always favours ``a``."""
        names = self.declaration.treatments
        i, j = names.index(a), names.index(b)
        diff = self.values[:, i] - self.values[:, j]
        return diff if self.declaration.higher_is_better else -diff


def refuse_averaged_pvalues(values: Sequence[float], context: str = "") -> None:
    """Refuse the operation the spec forbids, wherever someone reaches for it.

    Averaging p-values is not a conservative shortcut; it has no null distribution and the
    result is not a p-value of anything. The correct handling is an omnibus test followed
    by a post-hoc correction, which is what ``friedman`` and ``holm`` provide. This exists
    so the forbidden path raises rather than quietly returning a plausible number.
    """
    raise AggregatedPValue(
        f"averaging {len(values)} p-values{f' in {context}' if context else ''} produces a "
        "number with no null distribution — it is not a p-value of any hypothesis.\n"
        "Run the omnibus test, then correct the pairwise comparisons for multiplicity: "
        "friedman() followed by holm()."
    )
