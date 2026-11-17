"""Corpus hygiene: how a row is classified, and where a continuation is cut.

Two defects of the prior protocol live here, and both changed its reported numbers by
more than the effect it claimed to have found.

**Rows were never classified.** Measured on the real WikiText-2 test split, 33.7% of
rows are empty, 16.2% are section headers and 7.2% are short fragments. After the empty
ones are skipped, **24.5% of what remains is markup** — headers whose reference
continuation is a fragment of "= = =". Those rows score 0.6712 under the prior protocol
with an empty generation, against 0.2189 for actual prose, and the reported mean was
dominated by them.

**The cut was by characters.** ``cut_point = int(len(text) * 0.7)`` on a string cuts
mid-word: the row ``" = Robert Boulter = "`` became a prompt of ``" = Robert Boul"`` and
a reference of ``"ter = "``. The model was asked to complete a word fragment and scored
against a token that does not exist in the vocabulary as written.

Cutting on token boundaries and reporting strata separately are not refinements. Without
them a mean is a weighted average over incomparable populations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class RowKind(str, Enum):  # noqa: UP042 — StrEnum needs 3.11+ and this value is serialised
    """What a corpus row actually is.

    Inherits ``str`` so the member serialises as its own value into a manifest. StrEnum
    would be tidier but the JSON has to read the same either way, and this is the form
    that does not change what is written.
    """

    EMPTY = "empty"
    MARKUP = "markup"
    SHORT = "short"
    PROSE = "prose"


#: Rows shorter than this many words are fragments, not prose. Origin: the prior split's
#: own quantile binning put its lowest bin at 0–5 words, so the boundary is inherited
#: rather than invented.
SHORT_WORD_THRESHOLD = 15

#: Fraction of a row held back as the reference continuation.
CONTINUATION_FRACTION = 0.3

#: A continuation shorter than this cannot support an overlap metric: recall saturates on
#: a two-token reference and the score stops describing the generation. Rows below it are
#: excluded from overlap metrics and reported as excluded.
MIN_REFERENCE_TOKENS = 8


def classify(text: str) -> RowKind:
    """Classify a raw corpus row.

    Markup is detected before length, because a short header is markup rather than a
    short fragment and the two behave differently under an overlap metric.
    """
    stripped = text.strip()
    if not stripped:
        return RowKind.EMPTY
    if stripped.startswith("="):
        return RowKind.MARKUP
    if len(stripped.split()) < SHORT_WORD_THRESHOLD:
        return RowKind.SHORT
    return RowKind.PROSE


def strata(rows: list[str]) -> dict[str, float]:
    """Share of each row kind, recorded with every corpus that is prepared.

    Reported because a mean over a split whose composition is undeclared cannot be
    compared to anything, including itself after a filter changes.
    """
    counts = Counter(classify(r).value for r in rows)
    total = sum(counts.values()) or 1
    return {kind: counts.get(kind, 0) / total for kind in (k.value for k in RowKind)}


@dataclass(frozen=True)
class Split:
    """A row cut into a prompt and the continuation held back from it."""

    prompt_ids: list[int]
    reference_ids: list[int]
    kind: RowKind

    @property
    def reference_length(self) -> int:
        return len(self.reference_ids)

    @property
    def usable_for_overlap(self) -> bool:
        """Whether an overlap metric computed on this row would mean anything."""
        return (
            self.kind is RowKind.PROSE
            and self.reference_length >= MIN_REFERENCE_TOKENS
            and len(self.prompt_ids) > 0
        )


def split_on_token_boundary(
    token_ids: list[int],
    continuation_fraction: float = CONTINUATION_FRACTION,
    kind: RowKind = RowKind.PROSE,
) -> Split:
    """Cut a tokenised row into prompt and reference on a token boundary.

    Operating on token ids rather than a string is the whole point: there is no way to
    land mid-word, because a token is the smallest unit that exists.
    """
    if not 0.0 < continuation_fraction < 1.0:
        raise ValueError("continuation_fraction must lie strictly between 0 and 1")
    cut = len(token_ids) - int(round(len(token_ids) * continuation_fraction))
    cut = max(1, min(cut, len(token_ids) - 1)) if len(token_ids) > 1 else len(token_ids)
    return Split(
        prompt_ids=list(token_ids[:cut]),
        reference_ids=list(token_ids[cut:]),
        kind=kind,
    )


def prepare(
    rows: list[str],
    encode,
    max_length: int = 128,
    continuation_fraction: float = CONTINUATION_FRACTION,
) -> tuple[list[Split], dict[str, object]]:
    """Turn raw rows into splits, with a record of what was set aside and why.

    ``encode`` maps text to token ids. Returns the splits together with a report: the
    composition of the input, how many rows each filter removed, and the resulting
    reference-length distribution. The report is written into the corpus artifact,
    because a filtered evaluation set whose filtering is undocumented is not
    reproducible even by the people who built it.
    """
    composition = strata(rows)
    splits: list[Split] = []
    dropped = Counter()

    for row in rows:
        kind = classify(row)
        if kind is RowKind.EMPTY:
            dropped["empty"] += 1
            continue
        ids = encode(row)[:max_length]
        if len(ids) < 2:
            dropped["too_few_tokens"] += 1
            continue
        split = split_on_token_boundary(ids, continuation_fraction, kind)
        if not split.prompt_ids or not split.reference_ids:
            dropped["degenerate_split"] += 1
            continue
        splits.append(split)

    usable = [s for s in splits if s.usable_for_overlap]
    lengths = sorted(s.reference_length for s in splits)
    report: dict[str, object] = {
        "input_rows": len(rows),
        "input_composition": {k: round(v, 4) for k, v in composition.items()},
        "dropped": dict(dropped),
        "splits": len(splits),
        "usable_for_overlap": len(usable),
        "excluded_from_overlap": len(splits) - len(usable),
        "exclusion_reasons": {
            "not_prose": sum(1 for s in splits if s.kind is not RowKind.PROSE),
            "reference_shorter_than_minimum": sum(
                1
                for s in splits
                if s.kind is RowKind.PROSE and s.reference_length < MIN_REFERENCE_TOKENS
            ),
        },
        "reference_tokens": {
            "min": lengths[0] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "settings": {
            "continuation_fraction": continuation_fraction,
            "max_length": max_length,
            "short_word_threshold": SHORT_WORD_THRESHOLD,
            "min_reference_tokens": MIN_REFERENCE_TOKENS,
            "cut_unit": "tokens",
        },
    }
    return splits, report
