"""Retrieve corpora into the cache. The only stage that touches the network.

Everything downstream reads from the cache, so this runs once and its output is the study's
data of record. The retrieval date becomes the data vintage, and two results are comparable
only if they share one.

Failures are recorded rather than raised. Some corpora are gated behind an access
agreement, some change identifier, and a fetch that aborts on the first of those leaves the
others undone with nothing written down. The report says what was retrieved, what was not,
and why — which is also what the manuscript needs to state about its own inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .cache import is_cached, store
from .registry import CORPORA, Corpus

#: Which splits each corpus contributes. Training happens on the search corpus only; the
#: others receive the measured front, so they need no training split.
SPLITS: dict[str, tuple[str, ...]] = {
    "wikitext103": ("train", "validation", "test"),
    "tinystories": ("train", "validation"),
    "babylm": ("train", "validation"),
    "stack_python": ("train", "validation"),
}

#: Names differ between datasets; tried in order until one resolves.
SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "validation": ("validation", "valid", "dev", "test"),
    "test": ("test", "validation"),
}


@dataclass
class FetchOutcome:
    corpus: str
    split: str
    status: str  # cached | retrieved | unavailable | skipped
    rows: int = 0
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "split": self.split,
            "status": self.status,
            "rows": self.rows,
            "detail": self.detail,
        }


def _rows_from(dataset, corpus: Corpus) -> list[str]:
    """Pull the declared text column.

    Declared rather than inferred: guessing from a list of likely names works until an
    upstream schema change makes a different column match first, at which point the
    pipeline trains on the wrong field and reports nothing unusual.
    """
    if corpus.text_column not in dataset.column_names:
        raise KeyError(
            f"{corpus.identifier} has no column {corpus.text_column!r}; it offers "
            f"{dataset.column_names}. Update the registry rather than guessing."
        )
    return [row or "" for row in dataset[corpus.text_column]]


#: Fraction held out when a corpus publishes only one split, and the seed that fixes the
#: boundary. Both recorded in the cache, so the derived split is reproducible and is never
#: read as though upstream had drawn it.
DERIVED_VALIDATION_FRACTION = 0.1
DERIVED_SPLIT_SEED = 20260101


def _derive_splits(rows: list[str]) -> dict[str, list[str]]:
    """Split one upstream split into train and validation, deterministically."""
    import random

    order = list(range(len(rows)))
    random.Random(DERIVED_SPLIT_SEED).shuffle(order)
    cut = int(len(order) * (1 - DERIVED_VALIDATION_FRACTION))
    return {
        "train": [rows[i] for i in order[:cut]],
        "validation": [rows[i] for i in order[cut:]],
    }


def fetch_split(
    corpus_key: str,
    split: str,
    limit: int | None = None,
    force: bool = False,
) -> FetchOutcome:
    """Retrieve one split into the cache, or report why it could not be."""
    corpus: Corpus = CORPORA[corpus_key]

    if is_cached(corpus_key, split) and not force:
        return FetchOutcome(corpus_key, split, "cached", detail="already retrieved")

    try:
        from datasets import load_dataset
    except ImportError:
        return FetchOutcome(
            corpus_key, split, "unavailable",
            detail="the datasets package is not installed; it lives in the train extra",
        )

    aliases = ("train",) if corpus.single_upstream_split else SPLIT_ALIASES.get(split, (split,))
    last_error = ""
    for alias in aliases:
        try:
            kwargs = {"split": alias}
            if corpus.config:
                dataset = load_dataset(corpus.identifier, corpus.config, **kwargs)
            else:
                dataset = load_dataset(corpus.identifier, **kwargs)
            rows = _rows_from(dataset, corpus)
            if limit is not None:
                rows = rows[:limit]

            vintage = datetime.now(UTC).date().isoformat()
            if corpus.single_upstream_split:
                derived = _derive_splits(rows)
                record = store(corpus_key, split, derived[split], vintage=vintage)
                return FetchOutcome(
                    corpus_key, split, "retrieved", rows=record.rows,
                    detail=(
                        f"derived locally from the single upstream split {alias!r} "
                        f"(seed {DERIVED_SPLIT_SEED}, "
                        f"{DERIVED_VALIDATION_FRACTION:.0%} held out), "
                        f"{record.bytes_utf8:,} bytes"
                    ),
                )

            record = store(corpus_key, split, rows, vintage=vintage)
            return FetchOutcome(
                corpus_key, split, "retrieved", rows=record.rows,
                detail=f"as split {alias!r}, {record.bytes_utf8:,} bytes",
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"

    return FetchOutcome(corpus_key, split, "unavailable", detail=last_error)


def fetch_all(
    corpora: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
) -> list[FetchOutcome]:
    """Retrieve every declared split, continuing past whatever cannot be had.

    A gated dataset or a moved identifier stops that corpus, not the campaign. What is
    missing is reported so the manuscript can state which corpora its results actually
    rest on, rather than implying all four.
    """
    outcomes: list[FetchOutcome] = []
    for key in corpora or list(CORPORA):
        for split in SPLITS.get(key, ("train",)):
            outcomes.append(fetch_split(key, split, limit=limit, force=force))
    return outcomes
