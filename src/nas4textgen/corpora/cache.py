"""The data of record: retrieved once, read from cache thereafter.

Every downstream stage reads only from the cache, so the pipeline runs with the public
internet unavailable. That is not a convenience — a corpus that is re-downloaded per run
is a corpus that can change under the experiment, and a benchmark whose inputs moved is
not a benchmark.

The retrieval date is the study's **data vintage** and is recorded in every manifest. Two
results are comparable only if they share one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .registry import CORPORA, Corpus


def cache_root() -> Path:
    """Where corpora live. Overridable so tests never touch the real cache."""
    import os

    return Path(os.environ.get("NAS4TG_CACHE", "./data/cache")).resolve()


@dataclass(frozen=True)
class CachedSplit:
    """One split of one corpus, with everything that identifies it."""

    corpus: str
    split: str
    rows: int
    bytes_utf8: int
    sha256: str
    vintage: str
    identifier: str
    config: str | None
    revision: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "split": self.split,
            "rows": self.rows,
            "bytes_utf8": self.bytes_utf8,
            "sha256": self.sha256,
            "data_vintage": self.vintage,
            "identifier": self.identifier,
            "config": self.config,
            "revision": self.revision,
        }


def _digest(rows: Iterable[str]) -> tuple[str, int, int]:
    """Content hash, row count and byte count, computed in one pass.

    The hash is over the rows as stored, so a cache that has been edited — by a helpful
    cleanup script, or by a partial download — stops matching and is caught before it
    reaches an experiment.
    """
    hasher = hashlib.sha256()
    count = 0
    total_bytes = 0
    for row in rows:
        encoded = row.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
        count += 1
        total_bytes += len(encoded)
    return hasher.hexdigest(), count, total_bytes


def _paths(corpus_key: str, split: str) -> tuple[Path, Path]:
    base = cache_root() / corpus_key
    return base / f"{split}.jsonl", base / f"{split}.meta.json"


def is_cached(corpus_key: str, split: str) -> bool:
    data, meta = _paths(corpus_key, split)
    return data.exists() and meta.exists()


def store(corpus_key: str, split: str, rows: list[str], vintage: str | None = None) -> CachedSplit:
    """Write a split to the cache with its metadata.

    Called once per corpus and split, by the retrieval step. Everything after reads.
    """
    corpus: Corpus = CORPORA[corpus_key]
    data_path, meta_path = _paths(corpus_key, split)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    with data_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row}, ensure_ascii=False) + "\n")

    sha, count, total_bytes = _digest(rows)
    record = CachedSplit(
        corpus=corpus_key,
        split=split,
        rows=count,
        bytes_utf8=total_bytes,
        sha256=sha,
        vintage=vintage or datetime.now(UTC).date().isoformat(),
        identifier=corpus.identifier,
        config=corpus.config,
        revision=corpus.revision,
    )
    meta_path.write_text(
        json.dumps(record.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return record


class CacheMiss(RuntimeError):
    """Raised when a split is read that was never retrieved."""


class CacheCorrupt(RuntimeError):
    """Raised when cached content no longer matches the hash recorded with it."""


def load(corpus_key: str, split: str, verify: bool = True) -> tuple[list[str], CachedSplit]:
    """Read a split from the cache, verifying it still matches its recorded hash.

    Verification is on by default and costs one pass over data that is about to be read
    anyway. Skipping it would allow an experiment to run on content that has silently
    changed since it was measured, which is indistinguishable from a result until someone
    tries to reproduce it.
    """
    data_path, meta_path = _paths(corpus_key, split)
    if not data_path.exists() or not meta_path.exists():
        raise CacheMiss(
            f"{corpus_key}/{split} is not cached at {data_path.parent}. Retrieve it once "
            "with the corpus fetch step; downstream stages never reach the network."
        )

    record = CachedSplit(**{
        "corpus": (meta := json.loads(meta_path.read_text(encoding="utf-8")))["corpus"],
        "split": meta["split"],
        "rows": meta["rows"],
        "bytes_utf8": meta["bytes_utf8"],
        "sha256": meta["sha256"],
        "vintage": meta["data_vintage"],
        "identifier": meta["identifier"],
        "config": meta["config"],
        "revision": meta["revision"],
    })

    # Iterate the handle with newline="\n" rather than splitting the text on line
    # boundaries. `str.splitlines()` also breaks on U+2028, U+2029 and U+0085, which JSON
    # does not escape and which therefore sit raw inside a stored row — a Python file in
    # the code corpus contains twelve of them, and each one silently became an extra
    # "line" that no longer parsed. Prose corpora never exposed this. Reading by handle
    # also avoids holding a half-gigabyte corpus in memory twice.
    with data_path.open(encoding="utf-8", newline="\n") as handle:
        rows = [json.loads(line)["text"] for line in handle if line.strip()]

    if verify:
        sha, count, total_bytes = _digest(rows)
        if sha != record.sha256:
            raise CacheCorrupt(
                f"{corpus_key}/{split} no longer matches the hash recorded when it was "
                f"cached on {record.vintage}. The content changed after measurement, so "
                "any result computed from it is not reproducible. Re-retrieve it."
            )
        if count != record.rows or total_bytes != record.bytes_utf8:
            raise CacheCorrupt(
                f"{corpus_key}/{split} has {count} rows and {total_bytes} bytes against "
                f"{record.rows} and {record.bytes_utf8} recorded."
            )

    return rows, record
