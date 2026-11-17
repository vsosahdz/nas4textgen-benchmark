"""Tokenise each corpus once, into the cache, rather than once per job.

Every training job was re-tokenising the same 1.17 million rows of WikiText-103. It costs
about one minute forty on a cluster node, which is nothing until it is multiplied by the
1,099 jobs in the sweep: roughly thirty-three hours of GPU-priced time spent doing CPU work
that produces an identical answer every time.

Tokenising here also removes a source of variation nobody would look for. A job that
tokenises its own input depends on the tokeniser version present in its image, so a rebuild
that moved ``tokenizers`` by a patch release could change the token stream under half a
sweep, and every downstream number with it. Doing it once makes the token ids part of the
data of record, hashed like everything else, and the tokeniser's identity is recorded beside
them rather than inferred from whichever container happened to run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .cache import CacheMiss, cache_root, load

#: The tokeniser the whole benchmark is defined against. Not a parameter: changing it would
#: change every perplexity in the table, so it belongs in the code rather than in a flag.
TOKENISER = "gpt2"

#: Rows per encode call. Batching is what makes this fast; one call per row spends most of
#: its time in Python rather than in the tokeniser.
BATCH_ROWS = 1000


@dataclass(frozen=True)
class TokenisedSplit:
    corpus: str
    split: str
    sequences: int
    tokens: int
    tokeniser: str
    vocab_size: int
    tokenizers_version: str
    source_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "split": self.split,
            "sequences": self.sequences,
            "tokens": self.tokens,
            "tokeniser": self.tokeniser,
            "vocab_size": self.vocab_size,
            "tokenizers_version": self.tokenizers_version,
            # The hash of the text this came from. A tokenised split whose source has been
            # re-retrieved no longer describes it, and comparing hashes is how that is
            # noticed rather than assumed.
            "source_sha256": self.source_sha256,
        }


def _paths(corpus_key: str, split: str) -> tuple[Path, Path]:
    base = cache_root() / corpus_key
    return base / f"{split}.tokens.jsonl", base / f"{split}.tokens.meta.json"


def is_tokenised(corpus_key: str, split: str) -> bool:
    data, meta = _paths(corpus_key, split)
    return data.exists() and meta.exists()


def tokenise_split(corpus_key: str, split: str, force: bool = False) -> TokenisedSplit:
    """Encode one split and write the token ids beside the text they came from."""
    from tokenizers import __version__ as tokenizers_version
    from transformers import GPT2TokenizerFast

    rows, record = load(corpus_key, split)
    data_path, meta_path = _paths(corpus_key, split)

    if not force and meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        # Keyed on the source hash, not on the file's existence. A tokenised split left
        # over from an earlier retrieval would otherwise be reused against text it no
        # longer matches, and nothing would say so.
        if existing.get("source_sha256") == record.sha256:
            return TokenisedSplit(**{
                k: existing[k] for k in TokenisedSplit.__dataclass_fields__
            })

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENISER)
    non_blank = [row for row in rows if row.strip()]

    total_tokens = 0
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(non_blank), BATCH_ROWS):
            batch = non_blank[start : start + BATCH_ROWS]
            for ids in tokenizer(batch)["input_ids"]:
                total_tokens += len(ids)
                handle.write(json.dumps(ids) + "\n")

    tokenised = TokenisedSplit(
        corpus=corpus_key,
        split=split,
        sequences=len(non_blank),
        tokens=total_tokens,
        tokeniser=TOKENISER,
        vocab_size=tokenizer.vocab_size,
        tokenizers_version=tokenizers_version,
        source_sha256=record.sha256,
    )
    meta_path.write_text(
        json.dumps(tokenised.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return tokenised


def load_tokenised(corpus_key: str, split: str) -> tuple[list[list[int]], TokenisedSplit]:
    """Read pre-tokenised sequences, refusing any that no longer match their source text."""
    data_path, meta_path = _paths(corpus_key, split)
    if not data_path.exists() or not meta_path.exists():
        raise CacheMiss(
            f"{corpus_key}/{split} has not been tokenised. Run the tokenise step once; "
            "jobs read token ids rather than encoding the corpus on every node."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    record = TokenisedSplit(**{k: meta[k] for k in TokenisedSplit.__dataclass_fields__})

    _, source = load(corpus_key, split, verify=False)
    if source.sha256 != record.source_sha256:
        raise CacheMiss(
            f"{corpus_key}/{split} was tokenised from different text: the source now "
            f"hashes to {source.sha256[:12]} against {record.source_sha256[:12]} recorded. "
            "Re-tokenise; training on stale ids would be silent."
        )

    with data_path.open(encoding="utf-8", newline="\n") as handle:
        sequences = [json.loads(line) for line in handle if line.strip()]

    if len(sequences) != record.sequences:
        raise CacheMiss(
            f"{corpus_key}/{split} holds {len(sequences)} sequences against "
            f"{record.sequences} recorded."
        )
    return sequences, record


def tokeniser_fingerprint() -> str:
    """Identity of the tokeniser, for the run manifest.

    Two runs are comparable only if their token streams came from the same tokeniser, and
    a version string alone does not establish that. Hashing the vocabulary does.
    """
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENISER)
    vocabulary = tokenizer.get_vocab()
    serialised = json.dumps(vocabulary, sort_keys=True).encode()
    return hashlib.sha256(serialised).hexdigest()[:16]
