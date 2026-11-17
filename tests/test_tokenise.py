"""Tokenising once, and the staleness that would otherwise be silent."""

from __future__ import annotations

import pytest

from nas4textgen.corpora.cache import CacheMiss, store
from nas4textgen.corpora.tokenise import (
    is_tokenised,
    load_tokenised,
    tokenise_split,
)

pytest.importorskip("transformers", reason="the train extra is not installed")

ROWS = ["The quick brown fox.", "", "   ", "Jumps over the lazy dog.", "A third line."]


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.setenv("NAS4TG_CACHE", str(tmp_path))
    store("wikitext103", "validation", ROWS)
    return tmp_path


def test_blank_rows_are_dropped_before_encoding(cached):
    """A blank row encodes to nothing and would enter the benchmark as an empty sequence."""
    record = tokenise_split("wikitext103", "validation")
    assert record.sequences == 3
    assert record.tokens > 0

    sequences, _ = load_tokenised("wikitext103", "validation")
    assert len(sequences) == 3
    assert all(seq for seq in sequences)


def test_round_trip_matches_encoding_the_text_directly(cached):
    from transformers import GPT2TokenizerFast

    tokenise_split("wikitext103", "validation")
    sequences, _ = load_tokenised("wikitext103", "validation")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    expected = [tokenizer(r)["input_ids"] for r in ROWS if r.strip()]
    assert sequences == expected, "batching changed the token stream"


def test_the_tokeniser_that_produced_the_ids_is_recorded(cached):
    """A rebuild that moved tokenizers by a patch release could change the stream under
    half a sweep. The manifest has to name what produced these ids, not infer it."""
    record = tokenise_split("wikitext103", "validation")
    assert record.tokeniser == "gpt2"
    assert record.vocab_size == 50257
    assert record.tokenizers_version


def test_a_second_call_reuses_rather_than_recomputes(cached):
    first = tokenise_split("wikitext103", "validation")
    assert is_tokenised("wikitext103", "validation")
    second = tokenise_split("wikitext103", "validation")
    assert second.source_sha256 == first.source_sha256
    assert second.tokens == first.tokens


def test_reuse_is_keyed_on_the_source_hash_not_on_the_file_existing(cached):
    """Token ids left over from an earlier retrieval would otherwise be trained on
    against text they no longer match, and nothing would say so."""
    tokenise_split("wikitext103", "validation")

    store("wikitext103", "validation", ["Entirely different text now."])
    with pytest.raises(CacheMiss, match="different text"):
        load_tokenised("wikitext103", "validation")

    refreshed = tokenise_split("wikitext103", "validation")
    assert refreshed.sequences == 1


def test_a_truncated_token_file_is_refused(cached):
    tokenise_split("wikitext103", "validation")
    path = cached / "wikitext103" / "validation.tokens.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(CacheMiss, match="sequences"):
        load_tokenised("wikitext103", "validation")


def test_reading_before_tokenising_says_what_to_do(cached):
    with pytest.raises(CacheMiss, match="tokenise step"):
        load_tokenised("wikitext103", "validation")
