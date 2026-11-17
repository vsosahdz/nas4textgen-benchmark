"""The data of record: cached once, verified on every read."""

from __future__ import annotations

import json

import pytest

from nas4textgen.corpora.cache import CacheCorrupt, CacheMiss, load, store
from nas4textgen.corpora.registry import (
    CORPORA,
    SEARCH_CORPUS,
    MetricNotLicensed,
    require_licensed,
)

ROWS = [" A paragraph of text .", " = A header = ", "", " Another paragraph here ."]


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NAS4TG_CACHE", str(tmp_path))


def test_a_split_round_trips_with_its_vintage():
    record = store("wikitext103", "test", ROWS, vintage="2026-08-12")
    rows, loaded = load("wikitext103", "test")
    assert rows == ROWS
    assert loaded.sha256 == record.sha256
    assert loaded.vintage == "2026-08-12", "two results compare only if they share one"


def test_reading_an_uncached_split_fails_rather_than_fetching():
    """Downstream stages never reach the network; a miss is an error, not a download."""
    with pytest.raises(CacheMiss, match="never reach the network"):
        load("tinystories", "train")


def test_content_that_changed_after_caching_is_refused():
    """A corpus that moved under an experiment invalidates every result from it."""
    store("wikitext103", "test", ROWS)
    path = json.loads  # keep the import used
    from nas4textgen.corpora.cache import _paths

    data_path, _ = _paths("wikitext103", "test")
    data_path.write_text(
        data_path.read_text(encoding="utf-8").replace("paragraph", "PARAGRAPH"),
        encoding="utf-8",
    )

    with pytest.raises(CacheCorrupt, match="not reproducible"):
        load("wikitext103", "test")
    assert path is json.loads


def test_verification_can_be_skipped_but_is_on_by_default():
    store("wikitext103", "test", ROWS)
    from nas4textgen.corpora.cache import _paths

    data_path, _ = _paths("wikitext103", "test")
    data_path.write_text(
        data_path.read_text(encoding="utf-8").replace("paragraph", "PARAGRAPH"),
        encoding="utf-8",
    )
    rows, _ = load("wikitext103", "test", verify=False)
    assert rows  # readable, but the default path would have refused it


# ── corpus roles ───────────────────────────────────────────────────────────────

def test_generation_metrics_are_refused_on_the_search_corpus():
    """At this scale a model trained on Wikipedia does not produce coherent text.

    A generation score there is not a weak result; it is a number with no referent, and
    the prior work shows how readily such a number becomes a headline.
    """
    with pytest.raises(MetricNotLicensed, match="not licensed"):
        require_licensed(SEARCH_CORPUS, "mauve")


def test_generation_metrics_are_licensed_where_models_are_coherent():
    require_licensed("tinystories", "mauve")
    require_licensed("tinystories", "repetition_rate")


def test_likelihood_is_licensed_everywhere():
    for key in CORPORA:
        require_licensed(key, "perplexity")


def test_at_least_one_corpus_offers_an_ungameable_metric():
    """Generated code either parses or it does not. A parser cannot be inflated."""
    assert any(
        "syntactic_validity" in c.licensed_metrics for c in CORPORA.values()
    ), "after a metric failure, one machine-verifiable signal is not optional"


def test_every_corpus_states_its_role_and_reasoning():
    for key, corpus in CORPORA.items():
        assert corpus.role, f"{key} has no stated role"
        assert len(corpus.notes) > 80, f"{key} does not explain why it is here"
        assert corpus.licensed_metrics


def test_round_trip_survives_unicode_line_separators(tmp_path, monkeypatch):
    """A row is one line, whatever line-like characters it contains.

    JSON escapes the C0 control characters but leaves U+2028, U+2029 and U+0085 raw, while
    ``str.splitlines()`` treats all three as line boundaries. Reading a cache by splitting
    text therefore invents rows that do not parse. Real Python source in the code corpus
    carries U+2028, so this is a corpus that was retrieved, not a constructed edge case.
    """
    monkeypatch.setenv("NAS4TG_CACHE", str(tmp_path))

    rows = [
        "def f():\n    return 1",           # newline: escaped by json, safe either way
        "a = 1 b = 2",                 # line separator: the one that actually bit
        "x y",                          # paragraph separator
        "p\x85q",                            # next line
        "carriage\rreturn",                  # escaped by json as \r
        "form\x0cfeed",                      # escaped by json as \f
    ]
    record = store("stack_python", "train", rows)
    assert record.rows == len(rows)

    loaded, meta = load("stack_python", "train")
    assert loaded == rows, "a row containing a Unicode line separator was split"
    assert meta.sha256 == record.sha256


def test_load_rejects_content_edited_after_caching(tmp_path, monkeypatch):
    """Verification is the point of storing a hash; assert it actually fires."""
    monkeypatch.setenv("NAS4TG_CACHE", str(tmp_path))
    store("wikitext103", "validation", ["one", "two"])

    data = tmp_path / "wikitext103" / "validation.jsonl"
    data.write_text(
        json.dumps({"text": "one"}) + "\n" + json.dumps({"text": "edited"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CacheCorrupt):
        load("wikitext103", "validation")
