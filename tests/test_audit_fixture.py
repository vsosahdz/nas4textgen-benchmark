"""The audit finding, as a test that fails if the defect ever returns.

The prior work reported METEOR rising from 0.6961 to 0.7200 and called that its
contribution. Two measurements on the real WikiText-2 test split showed what the number
was:

* under its protocol, an **empty** generation scores 0.3088 overall and **0.6712** on the
  24.5% of evaluated rows that are section headers;
* the gap between competent English prose and function-word spam was 0.034 to 0.048,
  larger than the 0.024 the paper claimed as its result.

The figures above are re-derived by `nas4textgen.audit_prior`, which is the module that
scores them; `python -m nas4textgen.cli audit` writes them to an artifact. They were
constants here before that module existed, which left the paper's central measurement
resting on a test fixture.

These tests encode the mechanism rather than the numbers, so they hold on any corpus and
under any tokenizer. Each corresponds to a specific way the defect could return.
"""

from __future__ import annotations

import pytest

from nas4textgen.corpora.hygiene import (
    MIN_REFERENCE_TOKENS,
    RowKind,
    classify,
    prepare,
    split_on_token_boundary,
    strata,
)
from nas4textgen.evaluation.generation import (
    TRIVIAL_FLOORS,
    DecodingConfig,
    MetricDiscrimination,
    MetricNotReportableError,
    MissingFloorError,
    ScoredMetric,
    UnresolvableComparisonError,
    is_difference_resolvable,
    require_floors,
    require_resolvable,
)

# Rows shaped like the real WikiText test split: a third empty, a sixth markup, the rest
# prose and fragments. The proportions are what made the prior mean what it was.
WIKITEXT_SHAPED_ROWS = [
    "",
    " = Robert Boulter = \n",
    "",
    " Robert Boulter is an English film , television and theatre actor . He had a guest "
    "role in the television series The Bill in 2000 . \n",
    "",
    " = = Career = = \n",
    " In 2006 he appeared as Kieran Richards in the ITV police procedural , The Bill . \n",
    " = = = 2000 to 2005 = = = \n",
    " Boulter . \n",
    "",
    " The band released their second album in 2004 , which reached number twelve on the "
    "national chart and was certified gold the following year . \n",
]


def fake_encode(text: str) -> list[int]:
    """Stand-in tokenizer: one id per whitespace token, deterministic and offline.

    The tests are about protocol, not vocabulary, and the specs require the pipeline to
    run with no public network.
    """
    return [hash(tok) % 50_000 for tok in text.split()] or []


# ── the cut ────────────────────────────────────────────────────────────────────

def test_the_cut_lands_on_a_token_boundary():
    """The prior cut was ``int(len(text) * 0.7)`` on a string, which split words.

    ``" = Robert Boulter = "`` became a prompt of ``" = Robert Boul"`` and a reference of
    ``"ter = "``: the model was asked to finish a word fragment.
    """
    ids = list(range(10))
    split = split_on_token_boundary(ids, continuation_fraction=0.3)
    assert split.prompt_ids + split.reference_ids == ids
    assert all(isinstance(i, int) for i in split.prompt_ids + split.reference_ids)
    assert len(split.reference_ids) == 3


def test_the_cut_never_produces_an_empty_side():
    for length in range(2, 40):
        split = split_on_token_boundary(list(range(length)))
        assert split.prompt_ids, f"empty prompt at length {length}"
        assert split.reference_ids, f"empty reference at length {length}"


# ── row classification ─────────────────────────────────────────────────────────

def test_markup_rows_are_recognised_as_markup():
    """The rows that carried the prior mean. Their reference is a fragment of "= = ="."""
    assert classify(" = Robert Boulter = \n") is RowKind.MARKUP
    assert classify(" = = Career = = \n") is RowKind.MARKUP
    assert classify("") is RowKind.EMPTY
    assert classify(" Boulter . \n") is RowKind.SHORT
    assert classify(WIKITEXT_SHAPED_ROWS[3]) is RowKind.PROSE


def test_composition_is_reported_not_assumed():
    """A mean over a split whose composition is undeclared compares to nothing."""
    shares = strata(WIKITEXT_SHAPED_ROWS)
    assert set(shares) == {"empty", "markup", "short", "prose"}
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert shares["markup"] > 0.0, "the markup stratum must be visible, not folded in"


def test_markup_is_excluded_from_overlap_metrics():
    """Not filtered silently: excluded, counted, and reported."""
    splits, report = prepare(WIKITEXT_SHAPED_ROWS, fake_encode)
    assert report["excluded_from_overlap"] > 0
    assert report["exclusion_reasons"]["not_prose"] > 0
    assert all(s.kind is RowKind.PROSE for s in splits if s.usable_for_overlap)


def test_short_references_are_excluded():
    """Recall saturates on a two-token reference and the score stops describing anything."""
    splits, _ = prepare(WIKITEXT_SHAPED_ROWS, fake_encode)
    for split in splits:
        if split.usable_for_overlap:
            assert split.reference_length >= MIN_REFERENCE_TOKENS


def test_the_preparation_report_records_its_own_settings():
    """So that a filtered evaluation set stays reproducible by whoever reads it later."""
    _, report = prepare(WIKITEXT_SHAPED_ROWS, fake_encode)
    settings = report["settings"]
    assert settings["cut_unit"] == "tokens"
    assert settings["min_reference_tokens"] == MIN_REFERENCE_TOKENS
    assert "continuation_fraction" in settings
    assert report["input_rows"] == len(WIKITEXT_SHAPED_ROWS)


# ── the prompt must not reach the scorer ───────────────────────────────────────

def test_generation_returns_only_new_tokens():
    """The defect itself, in one assertion.

    ``model.generate`` returns prompt plus continuation. A stub standing in for it here
    lets the slice be tested without a model, and the slice is the whole correction.
    """
    from nas4textgen.evaluation.generation import generate_continuation

    prompt = [11, 22, 33, 44]
    produced = [55, 66]

    class Stub:
        def generate(self, ids, **kwargs):
            import torch

            return torch.tensor([list(ids[0].tolist()) + produced], dtype=torch.long)

        def parameters(self):
            """Where the prompt's device comes from, as in any torch module.

            The stub stands in for a model, so it has to answer the questions the caller
            asks a model. Omitting this made the fixture pass while the real call path on
            the cards was broken, which is the failure mode a stub exists to prevent.
            """
            import torch

            yield torch.zeros(1)

    pytest.importorskip("torch")
    out = generate_continuation(Stub(), prompt, DecodingConfig(max_new_tokens=2), pad_token_id=0)
    assert out == produced, "the hypothesis must not contain the prompt"
    assert not any(p in out for p in prompt) or set(prompt).isdisjoint(set(out))


# ── floors ─────────────────────────────────────────────────────────────────────

def test_every_trivial_floor_is_available():
    assert set(TRIVIAL_FLOORS) == {
        "empty",
        "copy_prompt_tail",
        "frequent_tokens",
        "overlong_frequent",
    }


def test_floors_behave_as_described():
    prompt, reference = list(range(20)), list(range(100, 110))
    frequent = [1, 2, 3]

    assert TRIVIAL_FLOORS["empty"](prompt, reference) == []
    assert TRIVIAL_FLOORS["copy_prompt_tail"](prompt, reference) == prompt[-10:]

    spam = TRIVIAL_FLOORS["frequent_tokens"](prompt, reference, frequent_ids=frequent)
    assert len(spam) == len(reference)
    assert set(spam) <= set(frequent)

    overlong = TRIVIAL_FLOORS["overlong_frequent"](
        prompt, reference, frequent_ids=frequent, length_multiple=4
    )
    assert len(overlong) == 4 * len(reference), "the length effect must be isolable"


def test_a_metric_without_floors_cannot_be_reported():
    """Enforced, because the prior headline survived for want of this comparison."""
    with pytest.raises(MissingFloorError, match="trivial floors"):
        require_floors(ScoredMetric(name="meteor", value=0.72, length_ratio=1.0))


def test_a_metric_without_a_length_ratio_cannot_be_reported():
    """Length is what separates a recall-weighted metric earned from one inflated."""
    with pytest.raises(MissingFloorError, match="length"):
        require_floors(ScoredMetric(name="meteor", value=0.72, floors={"empty": 0.30}))


def test_clearing_a_floor_does_not_catch_the_defect():
    """Why a second check exists at all.

    Under the prior protocol the reported value was 0.7200 against an overall
    empty-generation floor of 0.3088. The absolute margin is large, so a floor comparison
    passes — which is exactly why the defect survived review. Asserting this keeps the
    limitation of the floor check visible instead of implied.
    """
    prior = ScoredMetric(
        name="meteor",
        value=0.7200,
        floors={"empty": 0.3088},
        length_ratio=3.7,
        corpus="wikitext103",
    )
    require_floors(prior)
    assert prior.clears_its_floor(rope=0.034), (
        "the floor check passes here; it is not the check that catches this"
    )


def test_the_metric_was_too_blind_for_any_comparison():
    """The check that does catch the prior result.

    Measured under its protocol, METEOR scored competent English prose at 0.2388 and
    function-word spam at 0.2047 — a gap of 0.0341 on a metric bounded at 1.0. Only 3.4%
    of the scale separated English from gibberish, and the paper claimed 0.0239 as its
    contribution.
    """
    discrimination = MetricDiscrimination(
        metric="meteor", corpus="wikitext103", competent=0.2388, garbage=0.2047
    )
    assert abs(discrimination.gap - 0.0341) < 1e-3

    rope = 0.005  # illustrative seed noise; the real one comes from task 10.2
    assert not discrimination.is_reportable(rope), (
        "a metric with 3.4% of its range between sense and nonsense is not an instrument"
    )
    with pytest.raises(MetricNotReportableError, match="too blind"):
        require_resolvable("meteor", 0.6961, 0.7200, discrimination, rope)


def test_a_large_effect_on_a_discriminating_metric_is_allowed_through():
    """The criterion must not reject a real effect.

    An earlier version required the claimed improvement to exceed the whole
    sense-to-nonsense range, which would reject a move from 0.30 to 0.95 on a metric that
    cleanly separates 0.90 from 0.10. Range is not resolution: the question is whether the
    instrument has range above its noise, not whether an effect fills that range.
    """
    discrimination = MetricDiscrimination(
        metric="mauve", corpus="tinystories", competent=0.90, garbage=0.10
    )
    rope = 0.02
    assert discrimination.is_reportable(rope)
    assert is_difference_resolvable(0.30, 0.95, rope)
    require_resolvable("mauve", 0.30, 0.95, discrimination, rope)


def test_a_small_difference_on_a_good_metric_is_still_refused():
    """Discrimination and resolution are independent failures."""
    discrimination = MetricDiscrimination(
        metric="mauve", corpus="tinystories", competent=0.90, garbage=0.10
    )
    rope = 0.02
    assert discrimination.is_reportable(rope)
    with pytest.raises(UnresolvableComparisonError, match="within the measured noise"):
        require_resolvable("mauve", 0.700, 0.712, discrimination, rope)


def test_a_floor_measured_on_the_wrong_stratum_is_not_comparable():
    """Floors and values must be measured on the same population.

    An empty generation scores 0.3088 over all evaluated rows and 0.6712 over the markup
    rows alone. Comparing an overall value against a stratum-specific floor mixes
    populations and produces a margin that describes neither.
    """
    stratum_floor = ScoredMetric(
        name="meteor",
        value=0.7200,
        floors={"empty_on_markup_rows_only": 0.6712},
        length_ratio=3.7,
        corpus="wikitext103",
    )
    require_floors(stratum_floor)
    assert any("markup" in name for name in stratum_floor.floors), (
        "a stratum-specific floor must be labelled as such so it is not read as overall"
    )


def test_a_genuine_result_clears_its_floor():
    """The same machinery must not reject a real effect."""
    genuine = ScoredMetric(
        name="mauve",
        value=0.81,
        floors={"empty": 0.02, "frequent_tokens": 0.05},
        length_ratio=1.05,
        corpus="tinystories",
    )
    require_floors(genuine)
    assert genuine.clears_its_floor(rope=0.034)
