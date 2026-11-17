"""The re-derivation of the prior protocol, as tests over a corpus shaped like the real one.

``tests/test_audit_fixture.py`` encodes the *mechanism* of the defect and carries the
published decimals as constants. This file covers the module that derives them, which is a
different obligation: that the derivation is sound, that the protocol it reconstructs is the
one the prior work ran, and that the conclusion follows from the numbers rather than from the
prose around them.

No network and no tokeniser download. METEOR is computed over words in any case, and a test
that needs to reach the internet is a test that gets skipped on the machine where it would
have mattered.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nltk")

from nas4textgen.audit_prior import (  # noqa: E402
    PRIOR_PROMPT_FRACTION,
    PROTOCOLS,
    Case,
    audit,
    hypothesis_unrelated_prose,
    prior_cases,
    sensitivity,
)

# Rows shaped like the WikiText test split: markup headers, prose of varying length, and the
# empty rows that make up a third of it.
ROWS = [
    "",
    " = Robert Boulter = \n",
    " Robert Boulter is an English film , television and theatre actor . He had a guest "
    "role in the television series The Bill in 2000 . \n",
    "",
    " = = Career = = \n",
    " In 2006 he appeared as Kieran Richards in the ITV police procedural , The Bill , "
    "and later returned for a second series in the same role . \n",
    " = = = 2000 to 2005 = = = \n",
    " The band released their second album in 2004 , which reached number twelve on the "
    "national chart and was certified gold the following year . \n",
    "",
    " She studied at the Royal Academy of Dramatic Art and went on to work in repertory "
    "theatre across the north of England for several seasons . \n",
    " = = Reception = = \n",
    " The film received mixed reviews from critics , who praised the performances while "
    "criticising the length of the second act . \n",
]

VOCABULARY = sorted({word for row in ROWS for word in row.split()})
IDS = {word: index + 1 for index, word in enumerate(VOCABULARY)}
WORDS = {index: word for word, index in IDS.items()}


def encode(text: str) -> list[int]:
    for word in text.split():
        if word not in IDS:
            IDS[word] = len(IDS) + 1
            WORDS[IDS[word]] = word
    return [IDS[word] for word in text.split()]


def decode(ids) -> str:
    return " ".join(WORDS.get(i, "") for i in ids).strip()


@pytest.fixture(scope="module")
def report():
    return audit(ROWS, encode=encode, decode=decode, with_sensitivity=False)


class TestTheProtocolItReconstructs:
    def test_the_cut_is_by_characters_because_that_is_what_the_prior_code_did(self):
        """The reconstruction that matters.

        Cutting on a token boundary instead moves the headline figure by a third and lands
        on a protocol the prior work never ran — this project's corrected one, wearing the
        prior protocol's name. The character cut is why ``" = Robert Boulter = "`` yields a
        prompt ending mid-word, which is the behaviour being audited.
        """
        row = " = Robert Boulter = \n"
        case = prior_cases([row])[0]
        cut = int(len(row) * PRIOR_PROMPT_FRACTION)
        assert " ".join(case.prompt) == " ".join(row[:cut].split())
        assert PRIOR_PROMPT_FRACTION == 0.7

    def test_empty_rows_are_skipped_and_markup_is_not(self):
        """A third of the corpus is blank and a sixth is headers. Dropping the headers too
        would remove the stratum that carries the effect."""
        kinds = {case.kind for case in prior_cases(ROWS)}
        assert "markup" in kinds
        assert "empty" not in kinds


class TestTheDefect:
    def test_removing_the_prompt_takes_an_empty_generation_to_zero(self, report):
        """The attribution. The two protocols differ in one step, so the whole of the score
        an empty generation earns is the prompt being read back."""
        assert report.score("empty", "prompt_removed").meteor == 0.0
        assert report.score("empty", "prior").meteor > 0.1
        assert report.prompt_defect_contribution == pytest.approx(
            report.score("empty", "prior").meteor)

    def test_writing_nothing_scores_nothing_when_scored_correctly(self, report):
        """The control. Without it a high score for an empty generation proves nothing: it
        could be the metric being generous rather than the protocol being broken."""
        assert report.score("empty", "corrected").meteor == 0.0

    def test_every_hypothesis_gains_from_the_defect_not_just_the_empty_one(self, report):
        """What makes the gap attributable to the protocol rather than to one hypothesis."""
        for hypothesis in ("empty", "copy_prompt_tail", "frequent_tokens"):
            prior = report.score(hypothesis, "prior")
            removed = report.score(hypothesis, "prompt_removed")
            assert prior.meteor > removed.meteor, hypothesis


class TestTheArgument:
    def test_the_spread_is_measured_over_hypotheses_with_no_model_in_them(self, report):
        """What the conclusion rests on, and why it survives disagreement about the
        protocol's details."""
        values = [s.meteor for s in report.scores if s.protocol == "prior"]
        assert report.floor_spread == pytest.approx(max(values) - min(values))

    def test_the_reported_improvement_does_not_clear_the_spread(self, report):
        """The prior work's contribution is smaller than the metric's own range over text
        that contains no language modelling at all."""
        assert report.claimed_improvement < report.floor_spread
        assert report.improvement_exceeds_floor_spread is False

    def test_every_protocol_is_scored_for_every_hypothesis(self, report):
        """A missing cell would let a comparison be made against a column that was never
        computed, which is the shape of the error this module exists to correct."""
        assert len(report.scores) == len(PROTOCOLS) * 4


class TestWhatCouldGoWrongSilently:
    def test_unrelated_prose_is_never_the_rows_own_reference(self):
        """The first version of this hypothesis walked the pool in step with the rows and
        handed each row its own reference back, scoring 0.9999 and measuring the indexing.
        A floor that silently becomes a ceiling is worse than no floor."""
        pool = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]
        for index, reference in enumerate(pool):
            case = Case(prompt=("x",), reference=reference, kind="prose")
            assert hypothesis_unrelated_prose(case, pool=pool, index=index) != reference

    def test_the_empty_figure_moves_with_the_cut_and_is_reported_with_it(self):
        """The published decimal is a function of a parameter the published text does not
        repeat beside it. The table is what keeps a reader from quoting the number alone."""
        table = sensitivity(ROWS, fractions=(0.5, 0.7, 0.8))
        assert [row["prompt_fraction"] for row in table] == [0.5, 0.7, 0.8]
        scores = [row["empty_generation"] for row in table]

        # Not monotone, and the test does not ask it to be. METEOR balances precision
        # against recall, so lengthening the hypothesis while shortening the reference can
        # move the mean either way; an earlier version asserted an ordering the metric never
        # promised and failed on the first corpus it met.
        assert max(scores) - min(scores) > 0.01, (
            "the published decimal is quoted without the cut it was measured at, which is "
            "defensible only if the cut barely matters; this table is what shows it does")
        assert min(scores) > 0.0, (
            "no setting makes writing nothing score zero under the prior protocol, which "
            "is the claim that does not depend on recovering the exact parameters")

    def test_the_markup_share_is_reported_because_it_carries_the_effect(self, report):
        """A quarter of the evaluated rows are section headers whose reference is markup
        split mid-word. The share is what turns a per-row artefact into a corpus mean."""
        assert 0.0 < report.markup_share < 1.0
        markup = report.score("empty", "prior").by_kind.get("markup", 0.0)
        assert markup > report.score("empty", "prior").meteor, (
            "the headers must score above the corpus mean; if they do not, the stratum is "
            "not what is inflating the reported figure")
