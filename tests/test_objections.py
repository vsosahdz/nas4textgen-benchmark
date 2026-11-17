"""What the prior submission was told, and where this work answers it.

A resubmission that says the reviewers' comments were addressed without saying where is
asking the same reviewers to find the answers. These tests hold the registry to citing
evidence that exists, to quoting words the reviewers actually wrote, and to counting what is
still open rather than rounding it up to done.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nas4textgen.objections import (
    DISPOSITIONS,
    RECORDED,
    REVIEW_TEXT,
    EvidenceMissing,
    Objection,
    QuoteNotFound,
    RegistryIncomplete,
    open_objections,
    report,
    require_complete,
    require_evidence,
    require_quotes,
    verify_evidence,
    verify_quotes,
)

#: The repository root, for checking whether the reviews are in this checkout.
ROOT = Path(__file__).resolve().parents[1]

#: Every reviewer item in the review text. Reviewer 2 numbered
#: theirs 1 to 12; reviewer 1 wrote one comment; reviewer 3 wrote fourteen paragraphs, each
#: a distinct objection. Written down here so dropping one from the registry fails a test
#: rather than passing quietly with a smaller total.
EXPECTED_COUNTS = {"R1": 1, "R2": 12, "R3": 14}


class TestAnEntry:
    def test_an_objection_without_evidence_is_refused(self):
        """Saying an objection is answered without saying where asks the reviewer to find
        the answer."""
        with pytest.raises(ValueError, match="no evidence cited"):
            Objection(reviewer="R1", number="1", quote="a quote long enough to identify it",
                      summary="x", disposition="addressed", response="we did it", evidence=())

    def test_a_declined_objection_must_say_when_it_would_apply(self):
        """'It does not apply' is an assertion; 'it applies again the moment Lex-MODES
        returns to the slate' is an argument a reviewer can check."""
        with pytest.raises(ValueError, match="condition under which"):
            Objection(reviewer="R1", number="1", quote="a quote long enough to identify it",
                      summary="x", disposition="declined", response="it does not apply here",
                      evidence=("README.md",))

    def test_the_disposition_must_be_one_of_the_three(self):
        with pytest.raises(ValueError, match="disposition must be one of"):
            Objection(reviewer="R1", number="1", quote="a quote long enough to identify it",
                      summary="x", disposition="noted", response="mm", evidence=("README.md",))

    def test_a_fragment_too_short_to_identify_an_objection_is_refused(self):
        """A registry keyed to five words can be pointed at the wrong objection."""
        with pytest.raises(ValueError, match="too short"):
            Objection(reviewer="R1", number="1", quote="add more", summary="x",
                      disposition="addressed", response="done", evidence=("README.md",))


class TestCoverage:
    def test_every_objection_in_every_review_is_recorded(self):
        counts: dict[str, int] = {}
        for objection in RECORDED:
            counts[objection.reviewer] = counts.get(objection.reviewer, 0) + 1
        assert counts == EXPECTED_COUNTS
        assert report()["total"] == sum(EXPECTED_COUNTS.values())

    def test_no_objection_is_recorded_twice(self):
        keys = [(o.reviewer, o.number) for o in RECORDED]
        assert len(keys) == len(set(keys))

    def test_reviewer_two_is_numbered_as_they_numbered_it(self):
        """Reviewer 2 wrote a numbered list, so the registry uses their numbers: an answer
        filed under a number the reviewer did not use is an answer they have to hunt for."""
        numbers = sorted(int(o.number) for o in RECORDED if o.reviewer == "R2")
        assert numbers == list(range(1, 13))

    def test_every_response_argues_rather_than_asserts(self):
        for objection in RECORDED:
            assert len(objection.response) > 200, (
                f"{objection.reviewer} #{objection.number} answers in one line"
            )
            assert objection.summary


class TestEvidence:
    def test_every_citation_points_at_a_real_file(self):
        """A citation to a moved file reads as diligence and is not."""
        assert verify_evidence() == []
        require_evidence()

    def test_a_rotted_citation_is_caught(self, tmp_path):
        assert verify_evidence(tmp_path), "citations cannot all exist in an empty directory"
        with pytest.raises(EvidenceMissing, match="reads as diligence"):
            require_evidence(tmp_path)


class TestQuotes:
    def test_every_quote_is_verbatim_from_the_reviews(self):
        """The registry once reconstructed the objections from the code's own passing
        references, because the review text lived only in a conversation. Quoting the file
        is what stops an answer drifting onto a question nobody asked.

        Skipped where the reviews are absent. They are referee reports on an earlier
        submission — third-party writing addressed to an editor — and are kept locally
        rather than committed, so this check runs for whoever has them.
        """
        from nas4textgen.objections import ReviewsUnavailable

        try:
            assert verify_quotes() == []
            require_quotes()
        except ReviewsUnavailable as exc:
            pytest.skip(str(exc))

    def test_the_absence_of_the_reviews_is_reported_not_crashed_on(self, tmp_path):
        """A missing file must say it is missing. Raising FileNotFoundError from inside
        the registry reads as a defect in the registry rather than a checkout without the
        reviews in it."""
        from nas4textgen.objections import ReviewsUnavailable

        with pytest.raises(ReviewsUnavailable, match="kept locally"):
            verify_quotes(tmp_path)

    def test_an_invented_quote_is_caught(self, tmp_path):
        (tmp_path / REVIEW_TEXT).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / REVIEW_TEXT).write_text("nothing the registry quotes", encoding="utf-8")
        assert verify_quotes(tmp_path)
        with pytest.raises(QuoteNotFound, match="not having read them"):
            require_quotes(tmp_path)


class TestWhatIsStillOpen:
    def test_certifying_the_set_fails_while_anything_is_open(self):
        """The claim that every objection is answered is not true yet, and the registry
        should say so rather than letting a cover letter say otherwise."""
        assert open_objections()
        with pytest.raises(RegistryIncomplete, match="still open"):
            require_complete()

    def test_the_open_ones_wait_on_measurement_or_on_the_manuscript(self):
        """Nothing is pending for want of a decision — every open entry has its mechanism
        built and is waiting for a number or for prose."""
        for objection in open_objections():
            assert objection.evidence
            assert objection.response

    def test_the_dispositions_partition_the_registry(self):
        counted = sum(
            sum(o.disposition == name for o in RECORDED) for name in DISPOSITIONS
        )
        assert counted == len(RECORDED)

    def test_the_report_carries_its_own_gaps(self):
        payload = report()
        assert payload["complete"] is False
        assert payload["pending"] == len(open_objections())
        assert payload["broken_citations"] == []
        assert payload["unverified_quotes"] == []
        # Says whether it could check, rather than implying it did. Without the reviews an
        # empty list of unverified quotes would mean "none checked", not "all verified".
        assert payload["quotes_checked"] is (ROOT / REVIEW_TEXT).exists()
        assert "cover letter" in payload["why_open_is_reported"]
