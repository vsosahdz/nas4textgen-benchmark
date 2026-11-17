"""The reference check: what it compares, and what it refuses to decide.

The risk in this module is not that it fails — it is that it succeeds wrongly. A DOI
proposed from a near-miss title resolves to somebody else's paper, and a citation that points
confidently at the wrong work is worse than one with a gap in it. So the thresholds and the
refusals are what these tests pin.

No network. The registries are stubbed, because a test that needs the internet is one that
gets skipped on the machine where it would have mattered.
"""

from __future__ import annotations

import pytest

from nas4textgen.overleaf import Reference
from nas4textgen.references import (
    CONFIDENT,
    PLAUSIBLE,
    Candidate,
    check,
    is_preprint,
    normalise,
    report,
    similarity,
)


def entry(**fields) -> Reference:
    base = {"author": "Church, K. and Rau, L.", "title": "A Paper", "year": "1995"}
    base.update(fields)
    return Reference(key="Key1995", number=1, raw=base["title"],
                     entry_type="article", fields=base)


class TestComparingTitles:
    def test_brace_protection_is_not_a_difference(self):
        """Nearly every title in the carried list protects its capitalisation with braces.
        Comparing them literally makes a correct entry look like a mismatch."""
        assert normalise("Neural {Architecture} {Search}: {A} survey") == (
            "neural architecture search a survey")

    def test_case_and_punctuation_are_not_differences(self):
        assert similarity("Commercial Applications of NLP.",
                          "commercial applications of nlp") == pytest.approx(1.0)

    def test_a_different_paper_does_not_score_as_the_same_one(self):
        assert similarity("Neural Architecture Search: A Survey",
                          "Neural Architecture Search and Hardware Accelerator "
                          "Co-Search: A Survey") < CONFIDENT


class TestWhatItProposes:
    def _checked(self, monkeypatch, candidate, our_fields):
        import nas4textgen.references as module

        monkeypatch.setattr(module, "search_crossref", lambda *a, **k: candidate)
        monkeypatch.setattr(module, "search_arxiv", lambda *a, **k: None)
        monkeypatch.setattr(module, "search_openalex", lambda *a, **k: None)
        return check(entry(**our_fields), pause=0)

    def test_a_field_we_lack_is_proposed(self, monkeypatch):
        found = self._checked(
            monkeypatch,
            Candidate("crossref", "A Paper", 1.0, {"doi": "10.1/x", "volume": "38"}),
            {"title": "A Paper"})
        assert found.status == "matched"
        assert found.additions == {"doi": "10.1/x", "volume": "38"}

    def test_a_field_we_have_and_the_record_contradicts_is_a_conflict_not_an_edit(
            self, monkeypatch):
        """The case that found a Communications of the ACM paper filed under 'Advances in
        Traditional AI'. Overwriting silently would hide the error rather than report it."""
        found = self._checked(
            monkeypatch,
            Candidate("crossref", "A Paper", 1.0,
                      {"journal": "Communications of the ACM"}),
            {"title": "A Paper", "journal": "Advances in Traditional AI"})
        assert found.status == "conflict"
        assert found.conflicts["journal"] == (
            "Advances in Traditional AI", "Communications of the ACM")
        assert "journal" not in found.additions

    def test_a_formatting_difference_is_not_a_conflict(self, monkeypatch):
        found = self._checked(
            monkeypatch,
            Candidate("crossref", "A Paper", 1.0, {"journal": "Communications of the ACM"}),
            {"title": "A Paper", "journal": "communications of the ACM."})
        assert found.conflicts == {}


class TestWhatItRefusesToDecide:
    def _checked(self, monkeypatch, candidate):
        import nas4textgen.references as module

        for name in ("search_crossref", "search_arxiv", "search_openalex"):
            monkeypatch.setattr(module, name, lambda *a, **k: candidate)
        return check(entry(title="A Paper"), pause=0)

    def test_a_plausible_match_is_handed_to_a_human_not_applied(self, monkeypatch):
        """Between the thresholds the tool has an opinion and no authority. Applying a DOI
        from a 0.81 title match points the citation at a different paper."""
        middle = (PLAUSIBLE + CONFIDENT) / 2
        found = self._checked(
            monkeypatch, Candidate("crossref", "A Different Paper", middle,
                                   {"doi": "10.1/wrong"}))
        assert found.status == "needs a human"
        assert found.additions == {}

    def test_a_poor_match_is_reported_as_nothing_found(self, monkeypatch):
        found = self._checked(
            monkeypatch, Candidate("crossref", "Unrelated", 0.2, {"doi": "10.1/x"}))
        assert found.status == "not found"
        assert found.candidate is None

    def test_an_unreachable_registry_is_not_a_missing_record(self, monkeypatch):
        """Reported as unreachable rather than not found: one says the network failed and
        the other says the work is not in the registry, and conflating them turns an
        outage into a finding about the bibliography."""
        import nas4textgen.references as module

        def fail(*a, **k):
            raise module.LookupUnavailable("no route to host")

        for name in ("search_crossref", "search_arxiv", "search_openalex"):
            monkeypatch.setattr(module, name, fail)
        assert check(entry(title="A Paper"), pause=0).status == "unreachable"

    def test_nothing_in_the_report_claims_to_have_edited_anything(self):
        payload = report([])
        assert "Nothing here edits the bibliography" in payload["what_this_is_not"]


class TestRoutingToTheRightRegistry:
    def test_a_preprint_goes_to_arxiv(self):
        assert is_preprint(entry(journal="arXiv Preprint"))
        assert is_preprint(entry(journal="CoRR"))

    def test_a_journal_article_does_not(self):
        assert not is_preprint(entry(journal="Communications of the ACM"))

    def test_the_thresholds_are_declared_constants(self):
        """Chosen before the entries were checked, so that an entry cannot be admitted by
        moving the line it failed."""
        assert CONFIDENT == 0.92
        assert PLAUSIBLE == 0.75
        assert PLAUSIBLE < CONFIDENT


class TestApplyingWhatWasFound:
    """Folding proposals back in. The rule is that gaps get filled and disagreements do
    not, because overwriting a contradicted field deletes the evidence of the contradiction
    — which is how a Communications of the ACM paper sat under another journal's name."""

    def _entry(self, **fields):
        base = {"author": "A, B.", "title": "T", "journal": "J", "year": "2000"}
        base.update(fields)
        return Reference(key="K", number=1, raw="T", entry_type="article", fields=base)

    def _finding(self, **kwargs):
        from nas4textgen.references import Finding

        kwargs.setdefault("key", "K")
        kwargs.setdefault("title", "T")
        return Finding(**kwargs)

    def test_a_gap_the_registry_closed_is_filled(self):
        from nas4textgen.references import apply

        entries = [self._entry()]
        findings = [self._finding(status="matched",
                                  additions={"doi": "10.1/x", "volume": "9",
                                             "pages": "1--10"})]
        updated = apply(entries, findings)[0]
        assert updated.fields["doi"] == "10.1/x"
        assert updated.missing == ()
        assert updated.verified is True

    def test_a_contradicted_field_is_left_exactly_as_it_was(self):
        from nas4textgen.references import apply

        entries = [self._entry(journal="Advances in Traditional AI")]
        findings = [self._finding(
            status="conflict",
            conflicts={"journal": ("Advances in Traditional AI",
                                   "Communications of the ACM")},
            additions={"doi": "10.1/x", "volume": "38", "pages": "71--79"})]
        updated = apply(entries, findings)[0]
        assert updated.fields["journal"] == "Advances in Traditional AI"
        assert updated.fields["doi"] == "10.1/x"
        assert updated.verified is False, "a disagreement is not a verified entry"

    def test_the_marker_survives_an_entry_that_is_merely_complete(self):
        """Complete is not correct. The first entry of the prior list had every field a
        parser checks and four factual errors in them."""
        from nas4textgen.overleaf import UNVERIFIED
        from nas4textgen.references import apply

        entries = [self._entry(journal="Wrong Journal")]
        findings = [self._finding(status="conflict",
                                  conflicts={"journal": ("Wrong Journal", "Right One")},
                                  additions={"doi": "10.1/x", "volume": "1",
                                             "pages": "1--2"})]
        assert UNVERIFIED in apply(entries, findings)[0].as_bibtex()

    def test_an_unresolved_entry_is_passed_through_untouched(self):
        from nas4textgen.references import apply

        entries = [self._entry()]
        for status in ("not found", "needs a human", "unreachable"):
            updated = apply(entries, [self._finding(status=status)])[0]
            assert updated.fields == entries[0].fields, status
            assert updated.verified is False, status

    def test_the_rendered_file_keeps_citation_order(self):
        from nas4textgen.references import render_bibliography

        entries = [
            Reference(key=f"K{n}", number=n, raw="T", entry_type="article",
                      fields={"title": "T"})
            for n in (1, 2, 3)]
        text = render_bibliography(entries)
        assert text.index("@article{K1,") < text.index("@article{K2,")
        assert text.index("@article{K2,") < text.index("@article{K3,")


class TestOneRegistryFailingIsNotAVerdict:
    """A Crossref outage reported six entries as unreachable without ever asking the other
    registries — and OpenAlex matched all six at 1.00. An entry is only unreachable when
    every registry failed; anything else is a statement about the network, not the entry."""

    def _entry(self):
        return Reference(key="K", number=1, raw="T", entry_type="article",
                         fields={"title": "T", "author": "A, B.", "journal": "J",
                                 "year": "2000", "volume": "1", "pages": "1--2",
                                 "doi": "10.1/x"})

    def test_a_later_registry_answers_when_the_first_one_fails(self, monkeypatch):
        import nas4textgen.references as module

        def down(*a, **k):
            raise module.LookupUnavailable("down")

        monkeypatch.setattr(module, "search_crossref", down)
        monkeypatch.setattr(module, "search_arxiv", down)
        monkeypatch.setattr(module, "search_openalex",
                            lambda *a, **k: Candidate("openalex", "T", 1.0, {}))
        assert check(self._entry(), pause=0).status == "matched"

    def test_unreachable_needs_every_registry_to_fail(self, monkeypatch):
        import nas4textgen.references as module

        def down(*a, **k):
            raise module.LookupUnavailable("down")

        for name in ("search_crossref", "search_arxiv", "search_openalex"):
            monkeypatch.setattr(module, name, down)
        assert check(self._entry(), pause=0).status == "unreachable"

    def test_a_confident_match_stops_the_search(self, monkeypatch):
        """Three registries per entry is three times the requests. Once one of them is
        certain, asking the rest spends somebody else's rate limit for nothing."""
        import nas4textgen.references as module

        calls = []

        def crossref(*a, **k):
            calls.append("crossref")
            return Candidate("crossref", "T", 1.0, {})

        def openalex(*a, **k):
            calls.append("openalex")
            return Candidate("openalex", "T", 1.0, {})

        monkeypatch.setattr(module, "search_crossref", crossref)
        monkeypatch.setattr(module, "search_arxiv", lambda *a, **k: None)
        monkeypatch.setattr(module, "search_openalex", openalex)
        check(self._entry(), pause=0)
        assert calls == ["crossref"]


class TestAuthorNamesAreRepaired:
    """BibTeX reads `Parmar N.` as a given name with no surname and prints it where the
    family name belongs. Two carried entries are written that way, one of them the
    transformer paper, and the cost is a reference nobody can look up by author."""

    def _entry(self, authors):
        return Reference(key="K", number=1, raw="T", entry_type="article",
                         fields={"author": authors, "title": "T",
                                 "journal": "J", "year": "2017"})

    def test_a_missing_comma_is_restored(self):
        from nas4textgen.references import repair_authors

        fixed = repair_authors([self._entry("Vaswani, A. and Parmar N. and Jones, L.")])
        assert fixed[0].fields["author"] == (
            "Vaswani, A. and Parmar, N. and Jones, L.")

    def test_a_correct_name_is_left_alone(self):
        from nas4textgen.references import repair_authors

        authors = "Deb, K. and Pratap, A."
        assert repair_authors([self._entry(authors)])[0].fields["author"] == authors

    def test_a_full_given_name_is_not_mistaken_for_an_initial(self):
        """`Ronen Eldan` is a complete name in the other order, not a surname missing its
        comma. Inserting one would invert it."""
        from nas4textgen.references import repair_authors

        authors = "Ronen Eldan and Yuanzhi Li"
        assert repair_authors([self._entry(authors)])[0].fields["author"] == authors

    def test_an_entry_with_no_author_is_untouched(self):
        from nas4textgen.references import repair_authors

        entry = Reference(key="K", number=1, raw="T", entry_type="article",
                          fields={"title": "T"})
        assert repair_authors([entry])[0].fields == {"title": "T"}
