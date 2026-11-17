

class TestOrcidIsCheckedNotJustPresent:
    """An ORCID with one digit wrong is a well-formed identifier belonging to somebody
    else, and every index that ingests the release attributes the work to them. The
    standard provides a check digit for exactly this."""

    def test_the_recorded_orcids_pass_their_own_checksum(self):
        from nas4textgen.authorship import authors, require_orcid

        for author in authors():
            require_orcid(author.fields.get("orcid", ""))

    def test_one_digit_wrong_is_refused(self):
        import pytest

        from nas4textgen.authorship import OrcidInvalid, require_orcid

        with pytest.raises(OrcidInvalid):
            require_orcid("https://orcid.org/0000-0002-1099-8149")

    def test_the_x_check_digit_is_accepted(self):
        """ORCID uses X for a check value of ten. Rejecting it would refuse one identifier
        in eleven, and the ones it refuses are valid."""
        from nas4textgen.authorship import check_digit

        assert check_digit("0000-0002-1694-233X") == "X"

    def test_an_author_with_no_orcid_is_not_an_incomplete_author(self):
        """Not every researcher holds one. A rule that cannot tell `NONE` from a blank
        either blocks the release forever or gets switched off, and switched off it stops
        catching the blank it exists for."""
        from nas4textgen.authorship import DECLARED_ABSENT, require_orcid

        require_orcid(DECLARED_ABSENT)

    def test_a_blank_orcid_still_blocks_a_release(self):
        from nas4textgen.authorship import Author

        blank = Author(position=1, fields={"given-names": "A", "family-names": "B",
                                           "affiliation": "C", "orcid": "   "})
        assert "orcid" in blank.missing

    def test_the_release_gate_passes_now_that_the_authors_answered(self):
        from nas4textgen.authorship import require_complete

        require_complete()


class TestTheChecksumManifest:
    """A tag is only a citable artifact if a reader can tell the bytes have not moved."""

    def test_the_manifest_is_not_ignored_by_git(self):
        """The whole point is that the tag carries it. The artifacts directory is ignored
        wholesale, and a negation inside a directory ignored with a trailing slash has no
        effect because git never descends into it — so this was ignored at first."""
        import subprocess
        from pathlib import Path

        from nas4textgen.release import MANIFEST

        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(["git", "check-ignore", MANIFEST],
                                cwd=root, capture_output=True, text=True, check=False)
        assert result.returncode != 0, f"{MANIFEST} is ignored and would not reach the tag"

    def test_the_manifest_verifies_against_the_files_it_lists(self, tmp_path):
        from nas4textgen.release import collect, digest, render

        (tmp_path / "artifacts" / "benchmark").mkdir(parents=True)
        target = tmp_path / "artifacts" / "benchmark" / "table.csv"
        target.write_text("a,b\n1,2\n", encoding="utf-8")

        entries, absent = collect(tmp_path, ("artifacts/benchmark/table.csv",))
        assert absent == []
        assert entries[0].sha256 == digest(target)
        assert entries[0].sha256 in render(entries, absent)

    def test_a_declared_path_that_does_not_exist_is_named(self, tmp_path):
        """An artifact that silently stopped being produced is what a manifest is for."""
        from nas4textgen.release import collect

        _, absent = collect(tmp_path, ("artifacts/benchmark/table.csv",))
        assert absent == ["artifacts/benchmark/table.csv"]

    def test_nothing_to_publish_is_refused_rather_than_written_empty(self, tmp_path):
        import pytest

        from nas4textgen.release import NothingToPublish, write

        with pytest.raises(NothingToPublish):
            write(tmp_path)

    def test_the_format_is_the_one_the_documented_command_reads(self, tmp_path):
        """`sha256sum -c` wants `<digest>  <path>` and warns on a blank line, and a
        verification that prints warnings is one a reader stops reading."""
        from nas4textgen.release import collect, render

        (tmp_path / "artifacts" / "benchmark").mkdir(parents=True)
        (tmp_path / "artifacts" / "benchmark" / "table.csv").write_text("x", encoding="utf-8")
        text = render(*collect(tmp_path, ("artifacts/benchmark/table.csv",)))
        body = [line for line in text.splitlines() if not line.startswith("#")]
        assert all(line.strip() for line in body), "a blank line would be reported"
        for line in body:
            digest_part, _, path = line.partition("  ")
            assert len(digest_part) == 64 and path
