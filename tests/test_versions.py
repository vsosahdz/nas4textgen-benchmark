"""Numbered versions of the document, kept so an earlier one survives the next rebuild.

The build folder is regenerated in place, so the only way to still have v01 after building
v02 is to have copied it. These tests hold the copy to being complete, numbered without
gaps, and described well enough that the question asked of a version months later, which
one the reviewer saw, can be answered from the folder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nas4textgen.overleaf import TRANSIENT, next_version, snapshot

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT / "docs" / "manuscript" / "versions"


@pytest.fixture
def built(tmp_path):
    """A build folder with the files a compile leaves behind."""
    folder = tmp_path / "build"
    (folder / "tables").mkdir(parents=True)
    (folder / "main.tex").write_text("\\documentclass{elsarticle}", encoding="utf-8")
    (folder / "main.pdf").write_bytes(b"%PDF-1.5\n")
    (folder / "references.bib").write_text("@article{a,}", encoding="utf-8")
    (folder / "tables" / "table1.tex").write_text("\\begin{table}", encoding="utf-8")
    (folder / "main.log").write_text("rewritten by every compile", encoding="utf-8")
    (folder / "main.blg").write_text("rewritten by every compile", encoding="utf-8")
    return folder


class TestWhatAVersionKeeps:
    def test_the_compiled_document_is_kept(self, tmp_path, built):
        """The PDF is the thing the authors circulate and then refer to by name."""
        target = snapshot(tmp_path, built, label="v01")
        assert (target / "main.pdf").exists()
        assert (target / "main.tex").exists()
        assert (target / "tables" / "table1.tex").exists()

    def test_the_log_files_are_not(self, tmp_path, built):
        """They are rewritten by every compile and say nothing about the version."""
        target = snapshot(tmp_path, built, label="v01")
        assert not (target / "main.log").exists()
        assert not (target / "main.blg").exists()

    def test_every_transient_pattern_is_actually_excluded(self, tmp_path, built):
        for pattern in TRANSIENT:
            (built / pattern.replace("*", "main")).write_text("x", encoding="utf-8")
        target = snapshot(tmp_path, built, label="v01")
        left = [p.name for p in target.rglob("*")
                if p.suffix in {".log", ".blg", ".aux", ".out", ".bbl", ".toc"}]
        assert left == []

    def test_a_note_is_written_beside_the_copy(self, tmp_path, built):
        """A folder named v03 records that there was a third version and nothing about
        what changed in it."""
        target = snapshot(tmp_path, built, label="v02", note="Added the equations.")
        text = (target / "VERSION.md").read_text(encoding="utf-8")
        assert "# v02" in text
        assert "Added the equations." in text
        assert re.search(r"kept: \d{4}-\d{2}-\d{2}", text)

    def test_the_note_says_when_no_pdf_was_there_to_keep(self, tmp_path, built):
        """Silence here would let a sources-only folder pass for the document."""
        (built / "main.pdf").unlink()
        target = snapshot(tmp_path, built, label="v01")
        assert "compiled: no" in (target / "VERSION.md").read_text(encoding="utf-8")


class TestNumbering:
    def test_the_first_version_is_v01(self, tmp_path):
        assert next_version(tmp_path) == "v01"

    def test_each_version_follows_the_highest_kept(self, tmp_path, built):
        snapshot(tmp_path, built, label="v01")
        assert next_version(tmp_path) == "v02"
        snapshot(tmp_path, built, label="v02")
        assert next_version(tmp_path) == "v03"

    def test_a_gap_does_not_reuse_a_number(self, tmp_path, built):
        """Deleting v02 must not make the next build v02 again: two different documents
        under one label is worse than a gap in the sequence."""
        snapshot(tmp_path, built, label="v01")
        snapshot(tmp_path, built, label="v03")
        assert next_version(tmp_path) == "v04"

    def test_an_existing_version_is_never_overwritten(self, tmp_path, built):
        snapshot(tmp_path, built, label="v01")
        with pytest.raises(FileExistsError):
            snapshot(tmp_path, built, label="v01")


class TestTheVersionsThisProjectHasKept:
    def test_they_are_numbered_without_a_gap(self):
        if not VERSIONS.exists():
            pytest.skip("no versions kept yet")
        kept = sorted(int(m.group(1)) for d in VERSIONS.iterdir()
                      if d.is_dir() and (m := re.fullmatch(r"v(\d+)", d.name)))
        assert kept == list(range(1, len(kept) + 1)), kept

    def test_each_one_says_what_it_is(self):
        if not VERSIONS.exists():
            pytest.skip("no versions kept yet")
        for folder in sorted(VERSIONS.iterdir()):
            if folder.is_dir():
                assert (folder / "VERSION.md").exists(), f"{folder.name} has no note"
                assert (folder / "main.tex").exists(), f"{folder.name} has no document"
