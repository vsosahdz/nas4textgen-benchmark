"""Every file this project writes is UTF-8, and every number uses a decimal point.

Neither is cosmetic. ``Path.write_text()`` without an encoding uses whatever
``locale.getpreferredencoding()` returns — cp1252 on Windows, occasionally ASCII on a
minimal Linux image — and the corpora carry accented characters, em-dashes and quotation
marks that the JSON writer emits raw because ensure_ascii is off. A reader running the
replication package under a different locale would get mojibake in the manifests or a
UnicodeEncodeError partway through a sweep, and neither failure points at its cause.

The decimal separator matters for the same reason in the other direction: a table read as
"1.650" under one convention and 1.65 under another is not a table anyone can check.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"

#: Tests write files too, and a test that only holds the package to this standard would
#: pass on a machine where the suite itself cannot run.
AUDITED = (ROOT / "src", ROOT / "tests")

#: Text that survives a UTF-8 round trip and fails under cp1252 or ASCII. Drawn from what
#: the corpora actually contain rather than invented.
AWKWARD = "Ceará — «naïve» 数学   emoji 🙂 ratio ½"


def _text_io_calls_without_encoding() -> list[str]:
    """Every read_text/write_text/open in the package that does not name an encoding."""
    offenders = []
    for root in AUDITED:
      for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in ("read_text", "write_text", "open"):
                continue
            # Binary mode needs no encoding, and naming one there is an error.
            if any(
                isinstance(a, ast.Constant) and isinstance(a.value, str) and "b" in a.value
                for a in node.args
            ):
                continue
            if not any(kw.arg == "encoding" for kw in node.keywords):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}()")
    return offenders


def test_no_text_file_operation_relies_on_the_system_locale():
    offenders = _text_io_calls_without_encoding()
    assert not offenders, (
        "these read or write text without naming an encoding, so they inherit the "
        "system locale:\n  " + "\n  ".join(offenders)
    )


def test_the_corpus_cache_round_trips_non_ascii(tmp_path, monkeypatch):
    monkeypatch.setenv("NAS4TG_CACHE", str(tmp_path))
    from nas4textgen.corpora.cache import load, store

    store("wikitext103", "validation", [AWKWARD])
    rows, record = load("wikitext103", "validation")
    assert rows == [AWKWARD]

    # And the sidecar itself, which is where the missing encoding actually was.
    meta = tmp_path / "wikitext103" / "validation.meta.json"
    assert json.loads(meta.read_text(encoding="utf-8"))["sha256"] == record.sha256


def test_artifacts_are_written_as_utf8(tmp_path, monkeypatch):
    monkeypatch.setenv("NAS4TG_ARTIFACTS", str(tmp_path))
    from nas4textgen.artifacts import write_json, write_table

    path = write_json("t/x.json", {"note": AWKWARD})
    assert json.loads(path.read_text(encoding="utf-8"))["note"] == AWKWARD

    table = write_table("t/y.csv", [{"a": AWKWARD, "b": 1.5}], ["a", "b"])
    assert AWKWARD.split()[0] in table.read_text(encoding="utf-8")


def test_numbers_are_formatted_with_a_decimal_point():
    """Python's format spec is locale-independent unless the 'n' type is used.

    Asserted rather than assumed, because a single ``:n`` anywhere would render 1.5 as
    "1,5" on a machine with a European locale and quietly make a results table
    unreadable next to the ones produced elsewhere.
    """
    assert f"{1234.5:,.2f}" == "1,234.50"
    assert f"{0.85:.2f}" == "0.85"

    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert ":n}" not in text, f"{path.name} uses locale-dependent number formatting"
        assert "import locale" not in text, f"{path.name} imports locale"


@pytest.mark.parametrize("codec", ["utf-8"])
def test_every_source_file_is_itself_utf8(codec):
    for path in sorted(SOURCE.rglob("*.py")):
        path.read_bytes().decode(codec)
