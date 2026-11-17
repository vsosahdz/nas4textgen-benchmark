"""Tables generated from artifacts, never transcribed.

A number typed into a manuscript is a number that can disagree with the artifact it came
from, and by the time anyone notices the paper is submitted. These tests hold the generator
to that: it reads, it refuses to invent, and it says which command writes what it is missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nas4textgen.manuscript import (
    TABLES,
    MissingArtifact,
    generate,
    render_all,
    table_search_space,
)


@pytest.fixture
def artifacts(tmp_path):
    """A minimal artifact directory: enough for one table, missing the rest."""
    (tmp_path / "space").mkdir()
    (tmp_path / "space" / "enumeration_summary.json").write_text(json.dumps({
        "cardinality": 1099, "total_before_ceiling": 1344, "excluded": 245,
        "excluded_fraction": 0.1823, "parameter_ceiling": 150_000_000,
        "cost_front_size": 24, "params_range": [3_340_736, 144_892_416],
        "kv_bytes_range": [262_144, 50_331_648],
    }), encoding="utf-8")
    return tmp_path


class TestGeneration:
    def test_a_table_is_built_from_the_artifact_it_cites(self, artifacts):
        table = table_search_space(artifacts)
        values = {row["quantity"]: row["across the space"] for row in table.rows}
        assert values["architectures measured"] == "1,099"
        assert values["removed by the parameter ceiling of 150,000,000"] == "245 (18.2%)"
        assert "enumeration_summary.json" in table.source

    def test_the_anchor_column_holds_the_architecture_the_space_is_named_for(
            self, artifacts):
        """The claim the table exists to support.

        A space described only by its cardinality leaves the reader to take on trust that
        it contains the architecture it says it derives from. These are the anchor's own
        values, and the parameter count is the published checkpoint's to the parameter.
        """
        anchor = {row["quantity"]: row["DistilGPT2"]
                  for row in table_search_space(artifacts).rows}
        assert anchor["$h$, query heads"] == "12"
        assert anchor[r"$d_{\mathrm{head}}$, head dimension"] == "64"
        assert anchor["$n$, transformer blocks"] == "6"
        assert anchor[r"$\mathit{ffn}$, feed-forward ratio"] == "4"
        assert anchor["$k_v$, key-value heads, dividing $h$"] == "12"
        assert anchor[r"$e = h \cdot d_{\mathrm{head}}$, model width (derived)"] == "768"
        assert anchor["parameters"] == "81,912,576"

    def test_the_cardinality_rows_leave_the_anchor_column_empty(self, artifacts):
        """A count of the whole space is not a property of one architecture in it, and a
        number repeated into that column would read as one."""
        rows = {row["quantity"]: row["DistilGPT2"]
                for row in table_search_space(artifacts).rows}
        assert rows["grid before any constraint"] == ""
        assert rows["architectures measured"] == ""

    def test_a_missing_artifact_names_the_command_that_writes_it(self, tmp_path):
        """A gap left blank in a manuscript is worse than a build that stops."""
        with pytest.raises(MissingArtifact, match="nas4textgen.cli enumerate"):
            table_search_space(tmp_path)

    def test_partial_generation_reports_what_is_absent(self, artifacts):
        """During a campaign most artifacts do not exist yet, and a command that refused to
        produce anything until they all did would be useless for the months that matter.

        Two build from this fixture. The search space needs only the enumeration, and the
        replication tiers read no artifact at all, since a tier declares what someone
        else's machine will need rather than measuring anything on ours.
        """
        tables, missing = generate(artifacts)
        assert {table.number for table in tables} == {"1", "12"}
        assert len(missing) == len(TABLES) - len(tables)
        assert all(":" in entry for entry in missing), "each says which table and why"

    def test_the_document_lists_what_it_could_not_build(self, artifacts):
        tables, missing = generate(artifacts)
        rendered = render_all(tables, missing)
        assert "Not yet available" in rendered
        assert "nas4textgen.cli" in rendered, "and how to produce it"

    def test_nothing_is_transcribed(self, artifacts):
        """Changing the artifact must change the table."""
        before = table_search_space(artifacts).rows

        path = artifacts / "space" / "enumeration_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cardinality"] = 777
        path.write_text(json.dumps(payload), encoding="utf-8")

        after = table_search_space(artifacts).rows
        assert before != after
        assert any("777" in str(row["across the space"]) for row in after)


class TestRendering:
    def test_a_table_carries_its_source(self, artifacts):
        rendered = table_search_space(artifacts).render()
        assert "*Source:" in rendered
        assert rendered.startswith("**Table 1.**")

    def test_the_typeset_table_does_not_print_the_artifact_path(self, artifacts):
        """Provenance belongs to whoever regenerates the table. A reader of the journal
        cannot open `artifacts/space/enumeration_summary.json`, and a line of repository
        paths under every table is noise on the page."""
        rendered = table_search_space(artifacts).render_latex()
        assert "Source:" not in rendered
        assert "artifacts/" not in rendered
        assert ".json" not in rendered

    def test_the_markdown_keeps_the_provenance_the_typeset_table_drops(self, artifacts):
        """The two renderings have different readers. The Markdown is where a co-author
        checks a number against the artifact it came from."""
        table = table_search_space(artifacts)
        assert "enumeration_summary.json" in table.render()
        assert "enumeration_summary.json" not in table.render_latex()

    def test_a_coverage_note_survives_into_the_typeset_table(self):
        """A caveat about the measurement is not provenance. The share of the space a
        figure rests on changes how the figure should be read, so it stays on the page."""
        from nas4textgen.manuscript import Table

        rendered = Table(
            number="9", caption="x", columns=("a",), rows=[{"a": 1}],
            source="artifacts/somewhere.json",
            coverage="Built from 312 of 1,099 architectures.",
        ).render_latex()
        assert "312 of 1,099" in rendered
        assert "somewhere.json" not in rendered

    def test_a_table_without_a_coverage_note_ends_at_the_rule(self):
        """With the source gone there is nothing left to set below the table, and an empty
        \\footnotesize line would leave a blank strip under the rule."""
        from nas4textgen.manuscript import Table

        rendered = Table(number="0", caption="none", columns=("a",), rows=[{"a": 1}],
                         source="artifacts/nowhere.json").render_latex()
        assert "footnotesize" not in rendered
        assert "smallskip" not in rendered

    def test_coverage_travels_with_a_partial_table(self):
        """These artifacts are written while the sweep is still running, and a figure that
        would read as final if the caption were dropped is worse than one that cannot."""
        from nas4textgen.manuscript import Table

        table = Table(
            number="9", caption="x", columns=("a",), rows=[{"a": 1}],
            source="somewhere", coverage="Built from 312 of 1,099 architectures.",
        )
        assert "312 of 1,099" in table.render()

    def test_an_empty_table_renders_without_crashing(self):
        from nas4textgen.manuscript import Table

        rendered = Table(number="0", caption="none", columns=("a", "b"),
                         rows=[], source="nowhere").render()
        assert "| a | b |" in rendered

    def test_the_header_says_the_document_is_generated(self, artifacts):
        tables, missing = generate(artifacts)
        assert "Do not edit" in render_all(tables, missing)


def test_every_declared_table_is_reachable_from_the_real_artifacts():
    """The check that matters for the paper: with the pipeline's own output, every table in
    the manuscript order builds."""
    from pathlib import Path

    from nas4textgen.artifacts import artifacts_root

    root = artifacts_root()
    if not (root / "benchmark" / "summary.json").exists():
        pytest.skip("the benchmark table has not been assembled here")

    tables, missing = generate(Path(root))
    assert not missing, f"tables that could not be built: {missing}"
    assert len(tables) == len(TABLES)
    numbers = [t.number for t in tables]
    assert numbers == sorted(numbers, key=int), "tables must be in manuscript order"


ROOT = Path(__file__).resolve().parents[1]


class TestTheCoverageNotes:
    """The caveat each table carries about what it was computed from.

    Set below the table as a line of its own, the note ran the full column width while the
    table beside it was narrower and centred, so the two did not line up. It is a caveat
    about the measurement, so it belongs in the caption. Two of the notes were also wrong:
    they called the 1,103 measured rows architectures and divided them by the 1,099 of the
    space, and printed the result as 100.4 per cent of a space that cannot be exceeded.
    """

    def tables(self, artifacts):
        from nas4textgen.manuscript import generate

        built, _ = generate(artifacts)
        return built

    def test_no_note_claims_more_of_the_space_than_there_is(self):
        root = ROOT / "artifacts"
        if not (root / "benchmark" / "summary.json").exists():
            pytest.skip("no artifacts")
        for table in self.tables(root):
            for percentage in re.findall(r"(\d+(?:\.\d+)?)\s*(?:per cent|percent|\\?%)",
                                         f"{table.caption} {table.coverage}"):
                assert float(percentage) <= 100.0, (
                    f"Table {table.number} claims {percentage} per cent")

    def test_the_note_travels_in_the_caption_and_not_beside_the_table(self):
        root = ROOT / "artifacts"
        if not (root / "benchmark" / "summary.json").exists():
            pytest.skip("no artifacts")
        for table in self.tables(root):
            if not table.coverage:
                continue
            latex = table.render_latex()
            caption = latex.split(r"\caption{", 1)[1].split("\n")[0]
            assert table.coverage.rstrip(".") in caption.replace(r"\%", "%"), (
                f"Table {table.number} does not carry its note in the caption")
            assert r"\footnotesize" not in latex, (
                f"Table {table.number} still sets its note below the table")


class TestRunsAreNotCountedAsArchitectures:
    """1,103 runs, 1,099 architectures, and one label for both.

    The benchmark table printed the run count under "architectures measured", directly
    above a space of 1,099 and a coverage of 100 per cent. A reader asked where the extra
    four architectures were; there are none. One architecture was trained at five seeds to
    measure the region of practical equivalence, and its four further runs are the gap.
    """

    def summary(self):
        path = ROOT / "artifacts" / "benchmark" / "summary.json"
        if not path.exists():
            pytest.skip("no benchmark summary")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_counts_reconcile(self):
        summary = self.summary()
        assert summary["architectures"] + summary["replicate_rows"] == summary["rows"]

    def test_replicated_architectures_counts_architectures(self):
        """It was `rows - architectures`, which counts rows and is a different number."""
        summary = self.summary()
        assert summary["replicated_architectures"] <= summary["replicate_rows"]
        assert summary["replicated_architectures"] <= summary["architectures"]

    def test_coverage_is_measured_in_architectures(self):
        summary = self.summary()
        assert summary["coverage"] <= 1.0
        assert summary["coverage"] == pytest.approx(
            summary["architectures"] / summary["space_size"], abs=1e-4)

    def test_the_table_separates_the_two_counts(self):
        from nas4textgen.manuscript import table_benchmark_coverage

        root = ROOT / "artifacts"
        if not (root / "benchmark" / "summary.json").exists():
            pytest.skip("no artifacts")
        summary = self.summary()
        values = {row["quantity"]: row["value"]
                  for row in table_benchmark_coverage(root).rows}
        assert values["architectures measured"] == f"{summary['architectures']:,}"
        assert values["training runs"] == f"{summary['rows']:,}"
        assert values["architectures measured"] != values["training runs"], (
            "the two counts differ and the table must not print one for the other")
