"""The verification tier, and the ways a check like this fails silently.

A tier that regenerates into the directory it is checking passes unconditionally, forever,
and looks exactly like a tier that works. These tests hold the runner to comparing two
different trees, to excluding only what it says it excludes, and to refusing where it
cannot re-derive rather than counting an unreproduced artifact as reproduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nas4textgen.replication import STAGES, Determinism, Stage, Tier
from nas4textgen.verify import (
    EXCLUDED_KEYS,
    ArtifactOutcome,
    RootsCoincide,
    _resolve_rows,
    compare_files,
    compare_values,
    run,
    seed_inputs,
)


class TestTheOnlyWayThisCheckFails:
    def test_regenerating_into_the_tree_being_checked_is_refused(self, tmp_path):
        """Each file would be compared against itself. The check would pass forever and
        look exactly like a check that works."""
        with pytest.raises(RootsCoincide, match="pass\\s+unconditionally"):
            run(tmp_path, tmp_path)

    def test_regenerating_inside_the_tree_being_checked_is_refused(self, tmp_path):
        inner = tmp_path / "scratch"
        with pytest.raises(RootsCoincide, match="sits inside"):
            run(tmp_path, inner)


class TestComparison:
    def test_a_bit_exact_stage_gets_no_tolerance_the_manuscript_did_not_claim(self):
        differences = compare_values(1.0, 1.0 + 1e-12, tolerance=None)
        assert differences, "exact means exact"

    def test_a_tolerance_exact_stage_absorbs_a_difference_inside_its_tolerance(self):
        assert not compare_values(1.0, 1.0 + 1e-12, tolerance=1e-6)
        assert compare_values(1.0, 1.5, tolerance=1e-6)

    def test_an_integer_that_moved_is_never_a_rounding_difference(self):
        """A changed row count passing as noise would let coverage drift silently."""
        differences = compare_values({"rows": 321}, {"rows": 792}, tolerance=1e-3)
        assert differences and "rows" in differences[0].path

    def test_a_missing_key_is_reported_on_the_side_it_is_missing_from(self):
        left = compare_values({"a": 1}, {}, tolerance=None)
        right = compare_values({}, {"a": 1}, tolerance=None)
        assert left[0].regenerated == "<absent>"
        assert right[0].published == "<absent>"

    def test_two_nans_agree_because_they_came_from_the_same_undefined_quantity(self):
        assert not compare_values(float("nan"), float("nan"), tolerance=None)

    def test_a_length_difference_is_named_before_the_elementwise_ones(self):
        differences = compare_values([1, 2, 3], [1, 2], tolerance=None)
        assert differences[0].path.endswith("[len]")

    def test_booleans_are_compared_as_booleans_not_as_numbers(self):
        """Python would call True equal to 1.0 under a numeric comparison."""
        assert compare_values(True, 1.0, tolerance=1e-3)


class TestProvenanceIsExcludedOnPurpose:
    def test_a_timestamp_difference_does_not_fail_a_stage(self, tmp_path):
        published, regenerated = tmp_path / "a.json", tmp_path / "b.json"
        published.write_text(json.dumps(
            {"provenance": {"written_at": "2026-01-01", "command": "x"}, "value": 3}
        ), encoding="utf-8")
        regenerated.write_text(json.dumps(
            {"provenance": {"written_at": "2026-09-02", "command": "y"}, "value": 3}
        ), encoding="utf-8")
        status, differences, _ = compare_files(published, regenerated, None)
        assert status == "matched", differences

    def test_a_real_difference_beside_a_provenance_one_still_fails(self, tmp_path):
        """Dropping the header without saying so would let a difference hide in a block
        nobody looks at."""
        published, regenerated = tmp_path / "a.json", tmp_path / "b.json"
        published.write_text(json.dumps({"provenance": {"t": 1}, "value": 3}),
                             encoding="utf-8")
        regenerated.write_text(json.dumps({"provenance": {"t": 2}, "value": 4}),
                               encoding="utf-8")
        status, differences, _ = compare_files(published, regenerated, None)
        assert status == "differs"
        assert differences[0].path == "value"

    def test_what_is_excluded_is_named_rather_than_implicit(self):
        assert EXCLUDED_KEYS == ("provenance",)


class TestWhatCannotBeReDerivedHere:
    def test_the_benchmark_table_is_rebuilt_and_its_manifests_are_the_input(self):
        """It was an input, on the grounds that reading the manifests needs an account.
        The command had since been changed to read the published mirror, so nothing
        rebuilt the table, and an assembly bug that moved the measured front from 220
        architectures to 218 went unreported for as long as that stood."""
        table = next(s for s in STAGES if s.name == "benchmark table and exact front")
        assert "cloud" not in table.requires
        assert table.reads == ("artifacts/manifests/runs.jsonl",)

    def test_corpus_retrieval_is_an_input_because_it_reaches_the_network(self):
        fetch = next(s for s in STAGES if s.name == "corpus retrieval")
        assert "network" in fetch.requires

    def test_a_stage_needing_something_the_tier_forbids_is_not_counted_as_passed(self):
        outcome = ArtifactOutcome("s", "a.json", "input_only")
        assert outcome.ok, "an input is not a failure"
        assert ArtifactOutcome("s", "a.json", "missing_regenerated").ok is False

    def test_every_verify_stage_is_reachable_by_a_command_that_exists(self):
        """docs/REPLICATION.md printed `cli surrogate` and `cli characterise` and neither
        existed, so a reader following the instructions hit 'invalid choice'."""
        from nas4textgen.cli import commands

        available = commands()
        for stage in STAGES:
            if stage.tier is not Tier.VERIFY:
                continue
            command = stage.command.split()
            assert command[-1] in available, f"{stage.name}: `{stage.command}` does not exist"


class TestSeeding:
    def test_what_a_stage_declares_it_reads_is_put_where_it_looks(self, tmp_path):
        published, work = tmp_path / "published", tmp_path / "work"
        (published / "manifests").mkdir(parents=True)
        (published / "manifests" / "runs.jsonl").write_text("{}\n", encoding="utf-8")
        (published / "pilot").mkdir()
        (published / "pilot" / "results.json").write_text("{}", encoding="utf-8")
        work.mkdir()

        seeded = seed_inputs(work, published, Tier.VERIFY)
        assert "manifests/runs.jsonl" in seeded, (
            "the table stage reads the mirrored manifests, which this tier does not produce"
        )
        assert "pilot/results.json" in seeded, (
            "the characterisation reads the pilot's measured ROPE, and nothing in its "
            "writes or requires would have revealed that"
        )

    def test_the_measured_rope_is_a_declared_input_not_a_flag_default(self):
        stage = next(s for s in STAGES if s.name == "characterisation")
        assert "artifacts/pilot/results.json" in stage.reads


class TestSupersededSnapshots:
    def test_the_newest_coverage_is_compared_and_the_rest_are_superseded(self, tmp_path):
        """slate_at_176.json came from a table that has since grown; asking this tier to
        regenerate it would ask it to reproduce a table no longer published."""
        directory = tmp_path / "surrogate"
        directory.mkdir()
        for rows in (176, 321):
            (directory / f"slate_at_{rows}.json").write_text("{}", encoding="utf-8")

        resolved = _resolve_rows("surrogate/slate_at_<rows>.json", tmp_path)
        assert resolved == [
            ("surrogate/slate_at_176.json", True),
            ("surrogate/slate_at_321.json", False),
        ]

    def test_an_unresolvable_pattern_comes_back_flagged_rather_than_guessed_at(self, tmp_path):
        resolved = _resolve_rows("surrogate/slate_at_<rows>.json", tmp_path)
        assert resolved == [("surrogate/slate_at_<rows>.json", True)]


class TestTheDeclarationStaysHonest:
    def test_a_tolerance_exact_stage_names_its_tolerance(self):
        with pytest.raises(ValueError, match="tolerance nobody states"):
            Stage(name="x", command="c", tier=Tier.VERIFY,
                  determinism=Determinism.TOLERANCE_EXACT, reads=(), writes=())

    def test_requires_travels_into_the_manifest(self):
        stage = Stage(name="x", command="c", tier=Tier.VERIFY,
                      determinism=Determinism.BIT_EXACT, reads=(), writes=(),
                      requires=("cloud",))
        assert stage.as_dict()["requires"] == ["cloud"]


class TestSeedingCannotMaskAStageThatWroteNothing:
    def test_an_artifact_this_tier_regenerates_is_never_seeded(self, tmp_path):
        """The same failure as regenerating into the published tree, through the back
        door: a stage whose output was seeded on top would be compared against the copy
        that was put there, so writing nothing would pass."""
        published, work = tmp_path / "published", tmp_path / "work"
        (published / "space").mkdir(parents=True)
        (published / "space" / "enumeration.csv").write_text("h,d\n1,2\n", encoding="utf-8")
        (published / "benchmark").mkdir()
        (published / "benchmark" / "table.csv").write_text("x", encoding="utf-8")
        work.mkdir()

        seeded = seed_inputs(work, published, Tier.VERIFY)
        assert "space/enumeration.csv" not in seeded, (
            "the enumerate stage runs in this tier and writes it"
        )
        assert "benchmark/table.csv" not in seeded, (
            "the table stage runs here too, and seeding its output would compare the "
            "published table against itself"
        )
        assert not (work / "space" / "enumeration.csv").exists()


class TestRequirementsAreReadRelativeToTheTier:
    """The same requirement means different things at different tiers."""

    def test_the_table_is_an_input_under_verify_and_a_stage_under_partial(self):
        from nas4textgen.replication import TIER_PROVIDES

        assert "cloud" not in TIER_PROVIDES[Tier.VERIFY]
        assert "cloud" in TIER_PROVIDES[Tier.PARTIAL]

    def test_a_stage_that_submits_jobs_declares_it_needs_an_account(self):
        """Saying it required nothing was the same misclassification the table stage had:
        `sweep --wave 1 --submit` queues billable work in a subscription."""
        wave = next(s for s in STAGES if s.name == "one sweep wave")
        assert "cloud" in wave.requires

    def test_every_tier_that_quotes_a_cost_has_a_stage_that_incurs_it(self):
        """The full tier quoted twelve days and 174 USD for work no stage named, and a tier
        with no stages reports success for having done nothing."""
        from nas4textgen.replication import TIER_REQUIREMENTS

        for tier in TIER_REQUIREMENTS:
            assert any(s.tier is tier for s in STAGES), f"{tier.value} has no stage"

    def test_every_stage_requirement_is_provided_by_some_tier(self):
        """A requirement no tier supplies would make the stage permanently unreachable."""
        from nas4textgen.replication import TIER_PROVIDES

        everything = set().union(*TIER_PROVIDES.values())
        for stage in STAGES:
            assert not set(stage.requires) - everything, stage.name


class TestTheSmokeTierChecksStructureNotValues:
    def test_the_required_fields_are_the_ones_the_assembly_actually_reads(self):
        """A field missing here is a row silently absent from the benchmark twelve days
        later, so the list comes from what `assemble` consumes."""
        from nas4textgen.verify import MANIFEST_REQUIREMENTS

        for field_name in ("usable_for_benchmark", "genotype", "precision",
                           "data_vintage", "tokeniser_fingerprint"):
            assert field_name in MANIFEST_REQUIREMENTS

    def test_a_manifest_the_assembly_would_drop_fails_the_tier(self, tmp_path):
        from nas4textgen.verify import check_manifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({
            "genotype": {"values": [2, 32, 2, 2, 1]}, "objectives": {"perplexity": 1.0},
            "precision": "float32", "data_vintage": "x", "tokeniser_fingerprint": "y",
            "device": "cpu", "config": {"seed": 0},
        }), encoding="utf-8")
        missing = check_manifest(path)
        assert any("usable_for_benchmark" in entry for entry in missing), (
            "assemble drops every manifest without it"
        )

    def test_a_malformed_genotype_is_caught_by_arity_not_by_presence(self, tmp_path):
        from nas4textgen.verify import check_manifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({
            "usable_for_benchmark": True, "genotype": {"values": [2, 32]},
            "objectives": {}, "precision": "f", "data_vintage": "x",
            "tokeniser_fingerprint": "y", "device": "cpu", "config": {},
        }), encoding="utf-8")
        assert any("not five" in entry for entry in check_manifest(path))

    def test_a_manifest_that_was_never_written_is_named_as_such(self, tmp_path):
        from nas4textgen.verify import check_manifest

        missing = check_manifest(tmp_path / "absent.json")
        assert "was not written at all" in missing[0]

    def test_invalid_json_is_reported_rather_than_raising(self, tmp_path):
        from nas4textgen.verify import check_manifest

        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        assert "not valid JSON" in check_manifest(path)[0]

    def test_the_smoke_report_does_not_claim_a_comparison_it_did_not_make(self):
        from nas4textgen.verify import VerificationReport

        report = VerificationReport(tier=Tier.SMOKE, published_root=Path("a"),
                                    work_root=Path("b"))
        rendered = report.render()
        assert "recomputed and identical" not in rendered
        assert "meaningless" in rendered
