"""What the project promises about reproduction, and the promises it refuses to make.

A replication claim that is not qualified will be wrong for someone. Enumerating the space
is arithmetic and gives the same bits anywhere; training a transformer on a GPU does not
give the same bits twice on the same machine. These tests hold the declaration to that
distinction.
"""

from __future__ import annotations

import re

import pytest

from nas4textgen.replication import (
    STAGES,
    TIER_REQUIREMENTS,
    Determinism,
    Stage,
    Tier,
    manifest,
    stages_for,
)


class TestTheDeclaration:
    def test_a_tolerance_exact_stage_must_name_its_tolerance(self):
        """A tolerance nobody states is a tolerance nobody can check."""
        with pytest.raises(ValueError, match="without naming the tolerance"):
            Stage(
                name="x", command="y", tier=Tier.SMOKE,
                determinism=Determinism.TOLERANCE_EXACT,
                reads=(), writes=(),
            )

    def test_a_bit_exact_stage_may_not_also_claim_a_tolerance(self):
        """One of the two would be wrong, and a reader cannot tell which."""
        with pytest.raises(ValueError, match="one of the two is wrong"):
            Stage(
                name="x", command="y", tier=Tier.VERIFY,
                determinism=Determinism.BIT_EXACT,
                reads=(), writes=(), tolerance="±1",
            )

    def test_every_declared_stage_states_what_it_reads_and_writes(self):
        for stage in STAGES:
            assert stage.reads, f"{stage.name} declares no inputs"
            assert stage.writes, f"{stage.name} declares no outputs"
            assert stage.command.startswith("python -m nas4textgen"), stage.name

    def test_the_training_stages_are_the_tolerance_exact_ones(self):
        """And nothing else is. If an analysis stage became tolerance-exact it would mean
        an unseeded random number somewhere."""
        tolerant = {s.name for s in STAGES if s.determinism is Determinism.TOLERANCE_EXACT}
        assert "one training run" in tolerant
        assert "one sweep wave" in tolerant
        assert "enumeration and exact cost front" not in tolerant
        assert "benchmark table and exact front" not in tolerant

    def test_the_corpus_fetch_is_honest_about_upstream_drift(self):
        """It is the one stage whose inputs can change without anyone here doing anything,
        and a later fetch is a different experiment rather than a failed reproduction."""
        fetch = next(s for s in STAGES if s.name == "corpus retrieval")
        assert fetch.determinism is Determinism.TOLERANCE_EXACT
        assert "vintage" in fetch.tolerance
        assert "different experiment" in fetch.tolerance

    def test_the_tolerance_names_where_it_came_from(self):
        """A number derived from a measurement, not chosen. The seed ROPE is 0.651
        perplexity from five replicates."""
        run = next(s for s in STAGES if s.name == "one training run")
        assert "0.651" in run.tolerance
        assert "replicates" in run.tolerance


class TestTiers:
    def test_the_tiers_nest(self):
        """Running a wave without the checks that make its output interpretable is not a
        replication, it is a demonstration."""
        verify = set(stages_for(Tier.VERIFY))
        smoke = set(stages_for(Tier.SMOKE))
        partial = set(stages_for(Tier.PARTIAL))
        assert verify < smoke < partial

    def test_the_cheapest_tier_needs_no_gpu_and_no_account(self):
        """The claim that matters to a reviewer with a laptop."""
        verify = TIER_REQUIREMENTS[Tier.VERIFY]
        assert "no GPU" in verify["hardware"]
        assert verify["cost"] == "nothing"

    def test_every_tier_states_hardware_wall_clock_and_cost(self):
        for tier, requirements in TIER_REQUIREMENTS.items():
            for field in ("hardware", "wall_clock", "cost", "what_it_proves"):
                assert requirements.get(field), f"{tier.value} does not state {field}"

    def test_each_tier_says_what_it_does_not_prove(self):
        """The verification tier recomputes the analysis and retrains nothing, and saying
        so is the difference between checking a paper and checking its measurements."""
        assert "does not retrain" in TIER_REQUIREMENTS[Tier.VERIFY]["what_it_proves"]
        assert "meaningless" in TIER_REQUIREMENTS[Tier.SMOKE]["what_it_proves"]

    def test_the_full_tier_quotes_the_measured_price_and_the_measured_variation(self):
        full = TIER_REQUIREMENTS[Tier.FULL]
        assert "174" in full["cost"]
        assert "0.07" in full["what_it_proves"] and "0.33" in full["what_it_proves"]

    def test_every_tier_carries_the_compact_form_the_paper_table_prints(self):
        for tier, requirements in TIER_REQUIREMENTS.items():
            compact = requirements.get("compact")
            assert compact, f"{tier.value} has no compact form"
            for field in ("wall_clock", "establishes"):
                assert compact.get(field), f"{tier.value} compact does not state {field}"
            for field in ("hardware", "cost"):
                assert field not in compact, (
                    f"{tier.value} carries a second {field}: the table prints the hardware "
                    "in full and does not print the cost")

    def test_no_figure_in_the_table_disagrees_with_the_figure_in_the_document(self):
        """The paper prints the compact form and the replication document the long one. A
        price edited in one and not the other is the drift this pair exists to prevent, and
        the pair replaced a paper that promised the verification tier in two minutes where
        this dictionary said ten."""
        for tier, requirements in TIER_REQUIREMENTS.items():
            for field in ("wall_clock",):
                figures = set(re.findall(r"\d+(?:\.\d+)?", requirements["compact"][field]))
                missing = figures - set(re.findall(r"\d+(?:\.\d+)?", requirements[field]))
                assert not missing, (
                    f"{tier.value} {field}: {sorted(missing)} is in the table and not in "
                    f"the document's {field!r}"
                )

    def test_the_compact_form_is_the_shorter_of_the_two(self):
        """It exists to save space in a journal column. One that grew past the long form
        would be the long form with an extra place to edit."""
        for tier, requirements in TIER_REQUIREMENTS.items():
            compact, full = requirements["compact"], requirements
            assert len(compact["establishes"]) < len(full["what_it_proves"]), tier.value
            assert len(compact["wall_clock"]) <= len(full["wall_clock"]), tier.value


class TestManifest:
    def test_every_stage_appears_in_exactly_one_determinism_class(self):
        payload = manifest()
        bit = set(payload["determinism_summary"]["bit_exact"])
        tol = set(payload["determinism_summary"]["tolerance_exact"])
        assert not (bit & tol)
        assert bit | tol == {s.name for s in STAGES}

    def test_the_manifest_explains_why_there_are_two_classes(self):
        assert "misleading" in manifest()["why_two_classes"]

    def test_exactly_the_stages_that_train_are_resumable(self):
        """The rule rather than a literal set, so adding a stage does not mean editing this
        test to agree with it.

        Every stage that trains resumes from a checkpoint, because a spot node reclaimed
        four hours in must not repeat those hours — the campaign recovered work 270 times
        this way. Every analysis stage is seconds to minutes and idempotent, so marking one
        resumable would be noise claiming a property that costs code to provide.
        """
        for stage in STAGES:
            trains = stage.tier is not Tier.VERIFY
            assert stage.resumable is trains, (
                f"{stage.name} is in the {stage.tier.value} tier and "
                f"{'is not' if trains else 'is'} marked resumable"
            )


class TestGeneratedDocument:
    def test_the_document_matches_the_declaration(self):
        """A document and a program that describe the same thing drift apart the moment one
        is edited. Regenerating and comparing is what keeps them from doing so silently."""
        from pathlib import Path

        from nas4textgen.replication import render_markdown

        target = Path(__file__).resolve().parents[1] / "docs" / "REPLICATION.md"
        if not target.exists():
            # Generated rather than written, and kept with the paper rather than in the
            # repository. A checkout that has never run `cli replication` reaches this
            # legitimately; what the test guards is drift between the two when both are
            # present, which is a question that needs the document to exist.
            pytest.skip("no generated replication document in this checkout")
        assert target.read_text(encoding="utf-8") == render_markdown() + "\n", (
            "docs/REPLICATION.md is out of date; regenerate it"
        )

    def test_every_stage_command_appears_in_the_document(self):
        from nas4textgen.replication import render_markdown

        rendered = render_markdown()
        for stage in STAGES:
            assert stage.command in rendered, f"{stage.name} has no command in the document"

    def test_the_document_states_the_two_determinism_classes_and_why(self):
        from nas4textgen.replication import render_markdown

        rendered = render_markdown()
        assert "bit_exact" in rendered and "tolerance_exact" in rendered
        assert "misleading about the other" in rendered
        assert "0.07 to" in rendered, "the measured re-run variation belongs in the document"


class TestAuthorship:
    """No automated tool is listed as an author anywhere. Asserted, because several
    journals restrict it and a single stray trailer would be the counterexample."""

    FORBIDDEN = (
        "Co-Authored-By: Claude",
        "Co-authored-by: Claude",
        "Generated with Claude",
        "Co-Authored-By: GPT",
        "Co-Authored-By: Copilot",
    )

    def test_no_commit_message_names_a_tool_as_an_author(self):
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        log = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%B"],
            capture_output=True, text=True, check=False,
        ).stdout

        for marker in self.FORBIDDEN:
            assert marker not in log, f"a commit message contains {marker!r}"

    def test_no_tracked_file_lists_a_tool_as_an_author(self):
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True, text=True, check=False,
        ).stdout.split()

        for name in tracked:
            path = root / name
            if not path.is_file() or path.suffix in {".png", ".pt", ".jsonl"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for marker in self.FORBIDDEN:
                # This test file names the markers in order to search for them.
                if path.name == "test_replication.py":
                    continue
                assert marker not in text, f"{name} contains {marker!r}"

    def test_the_document_states_the_authorship_rule(self):
        from nas4textgen.replication import render_markdown

        assert "No automated tool is listed as an author" in render_markdown()
