"""Whether continuous integration runs what the replication document prints.

The workflow's own header says it "runs the same commands the documentation prints, so the
two cannot drift apart". That was a claim about intent with nothing enforcing it: the
document is generated from a declaration in code, the workflow is hand-written YAML, and
nothing compared them. Two commands the document printed did not exist at all for weeks —
`cli surrogate` and `cli characterise` — and CI passed the whole time, because it never ran
them.

So the comparison is a test. A verify-tier stage either appears in the workflow or is listed
below as one CI cannot run, with the reason. Adding a stage and forgetting CI then fails
here instead of passing quietly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nas4textgen.replication import STAGES, TIER_PROVIDES, Tier

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "verify.yml"

#: Verify-tier stages continuous integration does not run, and why. Each entry is a
#: standing admission rather than an oversight, so the list should be short and every line
#: should say what would have to change for the stage to move out of it.
NOT_IN_CI: dict[str, str] = {
    "corpus retrieval": (
        "reaches the dataset hosts. CI fetches a small slice for the smoke tier, but "
        "retrieving all four corpora in full would make every pull request wait on "
        "several gigabytes of download."
    ),
    "benchmark table and exact front": (
        "assembles from the run manifests in the artifact container, which needs the "
        "subscription. It moves into CI when the published table has a citable identifier "
        "the workflow can download — task 16.9."
    ),
    "surrogate study": (
        "reads the published benchmark table, which is not in the repository. Same "
        "dependency as the table stage: it moves into CI with 16.9."
    ),
    "characterisation": (
        "reads the published benchmark table and the pilot's measured ROPE, neither of "
        "which is in the repository. Moves into CI with 16.9."
    ),
}


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def commands_in_workflow() -> set[str]:
    """Subcommands the workflow invokes through the CLI."""
    return set(re.findall(r"python -m nas4textgen\.cli\s+([a-z-]+)", workflow_text()))


class TestTheWorkflowExists:
    def test_there_is_a_workflow(self):
        assert WORKFLOW.exists(), "16.7 asks for CI; the file is the deliverable"

    def test_it_declares_no_gpu(self):
        """The verify tier's whole claim is that it needs none."""
        assert "runs-on: ubuntu-latest" in workflow_text()

    def test_threading_is_pinned_because_bit_exactness_depends_on_it(self):
        """Multithreaded BLAS combines partial sums in completion order, so a reduction is
        correct without being bit-identical between runs."""
        for variable in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            assert variable in workflow_text(), variable


class TestNoDriftFromTheDocument:
    def test_every_verify_stage_is_run_or_explained(self):
        """A stage added without a CI step fails here rather than passing quietly."""
        run = commands_in_workflow()
        missing = []
        for stage in STAGES:
            if stage.tier is not Tier.VERIFY:
                continue
            subcommand = stage.command.split()[-1]
            if subcommand in run or stage.name in NOT_IN_CI:
                continue
            missing.append(f"{stage.name} (`{stage.command}`)")
        assert not missing, (
            "these verify-tier stages are neither run by CI nor listed as impossible:\n  "
            + "\n  ".join(missing)
        )

    def test_the_exclusions_are_real_stages(self):
        """An exclusion naming a stage that no longer exists hides a stage that does."""
        names = {stage.name for stage in STAGES}
        for excluded in NOT_IN_CI:
            assert excluded in names, f"{excluded!r} is not a stage"

    def test_every_excluded_stage_needs_something_ci_lacks(self):
        """A stage CI could run has no business on the exclusion list."""
        for name, reason in NOT_IN_CI.items():
            stage = next(s for s in STAGES if s.name == name)
            needs = set(stage.requires) - TIER_PROVIDES[stage.tier]
            reads_published = any(
                entry.startswith("artifacts/") for entry in stage.reads
            )
            assert needs or reads_published, (
                f"{name} requires nothing CI lacks, so it should be a step: {reason}"
            )

    def test_every_command_ci_runs_exists(self):
        """CI invoking a subcommand that does not exist would fail the run, but only if the
        step is reached — a step after a long one can sit broken for weeks."""
        from nas4textgen.cli import commands

        available = commands()
        for subcommand in commands_in_workflow():
            assert subcommand in available, f"CI runs `cli {subcommand}`, which does not exist"


class TestTheSmokeTierRuns:
    def test_ci_executes_the_smoke_tier(self):
        """16.7 asks for the verification *and* smoke tiers. The smoke tier is the only one
        that runs the training path, so without it CI proves the analysis reproduces and
        says nothing about whether a model can be trained at all."""
        assert "--tier smoke" in workflow_text(), (
            "no smoke-tier step; the training path is never exercised"
        )

    def test_the_smoke_tier_gets_a_corpus_to_train_on(self):
        """It reads the cache, and the cache is not committed."""
        text = workflow_text()
        assert "cli fetch" in text and "cli tokenise" in text, (
            "the smoke tier needs a corpus fetched and tokenised first"
        )
        assert "--limit" in text, (
            "fetching a corpus in full would make every pull request wait on gigabytes"
        )


class TestTheChecksThatCatchTheProjectsOwnMistakes:
    @pytest.mark.parametrize("step", [
        "audit_secrets",
        "audit_image",
        "check_env",
        "replication --check",
    ])
    def test_the_standing_audits_are_wired(self, step):
        assert step in workflow_text(), f"{step} is not run by CI"

    def test_the_objection_registry_is_checked(self):
        """Its citations point at files, and a rename breaks them silently."""
        assert "cli objections" in workflow_text()

    def test_authorship_is_checked_in_ci_not_only_by_convention(self):
        assert "co-authored-by" in workflow_text().lower()
