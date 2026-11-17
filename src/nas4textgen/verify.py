"""Re-derive the published analysis and check that it comes out the same.

The verification tier answers one question: does every number in the paper follow from the
published benchmark table? It retrains nothing, needs no GPU and no cloud account, and runs
in minutes — so it is the tier a reviewer can actually execute, which is the point of
having it.

**The regenerated artifacts go somewhere else.** A verification that rewrites the directory
it is checking compares each file against itself and passes unconditionally. That is not a
subtle bug; it is the only way this kind of check fails, and it fails silently and forever.
So the runner refuses when the two roots coincide, and every stage is executed with the
artifact root pointed at a scratch directory.

**The provenance header is excluded from the comparison, deliberately.** It records the
timestamp, the command line and the working revision, all of which differ between the
published run and the reader's by construction. Comparing it would make every stage fail;
dropping it without saying so would let a real difference hide inside a block nobody looks
at. So it is excluded by name, and what is excluded is reported.

**A stage promising bit-exactness is compared exactly.** No tolerance is applied where none
was declared, because a tolerance introduced by the checker is a tolerance the manuscript
never claimed. Tolerance-exact stages are compared against the tolerance their own
declaration names.

**Not every stage can be re-derived here, and each says why in its own declaration.** A
stage naming a requirement this tier does not assume — a network-reachable dataset host, a
cloud account holding the run manifests — has its outputs treated as inputs, and the report
names which. The benchmark table is the important case: assembling it reads the run
manifests from the artifact container, so a reader without an account cannot rebuild it,
and re-deriving it here would compare the published table against a fresher recomputation
and report the campaign's own progress as a reproduction failure. The table is the input;
what this tier checks is that every number in the paper follows from it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .replication import STAGES, TIER_PROVIDES, Determinism, Stage, Tier

#: Relative tolerance for stages that declare tolerance-exact reproduction. Applied to
#: floats only; an integer that moved is a different count, not a rounding difference.
DEFAULT_RELATIVE_TOLERANCE = 1e-6

#: Keys dropped before comparing. Everything here differs between two runs by
#: construction, and the list is reported so the exclusion is visible rather than assumed.
EXCLUDED_KEYS = ("provenance",)

#: Stages whose outputs live outside the artifact tree. They are inputs to this tier.
_CACHE_PREFIX = "data/cache/"


class RootsCoincide(RuntimeError):
    """Raised when the regeneration target is the directory being verified."""


@dataclass(frozen=True)
class Difference:
    """One place where the regenerated artifact and the published one disagree."""

    path: str
    published: object
    regenerated: object

    def render(self) -> str:
        return f"{self.path}: published {self.published!r}, regenerated {self.regenerated!r}"


@dataclass(frozen=True)
class ArtifactOutcome:
    """What happened to one artifact of one stage."""

    stage: str
    artifact: str
    #: matched | differs | missing_published | missing_regenerated | input_only |
    #: superseded
    status: str
    differences: tuple[Difference, ...] = ()
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("matched", "input_only", "superseded")

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "artifact": self.artifact,
            "status": self.status,
            "differences": [d.render() for d in self.differences[:20]],
            "difference_count": len(self.differences),
            "detail": self.detail,
        }


@dataclass
class StageOutcome:
    """One stage: whether its command ran, and whether its outputs match."""

    stage: Stage
    ran: bool
    exit_code: int | None
    artifacts: list[ArtifactOutcome] = field(default_factory=list)
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code in (0, None) and all(a.ok for a in self.artifacts)

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.name,
            "command": self.stage.command,
            "determinism": self.stage.determinism.value,
            "tolerance": self.stage.tolerance or None,
            "ran": self.ran,
            "exit_code": self.exit_code,
            "ok": self.ok,
            "artifacts": [a.as_dict() for a in self.artifacts],
            "stderr_tail": self.stderr_tail,
        }


@dataclass
class VerificationReport:
    """The whole tier."""

    tier: Tier
    published_root: Path
    work_root: Path
    stages: list[StageOutcome] = field(default_factory=list)
    #: Published artifacts copied in as this tier's inputs, before anything ran.
    seeded: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.stages)

    @property
    def compared(self) -> int:
        return sum(a.status == "matched" for s in self.stages for a in s.artifacts)

    @property
    def inputs_only(self) -> int:
        return sum(a.status == "input_only" for s in self.stages for a in s.artifacts)

    def failures(self) -> list[ArtifactOutcome]:
        return [a for s in self.stages for a in s.artifacts if not a.ok]

    def as_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier.value,
            "passed": self.passed,
            "published_root": str(self.published_root),
            "regenerated_into": str(self.work_root),
            "artifacts_compared": self.compared,
            "artifacts_treated_as_inputs": self.inputs_only,
            "seeded_from_published": list(self.seeded),
            "excluded_from_comparison": list(EXCLUDED_KEYS),
            "why_excluded": (
                "The provenance header records the timestamp, command line and working "
                "revision, which differ between two runs by construction. Excluded by name "
                "and reported, so the exclusion is visible rather than assumed."
            ),
            "why_a_separate_root": (
                "A verification that regenerates into the directory it is checking compares "
                "each file against itself and passes unconditionally."
            ),
            "stages": [s.as_dict() for s in self.stages],
            "failures": [a.as_dict() for a in self.failures()],
        }

    def render(self) -> str:
        lines = [
            f"  tier {self.tier.value}   published {self.published_root}",
            f"  regenerated into {self.work_root}",
            "",
        ]
        for outcome in self.stages:
            mark = "ok" if outcome.ok else "FAIL"
            lines.append(f"  {mark:<5}{outcome.stage.name:<38}"
                         f"{outcome.stage.determinism.value}")
            for artifact in outcome.artifacts:
                if artifact.status == "matched":
                    continue
                lines.append(f"        {artifact.status:<22}{artifact.artifact}")
                for difference in artifact.differences[:5]:
                    lines.append(f"          {difference.render()}")
                if artifact.detail:
                    lines.append(f"          {artifact.detail}")
            if not outcome.ok and outcome.stderr_tail:
                lines.append(f"        stderr: {outcome.stderr_tail[:200]}")
        # The smoke tier compares nothing against a published artifact, so saying its
        # outputs were "recomputed and identical" would claim a reproduction it did not
        # perform. It checks that the training path ran and wrote a usable manifest.
        if self.tier is Tier.SMOKE:
            return "\n".join(lines + [
                "",
                "  the training path ran end to end and wrote a manifest the benchmark",
                "  assembly would accept. Nothing here is compared against a published",
                "  value: at this budget the perplexity is meaningless, which is what the",
                "  tier's own declaration says.",
                *([f"  {len(self.failures())} check(s) failed"] if not self.passed else []),
            ])

        lines += [
            "",
            f"  {self.compared} artifact(s) recomputed and identical, "
            f"{self.inputs_only} treated as inputs",
            f"  {len(self.seeded)} published artifact(s) seeded as input: "
            f"{', '.join(self.seeded) or 'none'}",
        ]
        if not self.passed:
            lines.append(f"  {len(self.failures())} artifact(s) did not reproduce")
        return "\n".join(lines)


def _strip_excluded(payload: object) -> object:
    """Drop the keys that differ between two runs by construction."""
    if isinstance(payload, dict):
        return {
            key: _strip_excluded(value)
            for key, value in payload.items()
            if key not in EXCLUDED_KEYS
        }
    if isinstance(payload, list):
        return [_strip_excluded(item) for item in payload]
    return payload


def compare_values(
    published: object,
    regenerated: object,
    tolerance: float | None,
    path: str = "",
) -> list[Difference]:
    """Structural comparison, with tolerance applied to floats only.

    ``tolerance`` of ``None`` means exact. Integers are compared exactly regardless: an
    integer that moved is a different count, and calling that a rounding difference would
    let a changed row count pass as noise.
    """
    differences: list[Difference] = []

    if isinstance(published, dict) and isinstance(regenerated, dict):
        for key in sorted(set(published) | set(regenerated)):
            where = f"{path}.{key}" if path else key
            if key not in published:
                differences.append(Difference(where, "<absent>", regenerated[key]))
            elif key not in regenerated:
                differences.append(Difference(where, published[key], "<absent>"))
            else:
                differences += compare_values(
                    published[key], regenerated[key], tolerance, where
                )
        return differences

    if isinstance(published, list) and isinstance(regenerated, list):
        if len(published) != len(regenerated):
            differences.append(
                Difference(f"{path}[len]", len(published), len(regenerated))
            )
        for index, (left, right) in enumerate(zip(published, regenerated, strict=False)):
            differences += compare_values(left, right, tolerance, f"{path}[{index}]")
        return differences

    if isinstance(published, bool) or isinstance(regenerated, bool):
        # Type as well as value: Python calls True equal to 1.0, so a flag that became a
        # number would compare equal and the change would go unreported.
        if type(published) is not type(regenerated) or published != regenerated:
            differences.append(Difference(path, published, regenerated))
        return differences

    if isinstance(published, float) or isinstance(regenerated, float):
        try:
            left, right = float(published), float(regenerated)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            differences.append(Difference(path, published, regenerated))
            return differences
        if math.isnan(left) and math.isnan(right):
            return differences
        if tolerance is None:
            if left != right:
                differences.append(Difference(path, published, regenerated))
        elif not math.isclose(left, right, rel_tol=tolerance, abs_tol=0.0):
            differences.append(Difference(path, published, regenerated))
        return differences

    if published != regenerated:
        differences.append(Difference(path, published, regenerated))
    return differences


def compare_files(
    published: Path,
    regenerated: Path,
    tolerance: float | None,
) -> tuple[str, list[Difference], str]:
    """Compare one artifact. Returns a status, the differences, and a note."""
    if not published.exists():
        return "missing_published", [], f"{published} is not in the published tree"
    if not regenerated.exists():
        return "missing_regenerated", [], (
            f"the stage did not write {regenerated.name}; its command reported success, so "
            "either the output moved or it was never produced"
        )

    if published.suffix == ".json":
        left = _strip_excluded(json.loads(published.read_text(encoding="utf-8")))
        right = _strip_excluded(json.loads(regenerated.read_text(encoding="utf-8")))
        differences = compare_values(left, right, tolerance)
        return ("matched" if not differences else "differs"), differences, ""

    # CSV and everything else: compared line by line. Byte comparison would be simpler and
    # would report "the file differs" for a one-cell change, which is not enough to act on.
    left_lines = published.read_text(encoding="utf-8").splitlines()
    right_lines = regenerated.read_text(encoding="utf-8").splitlines()
    differences = []
    if len(left_lines) != len(right_lines):
        differences.append(Difference("[lines]", len(left_lines), len(right_lines)))
    for index, (left_line, right_line) in enumerate(
        zip(left_lines, right_lines, strict=False), start=1
    ):
        if left_line != right_line:
            differences.append(Difference(f"line {index}", left_line, right_line))
    return ("matched" if not differences else "differs"), differences, ""


def _resolve_rows(pattern: str, root: Path) -> list[tuple[str, bool]]:
    """Resolve a ``<rows>`` placeholder against what the published tree holds.

    The surrogate slate is written once per coverage — ``slate_at_176.json`` came from a
    176-row table and ``slate_at_321.json`` from a 321-row one — so the directory
    accumulates snapshots of the same stage at different points in the campaign. Only the
    highest count is the current output; the earlier ones are prior runs whose input no
    longer exists as a published artifact, and demanding their regeneration would ask this
    tier to reproduce a table that has since grown.

    So the newest is compared and the rest are reported as superseded. The same rule the
    manuscript tables already use to pick which slate to quote.
    """
    prefix, _, suffix = pattern.partition("<rows>")
    directory = (root / prefix).parent
    if not directory.is_dir():
        return [(pattern, True)]

    stem = Path(prefix).name
    found: list[tuple[int, str]] = []
    for candidate in directory.iterdir():
        name = candidate.name
        if not (name.startswith(stem) and name.endswith(suffix)):
            continue
        middle = name[len(stem):len(name) - len(suffix)] if suffix else name[len(stem):]
        if middle.isdigit():
            found.append((int(middle), str(candidate.relative_to(root))))
    if not found:
        return [(pattern, True)]

    found.sort()
    return [(relative, index != len(found) - 1) for index, (_, relative) in enumerate(found)]


def _expand(stage: Stage, root: Path) -> list[tuple[str, bool]]:
    """Which files a stage's declared outputs resolve to, and whether each is superseded.

    A declaration naming a path outside the artifact tree, or one with a placeholder this
    function cannot resolve, comes back flagged rather than guessed at — a path invented
    here would be compared against nothing and pass.
    """
    resolved: list[tuple[str, bool]] = []
    for declared in stage.writes:
        if declared.startswith(_CACHE_PREFIX):
            resolved.append((declared, True))
            continue
        relative = declared.removeprefix("artifacts/")
        if "<rows>" in relative:
            resolved += _resolve_rows(relative, root)
            continue
        if "<" in relative:
            resolved.append((declared, True))
            continue
        if declared.endswith("/"):
            base = root / relative.rstrip("/")
            if base.is_dir():
                for found in sorted(base.rglob("*")):
                    if found.is_file():
                        resolved.append((str(found.relative_to(root)), False))
            continue
        resolved.append((relative, False))
    return resolved


def seed_inputs(work_root: Path, published_root: Path, tier: Tier) -> list[str]:
    """Copy this tier's inputs into the scratch tree before anything reads them.

    The analyses read the published benchmark table, so it has to be where they look. That
    is also what a reader does: download the published table into the artifact directory
    and run the printed commands unchanged — which is the point of seeding rather than
    passing each stage a different path, since a command that only works with an extra flag
    is not the command the replication instructions print.
    """
    import shutil

    # Never seed something this tier regenerates. A stage whose output was seeded on top
    # would be compared against the copy that was put there, so a stage that silently
    # wrote nothing would pass — the same failure as regenerating into the published tree,
    # arriving through the back door. Stages that cannot run here are exempt: their
    # outputs are inputs, which is the whole reason they are seeded.
    provided = TIER_PROVIDES[tier]
    regenerated_here: set[str] = set()
    for stage in STAGES:
        if stage.tier is not tier or set(stage.requires) - provided:
            continue
        regenerated_here.update(
            relative for relative, unresolvable in _expand(stage, published_root)
            if not unresolvable
        )

    wanted: list[str] = []
    for stage in STAGES:
        if stage.tier is not tier:
            continue
        # Everything a stage of this tier declares it reads, when that is an artifact.
        # Derived from the declaration rather than from a guess about which stages need
        # help: the characterisation reads the pilot's measured ROPE and nothing in its
        # writes or requires would have revealed that, so it refused — correctly, and for
        # a dependency nobody had written down.
        wanted += [
            entry.removeprefix("artifacts/")
            for entry in stage.reads
            if entry.startswith("artifacts/") and "<" not in entry
        ]
        if set(stage.requires) - provided:
            wanted += [
                relative for relative, unresolvable in _expand(stage, published_root)
                if not unresolvable
            ]

    seeded: list[str] = []
    for relative in dict.fromkeys(wanted):
        if relative in regenerated_here:
            continue
        source = published_root / relative
        if not source.is_file():
            continue
        target = work_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        seeded.append(relative)
    return seeded


def run_stage(stage: Stage, work_root: Path, published_root: Path) -> StageOutcome:
    """Execute one stage into the scratch root and compare what it wrote."""
    unmet = set(stage.requires) - TIER_PROVIDES[stage.tier]
    if unmet:
        # The stage needs something its tier does not assume the reader has: a
        # network-reachable dataset host, a cloud account holding the run manifests, an
        # accelerator. Running it anyway would either fail on a reader's machine or —
        # worse, on ours — succeed and compare a published artifact against a fresher
        # recomputation, reporting the campaign's own progress as a reproduction failure.
        # So its outputs are inputs to this tier, and the report says which.
        #
        # Weighed against the tier rather than treated as an absolute veto, because the
        # same requirement means different things at different tiers: the benchmark table
        # needs an account, which makes it an input under `verify` and a stage that runs
        # under `partial`.
        return StageOutcome(
            stage=stage,
            ran=False,
            exit_code=None,
            artifacts=[ArtifactOutcome(
                stage.name, ", ".join(stage.writes), "input_only",
                detail=f"requires {', '.join(sorted(unmet))}, which the "
                       f"{stage.tier.value} tier does not assume; its outputs are inputs "
                       "here",
            )],
        )

    environment = {**os.environ, "NAS4TG_ARTIFACTS": str(work_root)}
    command = stage.command.replace("python -m", f"{sys.executable} -m", 1)
    completed = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        env=environment, cwd=Path.cwd(),
    )

    tolerance = (
        None if stage.determinism is Determinism.BIT_EXACT else DEFAULT_RELATIVE_TOLERANCE
    )
    outcome = StageOutcome(
        stage=stage,
        ran=True,
        exit_code=completed.returncode,
        stderr_tail=completed.stderr.strip().splitlines()[-1]
        if completed.stderr.strip() else "",
    )

    # Expand against the published tree: what the paper contains is the list to reproduce.
    # Expanding against the regenerated one would let a stage that wrote nothing pass with
    # an empty comparison.
    for relative, flagged in _expand(stage, published_root):
        if flagged and ("<" in relative or relative.startswith(_CACHE_PREFIX)):
            outcome.artifacts.append(ArtifactOutcome(
                stage.name, relative, "input_only",
                detail="path is declared with a placeholder this tier cannot resolve",
            ))
            continue
        if flagged:
            outcome.artifacts.append(ArtifactOutcome(
                stage.name, relative, "superseded",
                detail="a snapshot of this stage at an earlier coverage; its input table "
                       "has since grown, so regenerating it would need a table that is no "
                       "longer published",
            ))
            continue
        status, differences, detail = compare_files(
            published_root / relative, work_root / relative, tolerance
        )
        outcome.artifacts.append(ArtifactOutcome(
            stage.name, relative, status, tuple(differences), detail
        ))
    return outcome


def run(
    published_root: Path,
    work_root: Path,
    tier: Tier = Tier.VERIFY,
    only: tuple[str, ...] = (),
) -> VerificationReport:
    """Re-derive every stage of a tier and compare against the published artifacts."""
    published_root = published_root.resolve()
    work_root = work_root.resolve()
    if published_root == work_root:
        raise RootsCoincide(
            f"the regeneration target and the published tree are both {work_root}. Each "
            "file would be compared against itself and the tier would pass "
            "unconditionally, which is the only way a check like this fails."
        )
    if work_root.is_relative_to(published_root):
        raise RootsCoincide(
            f"{work_root} sits inside the published tree at {published_root}, so "
            "regenerating would add files to what is being verified"
        )

    work_root.mkdir(parents=True, exist_ok=True)
    report = VerificationReport(tier=tier, published_root=published_root,
                                work_root=work_root)
    report.seeded = seed_inputs(work_root, published_root, tier)

    if tier is Tier.SMOKE:
        # A different question, so a different check. This tier cannot compare values —
        # at a 200,000-token budget the perplexity is meaningless and the tier's own
        # declaration says it is not compared — so what it verifies is that the training
        # path ran and produced a manifest the benchmark assembly would accept.
        report.stages.append(run_smoke(work_root))
        return report

    for stage in STAGES:
        if stage.tier is not tier:
            continue
        if only and stage.name not in only:
            continue
        report.stages.append(run_stage(stage, work_root, published_root))
    return report


#: What the benchmark assembly needs from a manifest before a run can become a row. Taken
#: from what ``benchmark.assemble`` actually reads, so the smoke tier fails on a manifest
#: the campaign would reject — in minutes, rather than after twelve days of measurement.
MANIFEST_REQUIREMENTS: dict[str, str] = {
    "usable_for_benchmark": (
        "assemble drops every manifest without it, so a run missing this field measured "
        "something and contributed nothing"
    ),
    "genotype": "the five gene values; a malformed one is excluded as malformed_genotype",
    "objectives": "where the measured perplexity lives",
    "precision": "part of the comparability key; rows differing in it are refused",
    "data_vintage": "part of the comparability key",
    "tokeniser_fingerprint": (
        "part of the comparability key. Perplexity per token is not comparable across "
        "tokenisers, so a row without it cannot be placed beside another"
    ),
    "device": "recorded so a result names the hardware it was measured on",
    "config": "carries the seed, which together with the genes identifies the measurement",
}


class ManifestIncomplete(RuntimeError):
    """Raised when a run wrote a manifest the benchmark assembly would reject."""


def check_manifest(path: Path) -> list[str]:
    """Fields the assembly needs and this manifest does not have.

    The smoke tier's real question. It cannot check the numbers — at a two-hundred-thousand
    token budget the perplexity is meaningless and the tier's own declaration says it is not
    compared — so what it checks is that the run produced something the campaign could use.
    A field missing here is a row silently absent from the benchmark twelve days later,
    which is the failure this tier exists to catch early.
    """
    if not path.exists():
        return [f"{path} was not written at all"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path} is not valid JSON: {exc}"]

    missing = []
    for field_name, why in MANIFEST_REQUIREMENTS.items():
        if field_name not in payload:
            missing.append(f"{field_name}: {why}")
            continue
        if field_name == "genotype":
            values = (payload["genotype"] or {}).get("values") or []
            if len(values) != 5:
                missing.append(
                    f"genotype.values has {len(values)} entries, not five; assemble would "
                    "count this as malformed_genotype"
                )
    return missing


def run_smoke(work_root: Path) -> StageOutcome:
    """Run the training entry point at a tiny budget and check what it wrote.

    No comparison against published values, deliberately. What this proves is that the
    training path runs end to end and produces a manifest the benchmark would accept —
    which is a different claim from reproducing a measurement, and the tier's declaration
    says so rather than implying the stronger one.
    """
    stage = next(s for s in STAGES if s.tier is Tier.SMOKE)
    environment = {**os.environ, "NAS4TG_ARTIFACTS": str(work_root)}
    command = stage.command.replace("python -m", f"{sys.executable} -m", 1)
    command = command.replace("artifacts/smoke", str(work_root / "smoke"))

    completed = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        env=environment, cwd=Path.cwd(),
    )
    outcome = StageOutcome(
        stage=stage, ran=True, exit_code=completed.returncode,
        stderr_tail=completed.stderr.strip().splitlines()[-1]
        if completed.stderr.strip() else "",
    )

    manifest = work_root / "smoke" / "manifest.json"
    missing = check_manifest(manifest)
    outcome.artifacts.append(ArtifactOutcome(
        stage.name, "smoke/manifest.json",
        "matched" if not missing else "differs",
        tuple(Difference(entry.split(":")[0], "required", "<absent>") for entry in missing),
        detail="" if not missing else "; ".join(missing),
    ))
    return outcome
