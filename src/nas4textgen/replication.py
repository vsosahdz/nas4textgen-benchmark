"""What can be reproduced, at what cost, and how exactly.

A replication claim that is not qualified is a replication claim that will be wrong for
someone. This project's stages differ enormously in what they promise: enumerating the
search space is arithmetic and gives the same bits on any machine, while training a
transformer on a GPU does not give the same bits twice on the *same* machine. Publishing
both under the word "reproducible" would be true of one and misleading about the other.

So every stage declares three things.

**Its determinism class.** Bit-exact, or tolerance-exact against a named ROPE derived from
measured noise. The second is not a weaker version of the first; it is a different claim,
and the tolerance came from an experiment rather than from convention.

**Its tier.** How much hardware and time it needs. A reader with a laptop can run the whole
verification tier; the full sweep needs a GPU cluster for days, and saying so up front is
more useful than letting someone discover it.

**Its command.** One documented invocation, with what it reads and what it writes, so a
reader can run a stage without reconstructing it from the source.

The tiers are nested: verify runs on any CPU in minutes, smoke adds one GPU and a handful
of architectures, partial adds one wave, full is the campaign. Each is a real check rather
than a demonstration — the verification tier recomputes the published tables from the
published benchmark and fails if they differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Determinism(str, Enum):  # noqa: UP042 — the value is serialised into the manifest
    """What re-running a stage promises."""

    BIT_EXACT = "bit_exact"
    TOLERANCE_EXACT = "tolerance_exact"


class Tier(str, Enum):  # noqa: UP042 — same reason
    """What a stage costs to run."""

    VERIFY = "verify"
    SMOKE = "smoke"
    PARTIAL = "partial"
    FULL = "full"


#: What each tier needs, and roughly how long. Stated so a reader can decide before
#: starting rather than after.
#:
#: `compact` holds the short forms the table in the paper prints, beside the full forms
#: the replication document prints. Both are read from here so that the two cannot
#: describe the same tier differently, which they already had: the paper promised the
#: verification tier in under two minutes where this dictionary says ten. Hardware and
#: cost have no short form, because the table prints the hardware in full and does not
#: print the cost at all.
TIER_REQUIREMENTS: dict[Tier, dict] = {
    Tier.VERIFY: {
        "hardware": "any CPU, no GPU, no cloud account",
        "compact": {"wall_clock": "under ten minutes",
                    "establishes": "the analysis, recomputed from the published benchmark "
                                   "table; nothing is retrained"},
        "wall_clock": "under ten minutes",
        "cost": "nothing",
        "establishes": (
            "the analysis: every table and figure recomputes from the published benchmark "
            "table and matches. Nothing is retrained"
        ),
        "what_it_proves": (
            "every table and figure in the paper is recomputed from the published "
            "benchmark table and matches. It does not retrain anything, so it verifies the "
            "analysis rather than the measurements."
        ),
    },
    Tier.SMOKE: {
        "hardware": "one GPU, or a CPU with patience",
        "compact": {"wall_clock": "minutes",
                    "establishes": "the training path, run end to end, and a manifest with "
                                   "every field the benchmark assembly reads"},
        "wall_clock": "minutes",
        "cost": "negligible",
        "establishes": (
            "the training path: it runs end to end and writes a manifest carrying every "
            "field the benchmark assembly reads"
        ),
        "what_it_proves": (
            "the training path runs end to end and writes a manifest with every required "
            "field. The perplexities are meaningless at this budget and are not compared."
        ),
    },
    Tier.PARTIAL: {
        "hardware": "one GPU",
        "compact": {"wall_clock": "about a day",
                    "establishes": "one nested wave, which is a space-filling sample of the "
                                   "space on its own"},
        "wall_clock": "about a day for 150 architectures at 100M tokens",
        "cost": "roughly 26 USD at low-priority cloud pricing",
        "establishes": (
            "one nested wave: wave one is a space-filling sample on its own, enough for "
            "the regressor study and an interim front"
        ),
        "what_it_proves": (
            "one nested wave reproduces. Because the waves nest, wave one is a "
            "space-filling sample on its own and supports the surrogate study and an "
            "interim front."
        ),
    },
    Tier.FULL: {
        "hardware": "a GPU cluster; the campaign used five T4 nodes",
        "compact": {"wall_clock": "about twelve days",
                    "establishes": "the whole benchmark, tolerance-exact against the "
                                   "measured seed ROPE"},
        "wall_clock": "about twelve days at five concurrent nodes",
        "cost": "roughly 174 USD at the measured low-priority price",
        "establishes": (
            "the measurements: the whole benchmark, tolerance-exact against the measured "
            "seed ROPE and not bit-exact"
        ),
        "what_it_proves": (
            "the whole benchmark. Tolerance-exact against the measured seed ROPE, not "
            "bit-exact: GPU kernel scheduling is nondeterministic and the campaign's own "
            "re-runs of identical architectures moved perplexity by 0.07 to 0.33 per cent."
        ),
    },
}


#: What each tier assumes the machine running it has. The prose above says the same thing
#: in words; this is the machine-readable half, and the two are checked against each other
#: by a test — a tier that promised "no cloud account" while running a stage that needs one
#: would be a promise nobody could keep.
#:
#: A stage is executed only when the tier provides everything it requires. That comparison
#: is what makes `requires` mean the right thing in both directions: the benchmark table
#: needs an account, so the verification tier treats it as an input, and the partial tier —
#: which assumes an account — runs it.
TIER_PROVIDES: dict[Tier, frozenset[str]] = {
    Tier.VERIFY: frozenset(),
    Tier.SMOKE: frozenset({"gpu"}),
    Tier.PARTIAL: frozenset({"gpu", "cloud", "network"}),
    Tier.FULL: frozenset({"gpu", "cloud", "network"}),
}


@dataclass(frozen=True)
class Stage:
    """One documented step of the pipeline."""

    name: str
    command: str
    tier: Tier
    determinism: Determinism
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    #: What must already be true. A stage that silently produces nothing because its input
    #: was missing is worse than one that refuses.
    preconditions: tuple[str, ...] = ()
    #: Whether re-running after an interruption resumes rather than restarting.
    resumable: bool = False
    #: What re-running this stage needs that the machine may not have: "network" to reach
    #: a dataset host, "cloud" for an account holding the run manifests, "gpu" for an
    #: accelerator. A stage that needs any of these cannot be re-derived by a reader whose
    #: tier promises none of them, so the verification tier treats its outputs as inputs
    #: and says so, rather than counting an unreproduced artifact as reproduced.
    requires: tuple[str, ...] = ()
    tolerance: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.determinism is Determinism.TOLERANCE_EXACT and not self.tolerance:
            raise ValueError(
                f"{self.name} claims tolerance-exact reproduction without naming the "
                "tolerance. A tolerance nobody states is a tolerance nobody can check."
            )
        if self.determinism is Determinism.BIT_EXACT and self.tolerance:
            raise ValueError(
                f"{self.name} claims bit-exact reproduction and also names a tolerance; "
                "one of the two is wrong."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.name,
            "command": self.command,
            "tier": self.tier.value,
            "determinism": self.determinism.value,
            "tolerance": self.tolerance or None,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "preconditions": list(self.preconditions),
            "resumable": self.resumable,
            "requires": list(self.requires),
            "notes": self.notes,
        }


#: Every stage, in the order a reader would run them.
STAGES: tuple[Stage, ...] = (
    Stage(
        name="structural gate",
        command="python -m nas4textgen.cli gate",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        reads=("the search space definition in code",),
        writes=("artifacts/gate/structural_gate.json",),
        notes=(
            "Eight checks the space must pass before any sweep is submitted: cardinality, "
            "injectivity of the encoding, no inert genes, the reference architecture "
            "dominated. Arithmetic on integers, so any machine gives the same file."
        ),
    ),
    Stage(
        name="audit of the prior protocol",
        command="python -m nas4textgen.cli audit",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        reads=("the WikiText-2 test split, retrieved on first use",),
        writes=("artifacts/audit/prior_protocol.json",),
        requires=("network",),
        notes=(
            "Re-derives the figures section 5 rests on: what an empty generation scores "
            "under the prior protocol, on which strata, and how far apart four model-free "
            "hypotheses are. CPU only and a few seconds. Listed in the verify tier because "
            "a reader who can run nothing else should still be able to check the claim the "
            "paper makes about someone else's work."
        ),
    ),
    Stage(
        name="enumeration and exact cost front",
        command="python -m nas4textgen.cli enumerate",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        reads=("the search space definition in code",),
        writes=("artifacts/space/enumeration.csv", "artifacts/space/enumeration_summary.json"),
        notes="Closed-form parameter and cache counts for all 1,099 architectures.",
    ),
    Stage(
        name="sampling designs and execution order",
        command="python -m nas4textgen.cli designs",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        reads=("artifacts/space/enumeration.csv",),
        writes=("artifacts/space/sampling_designs.json",),
        notes=(
            "The maximin order the sweep runs in, and the nested waves. Seeded, and the "
            "order is recorded rather than recomputed at submission so a later change to "
            "this code cannot reorder a campaign already in flight."
        ),
    ),
    Stage(
        name="corpus retrieval",
        command="python -m nas4textgen.cli fetch",
        tier=Tier.VERIFY,
        determinism=Determinism.TOLERANCE_EXACT,
        tolerance=(
            "the corpora are fetched from upstream repositories that can change. The data "
            "vintage is recorded in every manifest and two results are comparable only if "
            "they share one. A fetch on a later date is a different experiment, not a "
            "failed reproduction."
        ),
        reads=("the corpus registry in code",),
        requires=("network",),
        writes=("data/cache/<corpus>/<split>.jsonl", "artifacts/corpora/retrieval.json"),
        preconditions=("network access to the dataset hosts",),
        notes="The only stage that reaches the public internet. Everything after reads the cache.",
    ),
    Stage(
        name="tokenisation",
        command="python -m nas4textgen.cli tokenise",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        reads=("data/cache/<corpus>/<split>.jsonl",),
        writes=("data/cache/<corpus>/<split>.tokens.jsonl",),
        preconditions=("the corpus is cached and its hash matches",),
        notes=(
            "Once, rather than once per job. Fixes the token stream so a rebuild that moved "
            "the tokeniser cannot change it under half a sweep."
        ),
    ),
    Stage(
        name="one training run",
        command=(
            "python -m nas4textgen.train --h 2 --d-head 32 --n 2 --ffn 2 --kv 1 "
            "--corpus wikitext103 --token-budget 200000 --allow-cpu "
            "--output artifacts/smoke"
        ),
        tier=Tier.SMOKE,
        determinism=Determinism.TOLERANCE_EXACT,
        tolerance=(
            "bit-exact on CPU, where deterministic kernels are enabled and cost nothing. On "
            "GPU, tolerance-exact against a seed ROPE of 0.651 perplexity measured from "
            "five replicates of one architecture."
        ),
        reads=("the tokenised cache",),
        writes=("artifacts/smoke/manifest.json",),
        preconditions=("the corpus is tokenised", "the output directory is writable"),
        resumable=True,
        notes=(
            "Checkpoints every ten minutes or longer, sized so the writing stays under five "
            "per cent of runtime. Resuming reproduces the uninterrupted result exactly on "
            "CPU."
        ),
    ),
    Stage(
        name="one sweep wave",
        command="python -m nas4textgen.cli sweep --wave 1 --submit",
        tier=Tier.PARTIAL,
        determinism=Determinism.TOLERANCE_EXACT,
        tolerance="the same seed ROPE, applied per architecture",
        reads=("the corpus cache in project storage", "the published training image"),
        writes=("one manifest per architecture in the artifact container",),
        requires=("cloud", "gpu"),
        preconditions=(
            "the platform is applied",
            "the corpus is uploaded and verified",
            "the image matches this checkout",
        ),
        resumable=True,
        notes=(
            "Waves nest, so wave one is a space-filling sample on its own — fill distance "
            "0.333 against 0.594 for a uniform random sample of the same size."
        ),
    ),
    Stage(
        name="benchmark table and exact front",
        command="python -m nas4textgen.cli table",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        reads=("artifacts/manifests/runs.jsonl",),
        writes=("artifacts/benchmark/table.csv", "artifacts/benchmark/summary.json"),
        preconditions=("at least one run met its token budget",),
        notes=(
            "Refuses rows measured under different precision, corpus vintage, token budget "
            "or tokeniser. Given the same manifests it produces the same table, and the "
            "manifests are published with it, so the verification tier rebuilds the table "
            "rather than taking it on trust. It did take it on trust: the stage still "
            "declared that it reads the manifests from the artifact container and so needs "
            "an account, which had stopped being true when the command was changed to read "
            "the published mirror. Nothing rebuilt the table for as long as that stood, and "
            "an assembly bug that moved the measured front from 220 architectures to 218 "
            "went unreported."
        ),
    ),
    Stage(
        name="surrogate study",
        command="python -m nas4textgen.cli surrogate",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        reads=("artifacts/benchmark/table.csv",),
        writes=(
            "artifacts/surrogate/slate_at_<rows>.json",
            "artifacts/surrogate/budget_curve.json",
        ),
        preconditions=("the benchmark table exists",),
        notes=(
            "Fourteen candidates on shared folds, every one scored against two trivial "
            "predictors and disqualified if it beats neither. Seeded throughout."
        ),
    ),
    Stage(
        name="characterisation",
        command="python -m nas4textgen.cli characterise",
        tier=Tier.VERIFY,
        determinism=Determinism.BIT_EXACT,
        # The pilot's ROPE is an input, not a flag default. The depth-versus-width verdict
        # is a statement about whether a difference exceeds measured noise, so the noise
        # has to come from the run that measured it — and declaring the dependency is what
        # lets the verification tier put it where the command looks.
        reads=("artifacts/benchmark/table.csv", "artifacts/pilot/results.json"),
        writes=("artifacts/characterisation/",),
        preconditions=("the benchmark table exists",),
        notes=(
            "Depth against width at matched parameter count, the grouped-query frontier, "
            "gene effects on measured quality, and whether the two cost fronts differ."
        ),
    ),
    Stage(
        name="the remaining waves",
        command="python -m nas4textgen.cli sweep --wave 5 --submit",
        tier=Tier.FULL,
        determinism=Determinism.TOLERANCE_EXACT,
        tolerance=(
            "the same seed ROPE as one wave, applied per architecture. GPU kernel "
            "scheduling is nondeterministic and the campaign's own re-runs of identical "
            "architectures moved perplexity by 0.07 to 0.33 per cent."
        ),
        reads=("the corpus cache in project storage", "the published training image"),
        writes=("one manifest per architecture in the artifact container",),
        preconditions=(
            "one wave has completed, so the throughput model exists",
            "the image matches this checkout",
        ),
        resumable=True,
        requires=("cloud", "gpu"),
        notes=(
            "The waves nest, so submitting the last one submits whatever the earlier ones "
            "did not cover — re-running it after an interruption skips every architecture "
            "already measured rather than paying for it again. Declared as a stage because "
            "the full tier quoted a cost and a duration for work no stage named, and a "
            "tier with no stages reports success for having done nothing."
        ),
    ),
)


def stages_for(tier: Tier) -> tuple[Stage, ...]:
    """Every stage a given tier runs, including the cheaper tiers it contains.

    The tiers nest: running the partial tier means running everything verify runs plus the
    smoke run plus one wave. Listing them separately would let a reader run a wave without
    the checks that make its output interpretable.
    """
    order = (Tier.VERIFY, Tier.SMOKE, Tier.PARTIAL, Tier.FULL)
    depth = order.index(tier)
    included = set(order[: depth + 1])
    return tuple(stage for stage in STAGES if stage.tier in included)


def manifest() -> dict[str, object]:
    """The whole declaration, for publication beside the artifacts."""
    return {
        "tiers": {
            tier.value: {
                **requirements,
                "stages": [s.name for s in STAGES if s.tier is tier],
            }
            for tier, requirements in TIER_REQUIREMENTS.items()
        },
        "stages": [stage.as_dict() for stage in STAGES],
        "determinism_summary": {
            "bit_exact": [
                s.name for s in STAGES if s.determinism is Determinism.BIT_EXACT
            ],
            "tolerance_exact": [
                s.name for s in STAGES if s.determinism is Determinism.TOLERANCE_EXACT
            ],
        },
        "why_two_classes": (
            "Enumerating the space is arithmetic and gives the same bits anywhere. Training "
            "a transformer on a GPU does not give the same bits twice on the same machine. "
            "Publishing both as 'reproducible' would be true of one and misleading about "
            "the other, so each stage says which it is and, where it is the second, names "
            "the tolerance and where the tolerance came from."
        ),
    }


def render_markdown() -> str:
    """The replication instructions, generated from the declaration above.

    Generated rather than written, because a document and a program that describe the same
    thing drift apart the moment one is edited. The tiers, commands, tolerances and costs a
    reader sees are the ones the code enforces.
    """
    lines = [
        "# Reproducing this benchmark",
        "",
        "Generated by `python -m nas4textgen.cli replication`. Do not edit by hand — the",
        "declaration lives in `src/nas4textgen/replication.py`, and a document maintained",
        "separately from the code it describes is a document that will disagree with it.",
        "",
        "## What each tier costs and what it proves",
        "",
        "The tiers nest: `partial` runs everything `smoke` runs, which runs everything",
        "`verify` runs. Running a wave without the checks that make its output",
        "interpretable is a demonstration rather than a replication.",
        "",
    ]

    for tier in (Tier.VERIFY, Tier.SMOKE, Tier.PARTIAL, Tier.FULL):
        needs = TIER_REQUIREMENTS[tier]
        lines += [
            f"### `{tier.value}`",
            "",
            f"- **hardware** — {needs['hardware']}",
            f"- **wall clock** — {needs['wall_clock']}",
            f"- **cost** — {needs['cost']}",
            f"- **proves** — {needs['what_it_proves']}",
            "",
        ]

    lines += [
        "## Determinism, per stage",
        "",
        "Two classes, and the difference is not a matter of degree. Enumerating the space",
        "is arithmetic and gives the same bytes on any machine. Training a transformer on",
        "a GPU does not give the same bytes twice on the same machine — the campaign's own",
        "re-runs of identical architectures at identical seeds moved perplexity by 0.07 to",
        "0.33 per cent. Publishing both as \"reproducible\" would be true of one and",
        "misleading about the other.",
        "",
        "| stage | tier | reproduces | tolerance |",
        "|---|---|---|---|",
    ]
    for stage in STAGES:
        tolerance = stage.tolerance.split(".")[0] if stage.tolerance else "exactly"
        lines.append(
            f"| {stage.name} | `{stage.tier.value}` | {stage.determinism.value} | {tolerance} |"
        )

    lines += ["", "## The stages", ""]
    for stage in STAGES:
        lines += [
            f"### {stage.name}",
            "",
            "```",
            stage.command,
            "```",
            "",
            f"- **reads** — {', '.join(stage.reads)}",
            f"- **writes** — {', '.join(stage.writes)}",
        ]
        if stage.preconditions:
            lines.append(f"- **requires** — {'; '.join(stage.preconditions)}")
        if stage.resumable:
            lines.append("- **resumable** — re-running after an interruption continues")
        if stage.notes:
            lines += ["", stage.notes]
        lines.append("")

    lines += [
        "## Authorship",
        "",
        "Every commit, contributor file and citation record in this repository names human",
        "researchers only. No automated tool is listed as an author or co-author anywhere,",
        "and a test asserts it.",
        "",
    ]
    return "\n".join(lines)
