"""Re-derivation of the prior protocol's METEOR figures, from the corpus that produced them.

Section 5 of the manuscript turns on two measurements: under the prior protocol an **empty**
generation scores METEOR 0.3088 on the real test split, and 0.6712 on the section headers
that make up a quarter of the evaluated rows. The claim built on them is that the reported
improvement of +0.024 is smaller than the metric's own range over text containing no model.
An earlier, independent measurement of the same two quantities gave 0.3067 and 0.6699; this
module reproduces them to within 0.002, which is what a second derivation is for.

Those figures were measured and were not reproducible from this repository. The regression
fixture in ``tests/test_audit_fixture.py`` carried them as constants, which stops the defect
from returning but does not establish the numbers; nothing in the pipeline derived them. For
a paper whose argument is that unverifiable numbers are how the prior result survived review,
leaving its own central numbers unverifiable is the one inconsistency it cannot afford.

**Two defects, separated.** The prior protocol cut each row at 70 per cent of its
*characters* — ``cut_point = int(len(text) * 0.7)`` — which lands mid-word, and then scored
the prompt and the continuation concatenated against the reference, so the hypothesis
contained text the model did not produce. Those are independent faults, and the three
protocols here isolate them: ``prior`` is what the prior work ran, ``prompt_removed`` changes
only whether the prompt is scored, and ``corrected`` additionally cuts on token boundaries
and restricts scoring to rows whose reference is long enough to carry a score. The difference
between the first two is the prompt defect and nothing else.

**Why the floors rather than a model.** Reporting that some architecture scores 0.72 says
nothing until something with no model in it is scored the same way. Every hypothesis here is
produced without consulting a model; where those score near the published result, the
published result is not evidence about a model.

Reconstructing the cut as a token boundary rather than a character offset moved the headline
figure from 0.3067 to 0.2119 and the markup figure from 0.6699 to 0.2830 — which is the
measurement this module exists to prevent anyone, including its author, from getting wrong
twice.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .corpora.hygiene import RowKind, classify, prepare

#: The corpus the prior work evaluated on. WikiText-2, not the 103 variant this project
#: searches over: the audit has to run on what produced the number being audited.
PRIOR_IDENTIFIER = "Salesforce/wikitext"
PRIOR_CONFIG = "wikitext-2-raw-v1"
PRIOR_SPLIT = "test"

#: The improvement the prior work reports as its contribution, from 0.6961 to 0.7200.
PRIOR_CLAIMED_IMPROVEMENT = 0.024

#: The share of each row given as the prompt, quoted from the prior work's Figure B.1: "70%
#: of a reference sentence as a prompt and must predict the remaining 30%". Applied to
#: characters, because that is what its implementation did — ``int(len(text) * 0.7)`` on a
#: string — and a reconstruction that cuts on token boundaries instead is measuring this
#: project's corrected protocol under the prior protocol's name.
PRIOR_PROMPT_FRACTION = 0.7

#: What the corrected protocol gives the model to predict, on token boundaries.
CORRECTED_CONTINUATION_FRACTION = 0.3

#: Prompt fractions the sensitivity table reports. The empty-generation figure is almost
#: entirely a function of how much of the row the prompt covers, so a figure quoted without
#: its cut is not a reproducible quantity — which is the point the table makes.
SENSITIVITY_FRACTIONS = (0.5, 0.6, 0.7, 0.8)

#: Function words, as a proxy for a model that has learnt frequency and nothing else. Scored
#: under the same protocol as everything else, which is what makes it a floor rather than an
#: anecdote.
FUNCTION_WORDS = (
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "for",
    "with", "as", "on", "by", "at", "from", "it", "he", "his", "which",
)

#: ``prior`` is what the prior work ran. Each later protocol removes one defect, so the
#: difference between consecutive columns is attributable to that defect alone.
PROTOCOLS = ("prior", "prompt_removed", "corrected")


class CorpusUnavailable(RuntimeError):
    """Raised when the prior corpus cannot be loaded and the audit cannot run."""


@dataclass(frozen=True)
class Case:
    """One evaluated row: what the model was given, and what it was asked to produce."""

    prompt: tuple[str, ...]
    reference: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class HypothesisScore:
    """One hypothesis scored under one protocol."""

    hypothesis: str
    protocol: str
    rows: int
    meteor: float
    by_kind: dict[str, float] = field(default_factory=dict)
    rows_by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "hypothesis": self.hypothesis,
            "protocol": self.protocol,
            "rows": self.rows,
            "meteor": round(self.meteor, 4),
            "by_kind": {k: round(v, 4) for k, v in sorted(self.by_kind.items())},
            "rows_by_kind": dict(sorted(self.rows_by_kind.items())),
        }


@dataclass(frozen=True)
class AuditReport:
    """What each protocol gives for every hypothesis, and what follows from it."""

    scores: tuple[HypothesisScore, ...]
    markup_share: float
    claimed_improvement: float
    rows_evaluated: int
    prompt_fraction: float = PRIOR_PROMPT_FRACTION
    sensitivity: tuple[dict[str, float], ...] = ()

    def score(self, hypothesis: str, protocol: str) -> HypothesisScore | None:
        for item in self.scores:
            if item.hypothesis == hypothesis and item.protocol == protocol:
                return item
        return None

    @property
    def floor_spread(self) -> float:
        """How far apart the model-free hypotheses are from each other.

        What the argument rests on, and the quantity that survives any disagreement about
        the protocol's details. Every hypothesis was produced without a model, so the spread
        between the best and the worst is the metric's own range over text that contains no
        language modelling at all. An improvement smaller than this spread cannot be told
        apart from moving between two ways of writing nothing.
        """
        values = [s.meteor for s in self.scores if s.protocol == "prior"]
        return max(values) - min(values) if values else 0.0

    @property
    def improvement_exceeds_floor_spread(self) -> bool:
        """Whether the prior work's reported gain is larger than that spread."""
        return self.claimed_improvement > self.floor_spread

    @property
    def prompt_defect_contribution(self) -> float:
        """What scoring the prompt is worth to a model that produces nothing.

        The single number that attributes the inflation, because ``prior`` and
        ``prompt_removed`` differ in that one step.
        """
        with_prompt = self.score("empty", "prior")
        without = self.score("empty", "prompt_removed")
        if with_prompt is None or without is None:
            return 0.0
        return with_prompt.meteor - without.meteor

    def as_dict(self) -> dict[str, object]:
        empty = {p: self.score("empty", p) for p in PROTOCOLS}
        return {
            "corpus": {
                "identifier": PRIOR_IDENTIFIER,
                "config": PRIOR_CONFIG,
                "split": PRIOR_SPLIT,
                "rows_evaluated": self.rows_evaluated,
            },
            "protocols": {
                "prior": (
                    f"the row is cut at {self.prompt_fraction:.0%} of its characters and the "
                    "hypothesis is the prompt and the continuation concatenated, scored over "
                    "every non-empty row including markup — what the prior work ran"
                ),
                "prompt_removed": (
                    "the same cut, with the hypothesis being the continuation alone; the "
                    "only difference from `prior` is the defect itself"
                ),
                "corrected": (
                    "cut on token boundaries with "
                    f"{CORRECTED_CONTINUATION_FRACTION:.0%} to predict, hypothesis is the "
                    "continuation alone, scored over prose rows whose reference is long "
                    "enough to carry a score"
                ),
                "documented_by": (
                    "the prior work's Figure B.1: 70% of a reference sentence as a prompt "
                    "and must predict the remaining 30%"
                ),
            },
            "markup_share": round(self.markup_share, 4),
            "scores": [s.as_dict() for s in self.scores],
            "sensitivity": [dict(row) for row in self.sensitivity],
            "finding": {
                "empty_generation_under_prior_protocol":
                    round(empty["prior"].meteor, 4) if empty["prior"] else None,
                "empty_generation_on_markup_rows":
                    round(empty["prior"].by_kind.get("markup", 0.0), 4)
                    if empty["prior"] else None,
                "empty_generation_with_the_prompt_removed":
                    round(empty["prompt_removed"].meteor, 4)
                    if empty["prompt_removed"] else None,
                "empty_generation_under_corrected_protocol":
                    round(empty["corrected"].meteor, 4) if empty["corrected"] else None,
                "prompt_defect_contribution": round(self.prompt_defect_contribution, 4),
                "floor_spread": round(self.floor_spread, 4),
                "claimed_improvement": self.claimed_improvement,
                "improvement_exceeds_floor_spread": self.improvement_exceeds_floor_spread,
            },
            "why_this_exists": (
                "The manuscript's section 5 reports these figures. Before this module they "
                "appeared only as constants in a regression fixture, so the paper's central "
                "measurement was the one number in it that no command re-derived."
            ),
        }


# ── hypotheses, none of which consult a model ──────────────────────────────────────────

def hypothesis_empty(case: Case, **_) -> tuple[str, ...]:
    """The model produces nothing."""
    return ()


def hypothesis_copy_prompt_tail(case: Case, **_) -> tuple[str, ...]:
    """The model repeats the end of its own prompt."""
    length = len(case.reference)
    return tuple(case.prompt[-length:]) if length else ()


def hypothesis_frequent_tokens(case: Case, **_) -> tuple[str, ...]:
    """Function-word spam, at the reference's length."""
    length = len(case.reference)
    return tuple(FUNCTION_WORDS[i % len(FUNCTION_WORDS)] for i in range(length))


def hypothesis_unrelated_prose(case: Case, pool: Sequence[tuple[str, ...]] = (),
                               index: int = 0, **_) -> tuple[str, ...]:
    """Competent English, written about something else.

    The upper member of the comparison. It is real prose, so a metric read as a measure of
    language quality should rank it above function-word spam; whether it does is the finding.

    The offset is half the pool and not zero. Walking the pool in step with the rows hands
    each row its own reference back, which scores a perfect 0.9999 and measures nothing but
    the indexing — the first version of this function did exactly that.
    """
    if not pool:
        return ()
    chosen = pool[(index + len(pool) // 2) % len(pool)]
    if chosen == case.reference and len(pool) > 1:
        chosen = pool[(index + 1) % len(pool)]
    return tuple(chosen)


HYPOTHESES: dict[str, Callable[..., tuple[str, ...]]] = {
    "empty": hypothesis_empty,
    "copy_prompt_tail": hypothesis_copy_prompt_tail,
    "frequent_tokens": hypothesis_frequent_tokens,
    "unrelated_prose": hypothesis_unrelated_prose,
}


# ── the evaluated rows, under each protocol's own cut ──────────────────────────────────

def prior_cases(rows: Sequence[str],
                prompt_fraction: float = PRIOR_PROMPT_FRACTION) -> list[Case]:
    """Rows cut the way the prior implementation cut them: by character offset.

    Empty rows are skipped, as they were there; everything else is kept, markup included.
    The cut lands mid-word often enough to matter — ``" = Robert Boulter = "`` becomes a
    prompt of ``" = Robert Boul"`` — and that is a property of the protocol being audited,
    not a flaw in this reconstruction of it.
    """
    cases: list[Case] = []
    for row in rows:
        kind = classify(row)
        if kind is RowKind.EMPTY:
            continue
        cut = int(len(row) * prompt_fraction)
        prompt, reference = row[:cut].split(), row[cut:].split()
        if not prompt or not reference:
            continue
        cases.append(Case(tuple(prompt), tuple(reference), kind.value))
    return cases


def corrected_cases(rows: Sequence[str], encode: Callable[[str], list[int]],
                    decode: Callable[[Sequence[int]], str]) -> list[Case]:
    """Rows under this project's protocol: token boundaries, and only scorable rows."""
    splits, _ = prepare(list(rows), encode,
                        continuation_fraction=CORRECTED_CONTINUATION_FRACTION)
    return [
        Case(tuple(decode(s.prompt_ids).split()),
             tuple(decode(s.reference_ids).split()),
             s.kind.value)
        for s in splits if s.usable_for_overlap
    ]


# ── scoring ────────────────────────────────────────────────────────────────────────────

def _meteor(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not hypothesis or not reference:
        # NLTK's aligner divides by the hypothesis length. An empty hypothesis scores zero
        # by definition, and saying so here keeps the floor from being a crash.
        return 0.0
    from nltk.translate.meteor_score import meteor_score

    return float(meteor_score([list(reference)], list(hypothesis)))


def score_hypothesis(cases: Sequence[Case], hypothesis: str, protocol: str,
                     pool: Sequence[tuple[str, ...]] = ()) -> HypothesisScore:
    """Score one hypothesis over the evaluated rows under one protocol."""
    generate = HYPOTHESES[hypothesis]
    include_prompt = protocol == "prior"

    per_kind: dict[str, list[float]] = {}
    scores: list[float] = []
    for index, case in enumerate(cases):
        continuation = generate(case, pool=pool, index=index)
        words = (tuple(case.prompt) + continuation if include_prompt else continuation)
        value = _meteor(case.reference, words)
        scores.append(value)
        per_kind.setdefault(case.kind, []).append(value)

    return HypothesisScore(
        hypothesis=hypothesis,
        protocol=protocol,
        rows=len(cases),
        meteor=statistics.fmean(scores) if scores else 0.0,
        by_kind={k: statistics.fmean(v) for k, v in per_kind.items()},
        rows_by_kind={k: len(v) for k, v in per_kind.items()},
    )


def audit(
    rows: Sequence[str],
    encode: Callable[[str], list[int]],
    decode: Callable[[Sequence[int]], str],
    max_rows: int | None = None,
    prompt_fraction: float = PRIOR_PROMPT_FRACTION,
    with_sensitivity: bool = True,
) -> AuditReport:
    """Run every hypothesis under every protocol over the prior work's evaluation rows."""
    prior = prior_cases(rows, prompt_fraction)
    corrected = corrected_cases(rows, encode, decode)
    if max_rows is not None:
        prior, corrected = prior[:max_rows], corrected[:max_rows]
    if not prior:
        raise CorpusUnavailable("no rows survived preparation; nothing to score")

    cases_for = {"prior": prior, "prompt_removed": prior, "corrected": corrected}
    pools = {name: tuple(c.reference for c in group if c.kind == RowKind.PROSE.value)
             for name, group in cases_for.items()}

    scores = [
        score_hypothesis(cases_for[protocol], hypothesis, protocol, pool=pools[protocol])
        for protocol in PROTOCOLS
        for hypothesis in HYPOTHESES
        if cases_for[protocol]
    ]

    kinds = Counter(case.kind for case in prior)
    return AuditReport(
        scores=tuple(scores),
        markup_share=kinds[RowKind.MARKUP.value] / len(prior),
        claimed_improvement=PRIOR_CLAIMED_IMPROVEMENT,
        rows_evaluated=len(prior),
        prompt_fraction=prompt_fraction,
        sensitivity=sensitivity(rows, max_rows=max_rows) if with_sensitivity else (),
    )


def sensitivity(
    rows: Sequence[str],
    fractions: Sequence[float] = SENSITIVITY_FRACTIONS,
    max_rows: int | None = None,
) -> tuple[dict[str, float], ...]:
    """What the empty-generation figure does as the prompt covers more of the row.

    Reported because the figure moves a long way: the prompt is the whole hypothesis when
    the model produces nothing, so a score quoted without the cut it was measured at is not
    a reproducible quantity. The conclusion does not move with it — there is no setting at
    which writing nothing scores zero under the prior protocol.
    """
    table: list[dict[str, float]] = []
    for fraction in fractions:
        cases = prior_cases(rows, fraction)
        if max_rows is not None:
            cases = cases[:max_rows]
        if not cases:
            continue
        empty = score_hypothesis(cases, "empty", "prior")
        table.append({
            "prompt_fraction": fraction,
            "rows": empty.rows,
            "empty_generation": round(empty.meteor, 4),
            "on_markup_rows": round(empty.by_kind.get("markup", 0.0), 4),
        })
    return tuple(table)


def load_prior_rows(limit: int | None = None) -> list[str]:
    """The prior work's evaluation split, from the dataset it names."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise CorpusUnavailable(f"datasets is not installed: {exc}") from exc

    try:
        data = load_dataset(PRIOR_IDENTIFIER, PRIOR_CONFIG, split=PRIOR_SPLIT)
    except Exception as exc:  # noqa: BLE001
        raise CorpusUnavailable(f"{type(exc).__name__}: {exc}") from exc

    rows = list(data["text"])
    return rows[:limit] if limit is not None else rows
