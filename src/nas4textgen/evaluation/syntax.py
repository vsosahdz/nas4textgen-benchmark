"""Does the generated code parse? The one quality signal that cannot be negotiated.

Every other measure in this project is a similarity heuristic with a threshold someone
chose, and the prior work's central failure was exactly that: a number that moved for
reasons unrelated to quality, reported as if it measured quality. A parser has no
threshold. It does not reward verbosity, cannot be inflated by copying the prompt, and has
no trivial floor above zero — empty output does not parse, and neither does plausible-
looking text that is not a program.

That last property is what earns this a place in the battery. It is the only metric here
whose floor is provably zero rather than measured to be low.

Two numbers are reported, and conflating them would hide the interesting case:

- **parse rate**: of the samples that were generated, how many are valid Python.
- **compile rate**: of those that parse, how many also survive compilation, which catches
  a smaller class of errors the grammar accepts.

A model that emits one perfect line and then stops scores well on both while being useless,
so length is reported beside them rather than folded into them.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SyntaxOutcome:
    """One generated sample, and what a parser made of it."""

    parses: bool
    compiles: bool
    error: str | None
    characters: int
    lines: int


def check_sample(source: str) -> SyntaxOutcome:
    """Parse and compile one sample, recording why it failed rather than only that it did.

    Empty output is not valid: ``ast.parse("")`` succeeds, because an empty module is a
    legal program. Accepting it would hand this metric the trivial floor it exists to avoid
    — a model that emits nothing would score a perfect parse rate.
    """
    stripped = source.strip()
    if not stripped:
        return SyntaxOutcome(
            parses=False,
            compiles=False,
            error="empty output: an empty module parses, so it is refused explicitly",
            characters=0,
            lines=0,
        )

    characters = len(source)
    lines = len(stripped.splitlines())

    try:
        tree = ast.parse(stripped)
    except SyntaxError as exc:
        return SyntaxOutcome(False, False, f"SyntaxError: {exc.msg}", characters, lines)
    except (ValueError, MemoryError, RecursionError) as exc:
        # Null bytes, absurd nesting depth. A generator can produce these and they are not
        # syntax errors, but they are not valid programs either.
        return SyntaxOutcome(False, False, f"{type(exc).__name__}: {exc}", characters, lines)

    # A parsed tree is not necessarily compilable: the grammar admits things the compiler
    # rejects, such as `return` outside a function. Reported separately because the gap
    # between the two is itself informative about what a model has learned.
    try:
        compile(tree, "<generated>", "exec")
    except (SyntaxError, ValueError) as exc:
        return SyntaxOutcome(True, False, f"{type(exc).__name__}: {exc}", characters, lines)

    return SyntaxOutcome(True, True, None, characters, lines)


@dataclass(frozen=True)
class SyntaxReport:
    samples: int
    parse_rate: float
    compile_rate: float
    empty_rate: float
    median_lines: float
    error_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "parse_rate": round(self.parse_rate, 6),
            "compile_rate": round(self.compile_rate, 6),
            "empty_rate": round(self.empty_rate, 6),
            "median_lines": self.median_lines,
            "error_counts": self.error_counts,
            "floor": 0.0,
            "floor_note": (
                "Provably zero rather than measured: empty output is refused, and text "
                "that is not a program does not parse. This is the only metric in the "
                "battery whose trivial baseline needs no measurement."
            ),
        }


def evaluate(samples: Sequence[str]) -> SyntaxReport:
    """Parse rate over a set of generated samples, with the failures counted by kind."""
    if not samples:
        raise ValueError(
            "no samples to evaluate. An empty set would report a parse rate of zero, "
            "which is indistinguishable from a model that produced nothing valid."
        )

    outcomes = [check_sample(sample) for sample in samples]
    parsed = [o for o in outcomes if o.parses]

    errors: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.error:
            # Group by kind, not by message: "SyntaxError: invalid syntax" repeated four
            # hundred times says less than knowing it is the only failure mode.
            kind = outcome.error.split(":", 1)[0]
            errors[kind] = errors.get(kind, 0) + 1

    line_counts = sorted(o.lines for o in outcomes)
    middle = len(line_counts) // 2
    median = (
        float(line_counts[middle])
        if len(line_counts) % 2
        else (line_counts[middle - 1] + line_counts[middle]) / 2
    )

    return SyntaxReport(
        samples=len(outcomes),
        parse_rate=len(parsed) / len(outcomes),
        # Conditioned on parsing, not on being generated. A model whose output rarely
        # parses would otherwise get a flattering compile rate for the few that did.
        compile_rate=(
            sum(o.compiles for o in parsed) / len(parsed) if parsed else 0.0
        ),
        empty_rate=sum(not o.characters for o in outcomes) / len(outcomes),
        median_lines=median,
        error_counts=dict(sorted(errors.items())),
    )
