"""Repetition, and the fact that overlap metrics reward it.

At this parameter scale, the characteristic failure is not incoherence but degeneration:
the model finds a token sequence it likes and emits it forever. This matters here for a
specific reason. An n-gram overlap metric computed against a reference will happily reward
a model that repeats a common phrase, because some of those n-grams do appear in the
reference. So the metric goes up while the text gets worse, which is the same class of
error this project exists to correct — a number moving for reasons unrelated to what it
claims to measure.

Reported alongside every generation score rather than as a separate study. A quality number
without a repetition number beside it is not interpretable at this scale.

Nothing here needs the model, only its output, so it runs on CPU in the verify tier and is
bit-exact.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

#: Above this share of repeated n-grams, text is degenerate in the ordinary sense of the
#: word — not "somewhat repetitive". Taken from the degeneration literature's usual
#: operating point rather than tuned here, because a threshold tuned on one's own results
#: is not a threshold.
DEGENERATE_REPETITION = 0.5


#: Longest and shortest mean generation length that may be compared on distinct-n, as a
#: ratio. Beyond this the length difference dominates the measure — see
#: ``require_comparable_lengths``.
MAX_LENGTH_RATIO_FOR_DISTINCT_N = 1.5


class LengthConfoundedError(RuntimeError):
    """Raised when distinct-n is compared across generations of very different lengths."""


def distinct_n(tokens: Sequence[int], n: int) -> float:
    """Share of n-grams that are unique. 1.0 means nothing repeats.

    Normalised by the number of n-grams rather than by tokens, which removes the trivial
    part of the length dependence. It does **not** remove all of it, and the difference
    matters enough to state plainly: the number of distinct n-grams a text can contain is
    bounded by its vocabulary, so distinct-n falls as a sample grows even when its
    repetition structure is unchanged. Repeating ``1 2 3`` scores 0.60 at six tokens and
    0.03 at ninety-six — the same text, a nineteen-fold difference.

    So distinct-n is only interpretable between samples of comparable length. Comparing it
    across architectures that generate different amounts would attribute a length
    difference to a quality difference, which is the error this project exists to correct.
    ``require_comparable_lengths`` refuses that comparison rather than reporting it.
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(tokens: Sequence[int], n: int = 4) -> float:
    """Share of n-grams that occur more than once. The complement of distinct-n."""
    if len(tokens) < n:
        return 0.0
    return 1.0 - distinct_n(tokens, n)


def longest_repeated_run(tokens: Sequence[int]) -> int:
    """Longest run of one token repeated consecutively.

    The blunt failure mode, and the one an n-gram measure understates: a model emitting the
    same token two hundred times has few distinct 4-grams but the ratio alone does not
    convey that the output is a single character repeated.
    """
    if not tokens:
        return 0
    longest = run = 1
    for previous, current in zip(tokens, tokens[1:], strict=False):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    return longest


def self_repetition(tokens: Sequence[int], window: int = 32) -> float:
    """Share of positions whose following n-gram already appeared in the preceding window.

    Catches cycling — a phrase repeating every few tokens — which distinct-n over a whole
    sample dilutes when the sample is long.
    """
    if len(tokens) < window + 4:
        return 0.0
    repeated = 0
    positions = 0
    for i in range(window, len(tokens) - 3):
        gram = tuple(tokens[i : i + 4])
        preceding = {
            tuple(tokens[j : j + 4]) for j in range(max(0, i - window), i)
        }
        positions += 1
        if gram in preceding:
            repeated += 1
    return repeated / positions if positions else 0.0


@dataclass(frozen=True)
class DegenerationReport:
    samples: int
    distinct_1: float
    distinct_2: float
    distinct_4: float
    repetition_rate_4: float
    self_repetition: float
    max_repeated_run: int
    degenerate_share: float

    @property
    def is_degenerate(self) -> bool:
        return self.repetition_rate_4 >= DEGENERATE_REPETITION

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "distinct_1": round(self.distinct_1, 6),
            "distinct_2": round(self.distinct_2, 6),
            "distinct_4": round(self.distinct_4, 6),
            "repetition_rate_4": round(self.repetition_rate_4, 6),
            "self_repetition": round(self.self_repetition, 6),
            "max_repeated_run": self.max_repeated_run,
            "degenerate_share": round(self.degenerate_share, 6),
            "is_degenerate": self.is_degenerate,
            "threshold": DEGENERATE_REPETITION,
            "why_reported": (
                "An n-gram overlap metric rewards repetition, because repeated common "
                "phrases do appear in the reference. Without this beside it, an overlap "
                "score can rise while the text degenerates."
            ),
        }


def evaluate(samples: Sequence[Sequence[int]]) -> DegenerationReport:
    """Degeneration measures over a set of tokenised generations."""
    if not samples:
        raise ValueError("no samples to evaluate")

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    per_sample_repetition = [repetition_rate(s, 4) for s in samples]

    return DegenerationReport(
        samples=len(samples),
        distinct_1=mean([distinct_n(s, 1) for s in samples]),
        distinct_2=mean([distinct_n(s, 2) for s in samples]),
        distinct_4=mean([distinct_n(s, 4) for s in samples]),
        repetition_rate_4=mean(per_sample_repetition),
        self_repetition=mean([self_repetition(s) for s in samples]),
        max_repeated_run=max((longest_repeated_run(s) for s in samples), default=0),
        # Share of samples that are individually degenerate, not just the mean rate. A
        # mean of 0.3 could be every sample mildly repetitive or a third of them collapsed,
        # and those are different failures.
        degenerate_share=mean(
            [float(r >= DEGENERATE_REPETITION) for r in per_sample_repetition]
        ),
    )


def require_comparable_lengths(
    baseline: Sequence[Sequence[int]],
    candidate: Sequence[Sequence[int]],
    label: str = "distinct-n",
) -> None:
    """Refuse a distinct-n comparison the lengths make meaningless.

    An exception rather than a warning, for the same reason the metric guards elsewhere in
    this package raise: the number would still be printed, and a printed number becomes a
    claim. A model that generates twice as much text scores worse on distinct-n for that
    reason alone, and nothing in the number says so.
    """
    if not baseline or not candidate:
        raise ValueError("both sets need at least one sample")

    mean_baseline = sum(len(s) for s in baseline) / len(baseline)
    mean_candidate = sum(len(s) for s in candidate) / len(candidate)
    if min(mean_baseline, mean_candidate) == 0:
        raise LengthConfoundedError(
            f"{label} cannot be compared: one set generated nothing."
        )

    ratio = max(mean_baseline, mean_candidate) / min(mean_baseline, mean_candidate)
    if ratio > MAX_LENGTH_RATIO_FOR_DISTINCT_N:
        raise LengthConfoundedError(
            f"{label} is not comparable here: mean generation lengths are "
            f"{mean_baseline:.0f} and {mean_candidate:.0f} tokens, a ratio of {ratio:.2f} "
            f"against a limit of {MAX_LENGTH_RATIO_FOR_DISTINCT_N}.\n"
            "distinct-n falls with length even when repetition structure is unchanged, so "
            "the difference here would be mostly length. Truncate both sets to a common "
            "length and compare that, or report the repetition rate at a fixed window "
            "instead."
        )


def overlap_reward_for_repetition(
    reference: Sequence[int], repeated_token: int, length: int, n: int = 4
) -> float:
    """How much n-gram precision a purely repetitive generation earns against a reference.

    This exists to be reported, not to score models. It puts a number on the claim above:
    that an overlap metric pays for degeneration. If the reference contains the repeated
    n-gram at all, precision is 1.0 for text of no value.
    """
    if length < n:
        return 0.0
    generated = [tuple([repeated_token] * n)] * (length - n + 1)
    reference_grams = Counter(
        tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)
    )
    matched = sum(1 for gram in generated if reference_grams[gram] > 0)
    return matched / len(generated)
