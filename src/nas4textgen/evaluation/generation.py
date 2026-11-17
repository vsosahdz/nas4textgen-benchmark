"""Generation and scoring, with the prior protocol's defect made unrepresentable.

``model.generate()`` returns the prompt followed by the continuation. The prior work
passed that whole sequence to the scorer while the reference held only the continuation,
so every score was computed over text the model did not produce. The correction is not a
flag to remember: :func:`generate_continuation` returns only the new tokens, and there is
no path through this module that yields a hypothesis containing the prompt.

Two reporting rules travel with it, because the defect was detectable and went undetected
for want of them.

**Every reference-based metric is reported beside its trivial floor.** Re-derived on the
real WikiText-2 test split under the protocol the prior work documents, an *empty* generation
scores METEOR 0.3088 overall and 0.6712 on the 24.5% of rows that are section headers, and
the spread across four model-free hypotheses is 0.0821 against a reported improvement of
0.024. A metric whose floor sits near its reported value is not measuring the model. The
figures come from `nas4textgen.audit_prior`, which derives them rather than asserting them.

**Generation length is always reported.** The gap between METEOR 0.70 and BLEU 0.21 was
the symptom, and length is the cause: METEOR weights recall nine to one, so a hypothesis
several times longer than its reference scores well on it and badly on anything
precision-based. Length is what makes that legible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecodingConfig:
    """How text was produced. Reported with every generation metric.

    The prior work never stated its decoding configuration, which makes its generation
    numbers unreproducible even with its weights in hand.
    """

    strategy: str = "greedy"
    max_new_tokens: int = 64
    temperature: float | None = None
    top_k: int | None = None
    top_p: int | None = None
    seed: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
        }


def generate_continuation(
    model,
    prompt_ids: Sequence[int],
    config: DecodingConfig,
    pad_token_id: int,
) -> list[int]:
    """Generate and return **only** the new tokens.

    The slice is the correction. ``model.generate`` returns prompt plus continuation, and
    everything after the prompt's length is what the model actually produced. Returning
    the full sequence is what made every prior score wrong, so this function cannot do it.
    """
    import torch

    # On the model's device, not the default one. A prompt built on the CPU and passed to
    # a model on the card raises at the first embedding lookup, which would have surfaced
    # only on the cloud nodes: every local test of this function runs on the CPU, where the
    # two devices coincide and the omission is invisible.
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long,
                       device=next(model.parameters()).device)
    kwargs: dict[str, object] = {
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": pad_token_id,
        "do_sample": config.strategy != "greedy",
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p

    with torch.no_grad():
        out = model.generate(ids, **kwargs)

    prompt_length = ids.shape[1]
    return out[0][prompt_length:].tolist()


# ── trivial floors ─────────────────────────────────────────────────────────────
# Each is a "model" that produces text without reading the prompt's meaning. Any
# reference-based metric is reported against all of them, and a metric a floor can reach
# is declared uninformative on that corpus rather than quietly kept.

def floor_empty(prompt_ids: Sequence[int], reference_ids: Sequence[int], **_) -> list[int]:
    """Generate nothing. The purest floor: the score with no model at all."""
    return []


def floor_copy_prompt_tail(
    prompt_ids: Sequence[int], reference_ids: Sequence[int], **_
) -> list[int]:
    """Echo the end of the prompt, matched to the reference length.

    Catches a metric that rewards fluent, on-topic text regardless of whether it
    continues anything.
    """
    n = len(reference_ids)
    return list(prompt_ids[-n:]) if n else []


def floor_frequent_tokens(
    prompt_ids: Sequence[int],
    reference_ids: Sequence[int],
    frequent_ids: Sequence[int] = (),
    **_,
) -> list[int]:
    """Repeat the corpus's most frequent tokens to the reference length.

    This is what a degenerate model emits, and under a recall-weighted metric it scores
    well because function words appear in almost every reference.
    """
    n = len(reference_ids)
    if not n or not frequent_ids:
        return []
    return [frequent_ids[i % len(frequent_ids)] for i in range(n)]


def floor_overlong_frequent(
    prompt_ids: Sequence[int],
    reference_ids: Sequence[int],
    frequent_ids: Sequence[int] = (),
    length_multiple: int = 4,
    **_,
) -> list[int]:
    """Frequent tokens at several times the reference length.

    Isolates the length effect from the content effect: if this outscores
    :func:`floor_frequent_tokens`, the metric is paying for verbosity.
    """
    n = len(reference_ids) * length_multiple
    if not n or not frequent_ids:
        return []
    return [frequent_ids[i % len(frequent_ids)] for i in range(n)]


TRIVIAL_FLOORS: dict[str, Callable[..., list[int]]] = {
    "empty": floor_empty,
    "copy_prompt_tail": floor_copy_prompt_tail,
    "frequent_tokens": floor_frequent_tokens,
    "overlong_frequent": floor_overlong_frequent,
}


@dataclass
class ScoredMetric:
    """A metric value with everything needed to judge whether it means anything."""

    name: str
    value: float
    floors: dict[str, float] = field(default_factory=dict)
    length_ratio: float | None = None
    decoding: dict[str, object] | None = None
    corpus: str | None = None
    rows_scored: int = 0
    rows_excluded: int = 0

    @property
    def best_floor(self) -> float:
        return max(self.floors.values()) if self.floors else float("-inf")

    def clears_its_floor(self, rope: float) -> bool:
        """Whether the value beats every trivial floor by more than the ROPE.

        Necessary but **not sufficient**, and the distinction is the whole lesson of the
        audit. Under the prior protocol the reported value was 0.72 against an overall
        empty-generation floor of 0.31, so this check passes — which is precisely why the
        defect survived review. An absolute margin over a floor says the metric is not
        pure noise; it says nothing about whether a *difference between two models* is
        real. For that, see :func:`is_improvement_credible`.
        """
        return self.value - self.best_floor > rope

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.name,
            "value": round(self.value, 6),
            "trivial_floors": {k: round(v, 6) for k, v in self.floors.items()},
            "best_floor": round(self.best_floor, 6) if self.floors else None,
            "margin_over_best_floor": (
                round(self.value - self.best_floor, 6) if self.floors else None
            ),
            "generated_to_reference_length_ratio": (
                round(self.length_ratio, 4) if self.length_ratio is not None else None
            ),
            "decoding": self.decoding,
            "corpus": self.corpus,
            "rows_scored": self.rows_scored,
            "rows_excluded": self.rows_excluded,
        }


#: A metric must put at least this multiple of the measured seed noise between competent
#: text and deliberate garbage before any comparison on it can be interpreted. Ten is an
#: order of magnitude: to resolve differences at noise scale, an instrument's range
#: between sense and nonsense should sit well above that noise. Tied to a measured
#: quantity rather than chosen by taste.
MIN_DISCRIMINATION_IN_ROPES = 10.0


@dataclass(frozen=True)
class MetricDiscrimination:
    """How far a metric separates competent text from garbage, measured per corpus.

    Obtained by scoring a reference-quality continuation and a deliberately degenerate one
    on the same rows. This is the metric's dynamic range on the task — not its resolution,
    which comes from seed noise — and a metric whose range is only a few noise units wide
    cannot support any comparison at all.

    Measured for METEOR under the prior protocol: competent English prose scored 0.2388,
    function-word spam scored 0.2047. A gap of 0.0341 on a metric bounded at 1.0 means
    3.4% of the scale separates English from gibberish. The paper's entire claimed
    contribution was 0.0239 — smaller than that gap, which is the comparison that makes
    the result unreadable.
    """

    metric: str
    corpus: str
    competent: float
    garbage: float

    @property
    def gap(self) -> float:
        return self.competent - self.garbage

    def is_reportable(self, rope: float) -> bool:
        """Whether this metric discriminates enough for any comparison to mean something.

        Note what this does *not* say: it does not require a claimed improvement to exceed
        the full sense-to-nonsense range. A metric that cleanly separates good text at
        0.90 from bad at 0.10 can perfectly well resolve a move from 0.30 to 0.95. The
        question is whether the instrument has range above its noise, not whether an
        effect fills that range.
        """
        return self.gap >= MIN_DISCRIMINATION_IN_ROPES * rope

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "corpus": self.corpus,
            "competent_text": round(self.competent, 6),
            "deliberate_garbage": round(self.garbage, 6),
            "discrimination": round(self.gap, 6),
        }


def is_difference_resolvable(baseline: float, candidate: float, rope: float) -> bool:
    """Whether a difference between two models exceeds the measured seed noise.

    The ROPE comes from training one architecture with several seeds and measuring the
    spread, so this compares an effect against instrumentation rather than against
    judgement.
    """
    return abs(candidate - baseline) > rope


class MetricNotReportableError(RuntimeError):
    """Raised when a metric is too blind on a corpus for comparisons to be interpreted."""


class UnresolvableComparisonError(RuntimeError):
    """Raised when a reported difference is within measured noise."""


def require_resolvable(
    metric_name: str,
    baseline: float,
    candidate: float,
    discrimination: MetricDiscrimination,
    rope: float,
) -> None:
    """Refuse a comparison whose metric is blind, or whose difference is within noise.

    Two independent failures, and the prior work committed the first. Its metric put
    0.0341 between English and gibberish while it claimed 0.0239 as a finding — so the
    instrument was nearly blind on the task, and no amount of care with the difference
    itself would have rescued a comparison made on it.
    """
    if not discrimination.is_reportable(rope):
        raise MetricNotReportableError(
            f"{metric_name!r} is not reportable on {discrimination.corpus!r}: it puts only "
            f"{discrimination.gap:.4f} between competent text ({discrimination.competent:.4f}) "
            f"and deliberate garbage ({discrimination.garbage:.4f}), which is under "
            f"{MIN_DISCRIMINATION_IN_ROPES:g}x the measured noise of {rope:.4f}. The metric "
            "is too blind on this task for any comparison to be interpreted."
        )
    if not is_difference_resolvable(baseline, candidate, rope):
        raise UnresolvableComparisonError(
            f"{metric_name!r}: the difference {abs(candidate - baseline):.4f} is within the "
            f"measured noise of {rope:.4f} and cannot be reported as an effect."
        )


class MissingFloorError(RuntimeError):
    """Raised when a reference-based metric is reported without its trivial floors."""


def require_floors(metric: ScoredMetric) -> None:
    """Refuse to report a reference-based metric that carries no floor.

    Enforced rather than documented. The prior work's headline survived review because
    nothing in the pipeline required the comparison that would have killed it.
    """
    if not metric.floors:
        raise MissingFloorError(
            f"{metric.name!r} was reported without trivial floors. Every "
            "reference-based metric must carry them: an empty generation scores METEOR "
            "0.3088 under the prior protocol, and 0.6712 on header rows."
        )
    if metric.length_ratio is None:
        raise MissingFloorError(
            f"{metric.name!r} was reported without a generated-to-reference length "
            "ratio. Length is what distinguishes a recall-weighted metric being earned "
            "from it being inflated."
        )
