"""Is the model's confidence worth anything? Perplexity alone cannot say.

Perplexity is a mean over the log-likelihood of the observed token. Two models can reach the
same perplexity while differing in whether their confidence means anything: one is right
when it is confident and unsure when it is wrong, the other is uniformly overconfident and
happens to average out. For a benchmark whose single search objective is perplexity, that
difference is worth measuring rather than assuming away — it is the most obvious way two
architectures with equal scores are not equally good.

Calibration also has a property the overlap metrics lack: its ideal value is known in
advance. Perfect calibration is zero error, not "higher than the baseline", so there is no
threshold to choose and nothing to tune.

Expected calibration error is reported with the bin counts it came from. A single ECE
number hides how much of it rests on almost-empty bins, and at this scale the confident
bins are sparse — which is itself a finding, not a nuisance to average over.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Equal-width confidence bins. Ten is the convention; stated so a reader can see it was a
#: convention rather than something chosen after looking at results.
DEFAULT_BINS = 10

#: A bin holding fewer samples than this contributes an estimate too noisy to interpret.
#: It still counts towards the error — dropping it would flatter the model — but it is
#: reported separately so a reader can see how much of the number is sparse.
SPARSE_BIN = 30


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Confidence minus accuracy. Positive is overconfident."""
        return self.mean_confidence - self.accuracy

    @property
    def is_sparse(self) -> bool:
        return self.count < SPARSE_BIN

    def as_dict(self) -> dict[str, object]:
        return {
            "range": [round(self.lower, 3), round(self.upper, 3)],
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 6),
            "accuracy": round(self.accuracy, 6),
            "gap": round(self.gap, 6),
            "sparse": self.is_sparse,
        }


@dataclass(frozen=True)
class CalibrationReport:
    samples: int
    expected_calibration_error: float
    maximum_calibration_error: float
    overconfidence: float
    bins: list[Bin]

    @property
    def sparse_share(self) -> float:
        """Share of samples sitting in bins too small to estimate accuracy from."""
        if not self.samples:
            return 0.0
        return sum(b.count for b in self.bins if b.is_sparse) / self.samples

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "expected_calibration_error": round(self.expected_calibration_error, 6),
            "maximum_calibration_error": round(self.maximum_calibration_error, 6),
            "overconfidence": round(self.overconfidence, 6),
            "sparse_share": round(self.sparse_share, 6),
            "bins": [b.as_dict() for b in self.bins],
            "ideal": 0.0,
            "why_reported": (
                "Two architectures can reach the same perplexity while differing in "
                "whether their confidence means anything. Unlike the overlap metrics, the "
                "ideal value here is known in advance rather than chosen."
            ),
        }


def evaluate(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = DEFAULT_BINS,
) -> CalibrationReport:
    """Expected calibration error from per-prediction confidence and correctness.

    ``confidences`` is the probability the model assigned to its own top prediction, and
    ``correct`` whether that prediction was the observed token. Deliberately not the
    probability of the observed token: that is what perplexity already measures, and
    calibration is about the relationship between a model's stated confidence and how often
    it is right — a different question that needs the model's own choice.
    """
    if len(confidences) != len(correct):
        raise ValueError(
            f"{len(confidences)} confidences against {len(correct)} outcomes; a "
            "misalignment here would produce a plausible number from mismatched pairs"
        )
    if not confidences:
        raise ValueError("no predictions to evaluate")
    if any(not 0.0 <= c <= 1.0 for c in confidences):
        raise ValueError(
            "confidences must be probabilities. Logits or unnormalised scores would bin "
            "into the top bucket and report near-perfect calibration."
        )

    edges = [i / bins for i in range(bins + 1)]
    built: list[Bin] = []
    total = len(confidences)
    weighted_error = 0.0
    worst = 0.0
    overconfidence = 0.0

    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        # Upper-inclusive on the last bin only, so a confidence of exactly 1.0 — common
        # once a model is certain — lands somewhere rather than being dropped.
        members = [
            (c, hit)
            for c, hit in zip(confidences, correct, strict=False)
            if (lower <= c < upper) or (upper == 1.0 and c == 1.0)
        ]
        if not members:
            built.append(Bin(lower, upper, 0, 0.0, 0.0))
            continue

        count = len(members)
        mean_confidence = sum(c for c, _ in members) / count
        accuracy = sum(hit for _, hit in members) / count
        entry = Bin(lower, upper, count, mean_confidence, accuracy)
        built.append(entry)

        share = count / total
        weighted_error += share * abs(entry.gap)
        worst = max(worst, abs(entry.gap))
        # Signed, and only the overconfident direction. A model that is underconfident is
        # miscalibrated too, but it is not the failure that makes a benchmark number
        # misleading — overconfidence is what makes a low perplexity look trustworthy.
        overconfidence += share * max(0.0, entry.gap)

    return CalibrationReport(
        samples=total,
        expected_calibration_error=weighted_error,
        maximum_calibration_error=worst,
        overconfidence=overconfidence,
        bins=built,
    )
