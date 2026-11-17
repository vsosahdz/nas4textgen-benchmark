"""Distributional distance between generated text and human text.

MAUVE embeds both sets of texts with a feature model, clusters the joint space, and
integrates a divergence frontier between the two histograms. It answers a question no
per-sample metric can: whether the model's output *distribution* looks like the reference
one, rather than whether any particular continuation resembles any particular reference.
That is the right question for open-ended generation and it is why the metric is here.

It comes with three properties that have to travel with the number, or the number is not a
measurement.

**Its trivial floor is at the top, not the bottom.** Every other metric in this battery has
a floor near zero that a degenerate strategy reaches. MAUVE's degenerate strategy is to
emit the reference text itself, which scores near one. So the floor that has to be measured
is the *ceiling*: what MAUVE gives a model that copies. Reported alongside the score,
because a MAUVE of 0.9 means one thing next to a copy-baseline of 0.95 and another next to
0.999.

**It is not a fixed function of the two text sets.** The value depends on the feature
model, the number of clusters, the sample count, how much variance the PCA keeps, the
k-means seed, and the exponential factor the area under the divergence curve is scaled by.
Two papers reporting "MAUVE 0.87" on the same data can both be right and mean different
things. Everything that moves it is recorded in the report.

**It needs enough samples to cluster.** With a few hundred texts the clustering is noise
and the divergence frontier is a property of the seed. The floor below is refused rather
than reported small.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

#: Feature model. GPT-2 large is what the MAUVE paper used and what published numbers are
#: comparable against; a smaller featurizer would make this project's figures incomparable
#: with everything else, for a saving that does not matter since it runs once per
#: architecture rather than once per token.
FEATURE_MODEL = "gpt2-large"

#: Clusters as a fraction of the sample count, the paper's own rule. Fixed here rather than
#: tuned, because tuning it against the result is choosing the result.
CLUSTER_FRACTION = 0.1

#: Explained variance the PCA keeps before clustering. This, not a dimension count, is
#: what sets the space k-means partitions, and it is the library's own default — pinned
#: here rather than left implicit so a version change that moves the default shows up as a
#: diff instead of as a shifted score.
KMEANS_EXPLAINED_VARIANCE = 0.9

#: k-means restarts and iteration cap. Recorded because a clustering that has not converged
#: is a different clustering, and both are the library's defaults rather than choices.
KMEANS_RESTARTS = 5
KMEANS_MAX_ITERATIONS = 500

#: Seed for the clustering. Fixed, and reported, because the partition depends on it.
KMEANS_SEED = 0

#: Points on the divergence curve, and the scaling applied to the area under it. The second
#: is the one to watch: MAUVE's headline number is an exponential rescaling of the raw
#: divergence, so two runs at different scaling factors are not on the same axis.
DIVERGENCE_CURVE_POINTS = 25
SCALING_FACTOR = 5

#: Below this many texts per side the clustering describes the seed rather than the model.
#: The MAUVE authors report instability under a few thousand; this is the point below which
#: the measurement is refused instead of reported with a caveat nobody reads.
MINIMUM_SAMPLES = 1000

#: Maximum tokens per text. Longer texts dominate the feature space; truncating both sides
#: identically is what keeps the comparison about content rather than about length.
MAX_TEXT_LENGTH = 256


class TooFewSamples(RuntimeError):
    """Raised when there are not enough texts for the clustering to mean anything."""


@dataclass(frozen=True)
class MauveResult:
    """One MAUVE measurement, with everything that moves it."""

    score: float
    generated_samples: int
    reference_samples: int
    #: What MAUVE gives text drawn from the reference distribution itself. The practical
    #: ceiling for this sample size and configuration, which is not 1.0 and varies with
    #: both — so a score is read against it rather than against unity.
    copy_ceiling: float | None = None
    #: What MAUVE gives a degenerate generator that emits one repeated string. The floor,
    #: measured rather than assumed to be zero.
    degenerate_floor: float | None = None
    feature_model: str = FEATURE_MODEL
    clusters: int = 0
    kmeans_explained_variance: float = KMEANS_EXPLAINED_VARIANCE
    kmeans_seed: int = KMEANS_SEED
    scaling_factor: int = SCALING_FACTOR
    max_text_length: int = MAX_TEXT_LENGTH

    @property
    def headroom(self) -> float | None:
        """How much of the gap to the achievable ceiling the model has closed.

        ``None`` when the ceiling was not measured. Reported because the raw score is not
        interpretable on its own: the same 0.87 is most of the way to a ceiling of 0.90 and
        barely half of the way to one of 0.999.
        """
        if self.copy_ceiling is None or self.degenerate_floor is None:
            return None
        span = self.copy_ceiling - self.degenerate_floor
        if span <= 0:
            return None
        return (self.score - self.degenerate_floor) / span

    def as_dict(self) -> dict[str, object]:
        return {
            "mauve": round(self.score, 6),
            "copy_ceiling": None if self.copy_ceiling is None else round(self.copy_ceiling, 6),
            "degenerate_floor": (
                None if self.degenerate_floor is None else round(self.degenerate_floor, 6)
            ),
            "headroom": None if self.headroom is None else round(self.headroom, 6),
            "generated_samples": self.generated_samples,
            "reference_samples": self.reference_samples,
            "feature_model": self.feature_model,
            "clusters": self.clusters,
            "kmeans_explained_variance": self.kmeans_explained_variance,
            "kmeans_seed": self.kmeans_seed,
            "kmeans_restarts": KMEANS_RESTARTS,
            "kmeans_max_iterations": KMEANS_MAX_ITERATIONS,
            "divergence_curve_points": DIVERGENCE_CURVE_POINTS,
            "scaling_factor": self.scaling_factor,
            "max_text_length": self.max_text_length,
            "why_the_configuration_is_reported": (
                "MAUVE is not a fixed function of the two text sets. The feature model, "
                "cluster count, sample size, explained-variance cut, k-means seed and the "
                "exponential scaling factor applied to the area under the divergence curve "
                "all move it, so a bare score is not comparable with another bare score."
            ),
        }


def require_enough_samples(generated: Sequence[str], reference: Sequence[str]) -> None:
    """Refuse a measurement whose clustering would describe the seed."""
    smaller = min(len(generated), len(reference))
    if smaller < MINIMUM_SAMPLES:
        raise TooFewSamples(
            f"{smaller} texts on the smaller side, against a minimum of {MINIMUM_SAMPLES}. "
            "Below that the k-means partition is a property of its seed rather than of the "
            "two distributions, and the divergence frontier integrates noise."
        )


def compute(
    generated: Sequence[str],
    reference: Sequence[str],
    device_id: int = -1,
    with_baselines: bool = True,
) -> MauveResult:
    """Score generated text against reference text, with its floor and ceiling.

    ``with_baselines`` measures both degenerate strategies as well, which triples the cost
    and is what makes the number readable. Turning it off is for a smoke run, not for a
    reported result.
    """
    import mauve as mauve_lib

    require_enough_samples(generated, reference)
    clusters = max(2, int(CLUSTER_FRACTION * min(len(generated), len(reference))))

    def score(target: Sequence[str], candidate: Sequence[str]) -> float:
        outcome = mauve_lib.compute_mauve(
            p_text=list(target),
            q_text=list(candidate),
            featurize_model_name=FEATURE_MODEL,
            device_id=device_id,
            max_text_length=MAX_TEXT_LENGTH,
            num_buckets=clusters,
            kmeans_explained_var=KMEANS_EXPLAINED_VARIANCE,
            kmeans_num_redo=KMEANS_RESTARTS,
            kmeans_max_iter=KMEANS_MAX_ITERATIONS,
            divergence_curve_discretization_size=DIVERGENCE_CURVE_POINTS,
            mauve_scaling_factor=SCALING_FACTOR,
            seed=KMEANS_SEED,
            verbose=False,
        )
        return float(outcome.mauve)

    copy_ceiling = None
    degenerate_floor = None
    if with_baselines:
        # The ceiling: the reference split in two disjoint halves and scored against each
        # other. Text drawn from the target distribution itself, which is the best any
        # generator could do and is not 1.0 at finite sample size. The halves must not
        # overlap — scoring a subset against the whole set it came from measures the
        # overlap rather than the distribution, and reports a ceiling that no generator
        # could reach because no generator is allowed to reuse the reference.
        midpoint = len(reference) // 2
        copy_ceiling = (
            score(reference[midpoint:], reference[:midpoint]) if midpoint >= 2 else None
        )
        # The floor: one string repeated. A generator that has collapsed, which is the
        # failure mode open-ended generation actually exhibits at this scale.
        degenerate_floor = score(reference, [reference[0][:MAX_TEXT_LENGTH]] * len(generated))

    return MauveResult(
        score=score(reference, generated),
        generated_samples=len(generated),
        reference_samples=len(reference),
        copy_ceiling=copy_ceiling,
        degenerate_floor=degenerate_floor,
        clusters=clusters,
    )


@dataclass
class MauveUnavailable:
    """Recorded when the metric could not run, so the gap is visible in the artifact.

    A battery that silently omits a metric it failed to compute reports a smaller battery
    than it claims. The reason travels with the omission.
    """

    reason: str
    detail: str = ""
    fields: tuple[str, ...] = field(default_factory=lambda: ("mauve",))

    def as_dict(self) -> dict[str, object]:
        return {"mauve": None, "unavailable": self.reason, "detail": self.detail}
