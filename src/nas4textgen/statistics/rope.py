"""The region of practical equivalence, measured rather than chosen.

Every Bayesian statement in this project needs a width below which a difference does not
matter. That width decides which results get called equivalent, so choosing it after
seeing the results is a way of choosing the conclusion. The spec therefore requires it to
come from a measurement of the relevant noise, and to be reported with that measurement
attached.

The relevant noise is whatever varies when nothing that should matter has changed. Rerun
the same architecture, same corpus, same budget, with different seeds: the spread of the
resulting perplexities is the floor below which no architectural claim can be made, because
that much movement happens with no architecture change at all.

A difference smaller than the noise is not a small effect. It is not an effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: How many measurements before a spread is worth calling a noise estimate. Below this the
#: standard deviation is itself so uncertain that a ROPE built on it is arbitrary in a
#: different way than simply picking one.
MIN_REPLICATES = 5

#: The ROPE as a multiple of the measured standard deviation. Half a standard deviation is
#: the usual convention; stated here so a reader can see it is a convention applied to a
#: measurement, not a number chosen to make a result come out.
ROPE_SIGMA_MULTIPLE = 0.5


class InsufficientReplicates(RuntimeError):
    """Raised when too few repeated measurements exist to estimate noise."""


@dataclass(frozen=True)
class MeasuredRope:
    """A ROPE and the measurement that produced it. Neither travels without the other."""

    width: float
    metric: str
    source: str
    replicates: int
    mean: float
    std: float
    spread: float
    sigma_multiple: float

    @property
    def description(self) -> str:
        return (
            f"{self.width:.6g} on {self.metric}, from {self.sigma_multiple}x the standard "
            f"deviation of {self.replicates} replicates of {self.source} "
            f"(mean {self.mean:.6g}, sd {self.std:.6g}, range {self.spread:.6g})"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rope": self.width,
            "metric": self.metric,
            "source": self.source,
            "replicates": self.replicates,
            "mean": self.mean,
            "std": self.std,
            "observed_range": self.spread,
            "sigma_multiple": self.sigma_multiple,
            "relative_width": self.width / self.mean if self.mean else None,
            "description": self.description,
            "why": (
                "Derived from repeated measurements under identical conditions, so it is "
                "the floor below which no claim about architecture can be made: that much "
                "movement happens with nothing changed."
            ),
        }


def from_replicates(
    values: Sequence[float],
    metric: str,
    source: str,
    sigma_multiple: float = ROPE_SIGMA_MULTIPLE,
) -> MeasuredRope:
    """Derive a ROPE from repeated measurements of one configuration.

    ``values`` must be the same architecture, corpus and budget, differing only in the seed
    — or in whatever nuisance factor the comparison intends to treat as noise. Passing
    measurements that differ in something meaningful would inflate the ROPE and quietly
    declare real differences equivalent, which is the failure mode this guards against by
    demanding the source be named.
    """
    array = np.asarray(values, dtype=float)
    if array.size < MIN_REPLICATES:
        raise InsufficientReplicates(
            f"{array.size} replicate(s) of {source}; at least {MIN_REPLICATES} are needed "
            "before a standard deviation is a noise estimate rather than a guess. Without "
            "them, report the comparison as undecided rather than inventing a width."
        )
    if not np.isfinite(array).all():
        raise ValueError("replicates contain non-finite values")

    # Sample standard deviation, ddof=1: these are a sample of the runs that could have
    # happened, not the population of them.
    std = float(array.std(ddof=1))
    if std == 0:
        raise InsufficientReplicates(
            f"all {array.size} replicates of {source} are identical, so this measures no "
            "noise. Either the pipeline is bit-exact here — in which case say so and use "
            "a ROPE from a stage that is not — or the replicates are not independent."
        )

    return MeasuredRope(
        width=sigma_multiple * std,
        metric=metric,
        source=source,
        replicates=int(array.size),
        mean=float(array.mean()),
        std=std,
        spread=float(array.max() - array.min()),
        sigma_multiple=sigma_multiple,
    )


def from_seed_matrix(
    values: Sequence[Sequence[float]],
    metric: str,
    source: str,
    sigma_multiple: float = ROPE_SIGMA_MULTIPLE,
) -> MeasuredRope:
    """Pool seed-to-seed noise across several configurations.

    Rows are configurations, columns are seeds. Pooling the within-row variance gives a
    noise estimate that does not depend on which single architecture happened to be chosen
    for the replicate study — and a ROPE that rested on one architecture would be a claim
    that its noise is everyone's.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("expected a configurations-by-seeds matrix")
    if matrix.shape[1] < 2:
        raise InsufficientReplicates("at least two seeds per configuration are needed")
    if matrix.shape[0] * matrix.shape[1] < MIN_REPLICATES:
        raise InsufficientReplicates(
            f"{matrix.size} measurements in total; at least {MIN_REPLICATES} are needed"
        )

    # Pooled within-configuration variance, which is the seed noise with any real
    # differences between configurations removed.
    within = matrix - matrix.mean(axis=1, keepdims=True)
    dof = matrix.shape[0] * (matrix.shape[1] - 1)
    pooled_std = float(np.sqrt((within**2).sum() / dof))

    if pooled_std == 0:
        raise InsufficientReplicates(
            f"seed variation in {source} is exactly zero across {matrix.size} runs"
        )

    return MeasuredRope(
        width=sigma_multiple * pooled_std,
        metric=metric,
        source=f"{source} ({matrix.shape[0]} configurations x {matrix.shape[1]} seeds)",
        replicates=int(matrix.size),
        mean=float(matrix.mean()),
        std=pooled_std,
        spread=float(matrix.max() - matrix.min()),
        sigma_multiple=sigma_multiple,
    )
