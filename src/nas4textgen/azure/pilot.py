"""The cheap experiments that turn the sweep's cost from a range into a number.

The canary asked whether the platform works. The pilot asks what the sweep will cost and
how many seeds it needs, and it exists because two numbers the campaign depends on are
currently guesses.

**Throughput across the space.** Four measured architectures do not describe 1,099: the
affine fit misses the smallest by 40%, so ``ThroughputModel`` declines to project and
timeouts fall back to the slowest rate ever seen. That is safe and wasteful. Twenty
architectures spread across the compute range make the fit checkable, and the residual
becomes evidence rather than an unknown.

**Seed variance.** Every Bayesian statement in the statistical protocol needs a ROPE, and
the spec requires it derived from measured noise rather than chosen. Nothing has measured
that noise yet, so the whole comparison layer is currently blocked on a number nobody has.
One architecture at several seeds produces it.

The two share a wave because they share a fixed cost — the queue, the image pull, the
corpus mount — and because a pilot that had to be run twice is a pilot people skip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..space.cost import forward_flops_per_token, parameter_count
from ..space.enumerate import enumerate_with_ceiling
from ..space.genotype import GENES, Genotype

#: Architectures for the throughput calibration. Enough that a two-parameter fit has
#: eighteen degrees of freedom rather than two, and few enough that the whole pilot costs
#: about one per cent of the sweep.
CALIBRATION_SIZE = 20

#: Seeds for the variance measurement. Five is the minimum ``rope.from_replicates``
#: accepts, and the reason is there: below it the standard deviation is itself so uncertain
#: that a ROPE built on it is arbitrary in a different way than simply picking one.
VARIANCE_SEEDS = 5


@dataclass(frozen=True)
class PilotMember:
    genotype: Genotype
    purpose: str  # calibration | variance
    seed: int
    why: str

    def as_dict(self) -> dict[str, object]:
        return {
            "genotype": {gene: getattr(self.genotype, gene) for gene in GENES},
            "e": self.genotype.e,
            "params": parameter_count(self.genotype),
            "flops_per_token": forward_flops_per_token(self.genotype),
            "purpose": self.purpose,
            "seed": self.seed,
            "why": self.why,
        }


def calibration_set(size: int = CALIBRATION_SIZE) -> list[Genotype]:
    """Architectures spread evenly over log compute, not over the space.

    Spreading over the space would sample its density, and this space is dense where
    architectures are cheap — the fit would then be well-determined exactly where it is
    already known and unconstrained at the expensive end, which is the end that sets the
    timeouts and the cost.

    Log spacing because the compute range spans three orders of magnitude; linear spacing
    would put almost every point in the top decade.
    """
    space = sorted(enumerate_with_ceiling().genotypes, key=forward_flops_per_token)
    flops = np.array([forward_flops_per_token(g) for g in space], dtype=float)

    targets = np.logspace(np.log10(flops[0]), np.log10(flops[-1]), size)
    chosen: list[Genotype] = []
    for target in targets:
        index = int(np.argmin(np.abs(flops - target)))
        # Nearest unused, so a cluster of identical-cost architectures cannot collapse
        # several targets onto one point and quietly shrink the sample.
        while space[index] in chosen and index + 1 < len(space):
            index += 1
        chosen.append(space[index])
    return chosen


def variance_architecture() -> Genotype:
    """The architecture the seed study repeats.

    Near the median of compute rather than at an extreme. A ROPE measured on the smallest
    architecture would describe the noise of a model that barely trains, and one measured
    on the largest would cost five times as much to obtain; the median is where most of the
    space's runs actually sit, so its noise is the one most comparisons will be made
    against.
    """
    space = sorted(enumerate_with_ceiling().genotypes, key=forward_flops_per_token)
    return space[len(space) // 2]


def plan(size: int = CALIBRATION_SIZE, seeds: int = VARIANCE_SEEDS) -> list[PilotMember]:
    """Every run the pilot needs, with what each one is for."""
    members = [
        PilotMember(
            genotype=genotype,
            purpose="calibration",
            seed=0,
            why="one point on the throughput curve, spread over log compute",
        )
        for genotype in calibration_set(size)
    ]

    repeated = variance_architecture()
    members.extend(
        PilotMember(
            genotype=repeated,
            purpose="variance",
            seed=seed,
            why=(
                "identical except for the seed, so the spread is the noise floor below "
                "which no architectural claim can be made"
            ),
        )
        # Seed 0 of this architecture may already exist from the calibration set; the
        # submitter deduplicates by fingerprint, so overlapping is free rather than wasteful.
        for seed in range(seeds)
    )
    return members


def summary(members: list[PilotMember]) -> dict[str, object]:
    """What the pilot is, and what each half unblocks."""
    calibration = [m for m in members if m.purpose == "calibration"]
    variance = [m for m in members if m.purpose == "variance"]
    flops = [forward_flops_per_token(m.genotype) for m in calibration]

    return {
        "runs": len(members),
        "calibration": {
            "architectures": len(calibration),
            "flops_per_token_range": [min(flops), max(flops)],
            "decades_covered": round(np.log10(max(flops) / min(flops)), 2),
            "unblocks": (
                "a throughput fit with real degrees of freedom. Four points miss the "
                "smallest architecture by 40%, so the model declines to project and every "
                "timeout falls back to the slowest rate measured — safe and wasteful."
            ),
        },
        "variance": {
            "architecture": {
                gene: getattr(variance[0].genotype, gene) for gene in GENES
            } if variance else None,
            "seeds": len(variance),
            "unblocks": (
                "the ROPE. Every Bayesian statement in the statistical protocol needs one, "
                "the spec requires it derived from measured noise rather than chosen, and "
                "nothing has measured that noise yet."
            ),
        },
        "members": [m.as_dict() for m in members],
    }
