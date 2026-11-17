"""How many seeds, and where. The answer is not a single number.

Uniform seeding is the obvious design and it is wasteful. Seed noise only changes a
conclusion where two architectures are close enough for it to reorder them, and in this
space most are not: across the pilot's measured architectures the median gap between
cost-adjacent neighbours is nine and a half standard deviations of seed noise. Paying three
times over for a third seed on a pair separated by fifty sigma buys nothing.

So the campaign runs one seed everywhere and replicates only where a second measurement can
change the answer — which is on and near the Pareto front, since that is the only place a
reordering alters what the benchmark reports.

**What the pilot measured.** Seed noise is 0.651 perplexity, one standard deviation, from
five replicates of one architecture. Simulating front membership under that noise over the
pilot's twenty-three architectures: one seed recovers the true front with Jaccard 0.95 and
about 0.8 membership errors per draw; three seeds reach 1.00 and 0.5 errors. Three times the
cost for a third fewer errors.

**What it could not measure.** Those twenty-three were chosen to spread over compute, so
they are unusually well separated. The real space has 1,099 points and its front will be
denser, so front errors will be *more* common than the simulation shows, not less. That is
stated rather than smoothed over, and it is the reason the replication set is chosen after
the sweep from the measured front rather than guessed at now.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Seeds every architecture gets. One, because the alternative is three times the cost for
#: a reduction in front errors the pilot measured at about a third.
BASE_SEEDS = 1

#: Seeds an architecture near the front gets, counting the base one.
FRONT_SEEDS = 5

#: How close to the front is near enough to matter, in units of measured seed noise.
#:
#: An architecture two standard deviations behind the front can be pushed onto it by noise
#: in a single measurement, so it is part of the question even though its point estimate is
#: not. Beyond that the probability is small enough that replicating is spending compute on
#: a conclusion that will not change.
FRONT_MARGIN_SIGMAS = 2.0


@dataclass(frozen=True)
class SeedPlan:
    base_seeds: int
    front_seeds: int
    architectures: int
    near_front: int
    sigma: float

    @property
    def total_runs(self) -> int:
        return self.architectures * self.base_seeds + self.near_front * (
            self.front_seeds - self.base_seeds
        )

    @property
    def uniform_runs(self) -> int:
        """What seeding everything to front depth would have cost."""
        return self.architectures * self.front_seeds

    def as_dict(self) -> dict[str, object]:
        return {
            "base_seeds": self.base_seeds,
            "front_seeds": self.front_seeds,
            "architectures": self.architectures,
            "near_front": self.near_front,
            "runs": self.total_runs,
            "runs_if_uniform": self.uniform_runs,
            "saved_fraction": round(1 - self.total_runs / self.uniform_runs, 4),
            "seed_sigma": self.sigma,
            "margin_sigmas": FRONT_MARGIN_SIGMAS,
            "why": (
                "Seed noise only changes a conclusion where two architectures are close "
                "enough to reorder. The pilot's median gap between cost-adjacent "
                "neighbours is nine and a half sigma, so replicating everywhere pays three "
                "times over for pairs separated by fifty. Replication goes where a second "
                "measurement can move a point on or off the front."
            ),
            "caveat": (
                "The pilot's architectures were chosen to spread over compute and are "
                "unusually well separated. A front over 1,099 points will be denser and "
                "its membership errors more common, which is why this set is selected from "
                "the measured front after the sweep rather than predicted before it."
            ),
        }


def near_front(objectives: np.ndarray, sigma: float,
               margin_sigmas: float = FRONT_MARGIN_SIGMAS) -> np.ndarray:
    """Indices worth replicating: on the front, or within noise of reaching it.

    ``objectives`` is minimised, with the quality objective in the last column — the one
    seed noise acts on. Cost columns are computed exactly and carry no noise, so an
    architecture can only move onto the front by its quality improving.
    """
    from ..space.enumerate import non_dominated

    front = set(non_dominated(objectives).tolist())
    if sigma <= 0 or margin_sigmas <= 0:
        return np.array(sorted(front), dtype=int)

    # One point at a time, against the others unshifted.
    #
    # The first version shifted every quality value at once and re-derived the front, which
    # cannot work: a uniform shift moves the whole cloud and leaves dominance exactly as it
    # was. The question is not "where would the front be if everything improved" but "which
    # points would reach the current front if they alone improved", and that has to be asked
    # per point.
    improved = objectives[:, -1] - margin_sigmas * sigma
    promoted = set()
    for index in range(len(objectives)):
        if index in front:
            continue
        candidate = objectives[index].copy()
        candidate[-1] = improved[index]

        others = np.delete(objectives, index, axis=0)
        dominated = np.all(others <= candidate, axis=1) & np.any(others < candidate, axis=1)
        if not dominated.any():
            promoted.add(index)

    return np.array(sorted(front | promoted), dtype=int)


def plan(objectives: np.ndarray, sigma: float) -> SeedPlan:
    """The campaign's seeding, given a measured front and the measured noise."""
    return SeedPlan(
        base_seeds=BASE_SEEDS,
        front_seeds=FRONT_SEEDS,
        architectures=len(objectives),
        near_front=len(near_front(objectives, sigma)),
        sigma=sigma,
    )
