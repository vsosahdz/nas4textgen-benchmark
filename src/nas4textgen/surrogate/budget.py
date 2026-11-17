"""How many measured architectures a surrogate needs before it is worth trusting.

The question the campaign actually faces. A surrogate is only useful if it can be fitted on
fewer runs than the search would otherwise spend, so the number that matters is not its
score on the full table — it is the point on the curve where its score becomes good enough
to act on, and whether that point is reached before the budget it was meant to save.

Three things are measured against training-set size, because they stop improving at
different sizes and the largest of those is the answer.

**Rank agreement.** Kendall tau, globally and on the top decile. Where the curve flattens,
more labels buy nothing.

**Top-k reliability.** Precision at k across repeated fits with different training samples.
A surrogate whose top-10 recommendation changes every time it is refitted is not reliable at
that budget, however good its average precision looks.

**Distance to the measured front.** The front a surrogate predicts against the front that
was measured, in IGD+. This is the closest thing to the quantity the second paper cares
about, and a model can rank well while predicting a front in the wrong place.

The training sets are the sweep's own nested waves rather than random subsets, so the curve
describes what this campaign would have seen had it stopped early — not what an idealised
sample would have given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Training sizes the curve is evaluated at. The first is deliberately small: a surrogate
#: that needs three hundred labels has not saved anything on a space of 1,099.
DEFAULT_SIZES = (25, 50, 100, 150, 250, 400, 600)

#: Repeats per size, to separate a model's score from its stability. A single fit cannot
#: distinguish a reliable predictor from a lucky sample.
REPEATS = 5

#: Budget at which top-k reliability is reported. Ten because that is the order of a wave a
#: practitioner would actually spend on the surrogate's advice.
RELIABILITY_K = 10


@dataclass
class BudgetPoint:
    """One training size, for one candidate."""

    size: int
    candidate: str
    tau: float
    tau_sd: float
    tau_top_decile: float
    precision_at_k: float
    #: Share of the top-k that is common to every repeat. A model scoring 0.9 precision with
    #: a different ten each time is not usable at this budget.
    topk_stability: float
    igd_plus_to_measured_front: float

    def as_dict(self) -> dict[str, object]:
        return {
            "training_rows": self.size,
            "candidate": self.candidate,
            "kendall_tau": round(self.tau, 4),
            "kendall_tau_sd": round(self.tau_sd, 4),
            "kendall_tau_top_decile": round(self.tau_top_decile, 4),
            f"precision_at_{RELIABILITY_K}": round(self.precision_at_k, 4),
            "topk_stability": round(self.topk_stability, 4),
            "igd_plus_to_measured_front": round(self.igd_plus_to_measured_front, 6),
        }


@dataclass
class BudgetCurve:
    points: list[BudgetPoint] = field(default_factory=list)
    floor_tau: dict[int, float] = field(default_factory=dict)
    total_rows: int = 0

    def for_candidate(self, name: str) -> list[BudgetPoint]:
        return sorted(
            (p for p in self.points if p.candidate == name), key=lambda p: p.size
        )

    def saturation(self, name: str, tolerance: float = 0.01) -> int | None:
        """Smallest size after which tau never improves by more than ``tolerance``.

        The point where more labels stop buying rank agreement. None when the curve is still
        climbing at the largest size measured, which is a finding rather than a gap — it
        means the surrogate has not saturated within the budget the campaign can afford.
        """
        points = self.for_candidate(name)
        best = max((p.tau for p in points), default=float("nan"))
        for point in points:
            if best - point.tau <= tolerance:
                return point.size
        return None

    def rows_to_beat_floor(self, name: str) -> int | None:
        """Smallest size at which the candidate outranks the parameter-count floor.

        The number that decides whether the surrogate is worth fitting at all. If it needs
        four hundred labels to beat sorting by parameter count, then on a space of 1,099 it
        has saved nothing.
        """
        for point in self.for_candidate(name):
            floor = self.floor_tau.get(point.size)
            if floor is not None and point.tau > floor:
                return point.size
        return None

    def as_dict(self) -> dict[str, object]:
        names = sorted({p.candidate for p in self.points})
        return {
            "total_measured_rows": self.total_rows,
            "sizes": sorted({p.size for p in self.points}),
            "repeats_per_size": REPEATS,
            "floor_tau_by_size": {str(k): round(v, 4) for k, v in self.floor_tau.items()},
            "per_candidate": {
                name: {
                    "saturates_at": self.saturation(name),
                    "rows_to_beat_parameter_floor": self.rows_to_beat_floor(name),
                    "points": [p.as_dict() for p in self.for_candidate(name)],
                }
                for name in names
            },
            "why_these_measures": (
                "Rank agreement says whether the model orders the space; top-k stability "
                "says whether its recommendation survives being refitted, which average "
                "precision hides; distance to the measured front is the quantity the search "
                "paper cares about, and a model can rank well while placing the front "
                "wrongly. They saturate at different sizes and the largest is the answer."
            ),
        }


def _predicted_front(
    predicted_quality: np.ndarray, costs: np.ndarray
) -> np.ndarray:
    """The front a surrogate would report: exact costs, predicted quality."""
    from ..space.enumerate import non_dominated

    objectives = np.column_stack([costs, predicted_quality])
    return objectives[non_dominated(objectives)]


def measure(
    candidates: list,
    features: np.ndarray,
    target: np.ndarray,
    costs: np.ndarray,
    params: np.ndarray,
    order: list[int] | None = None,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    repeats: int = REPEATS,
    seed: int = 0,
) -> BudgetCurve:
    """Trace each candidate's score against training-set size.

    ``order`` is the sweep's execution order when given, so a prefix of it is what the
    campaign would actually have had at that budget. Without it, random subsets are used and
    the curve describes an idealised sample instead — which is a different and easier
    question, so the distinction is recorded.
    """
    from ..search.scoring import igd_plus, normalise
    from .evaluate import kendall_tau, precision_at_k, top_decile_tau

    rng = np.random.default_rng(seed)
    available = len(target)

    # Bounds over the full objective vector — both costs and quality — because that is
    # what the front lives in. Taking them from the costs alone left a two-element vector
    # normalising a three-column front, which numpy refused rather than silently doing
    # something wrong.
    full = np.column_stack([costs, target])
    lower, upper = full.min(axis=0), full.max(axis=0)

    measured_front = _predicted_front(target, costs)
    normalised_front = normalise(measured_front, lower, upper)

    curve = BudgetCurve(total_rows=available)

    # One evaluation set for every size and every repeat, held out from all of them.
    #
    # Without this, top-k stability is meaningless: each repeat predicts on its own
    # complement, so intersecting the top-ten index sets compares recommendations over
    # different architectures and reports near-zero however stable the model is. A common
    # held-out set is what makes "the same ten" a question that can be asked.
    holdout_size = max(RELIABILITY_K * 3, available // 4)
    shuffled = rng.permutation(available)
    evaluation = np.sort(shuffled[:holdout_size])
    pool = np.sort(shuffled[holdout_size:])

    for size in sizes:
        if size > len(pool):
            continue

        # The same training sets for every candidate at this size, so the comparison is not
        # between models on different samples.
        samples: list[np.ndarray] = []
        if order is not None:
            # A prefix of the execution order, which is what the campaign would actually
            # have had at this budget. Deterministic, so one sample: repeating it would
            # measure nothing.
            prefix = np.array([i for i in order if i in set(pool.tolist())][:size])
            if len(prefix) == size:
                samples.append(prefix)
        if not samples:
            samples = [rng.choice(pool, size=size, replace=False) for _ in range(repeats)]

        for candidate in candidates:
            if candidate.build is None and candidate.closed_form is None:
                continue

            taus, top_taus, precisions, fronts = [], [], [], []
            topk_sets: list[frozenset[int]] = []
            try:
                for train in samples:
                    test = evaluation
                    if candidate.closed_form is not None:
                        predicted = candidate.closed_form(features[test], params[test])
                    else:
                        model = candidate.build()
                        if candidate.ranking_loss:
                            model.fit(
                                features[train],
                                np.argsort(np.argsort(-target[train])),
                                group=[len(train)],
                            )
                            predicted = -np.asarray(model.predict(features[test]), float)
                        else:
                            model.fit(features[train], target[train])
                            predicted = np.asarray(model.predict(features[test]), float)

                    taus.append(kendall_tau(predicted, target[test]))
                    top_taus.append(top_decile_tau(predicted, target[test]))
                    if len(test) >= RELIABILITY_K:
                        precisions.append(
                            precision_at_k(predicted, target[test], RELIABILITY_K)
                        )
                        topk_sets.append(
                            frozenset(np.argsort(predicted)[:RELIABILITY_K].tolist())
                        )

                    front = _predicted_front(predicted, costs[test])
                    fronts.append(
                        igd_plus(normalise(front, lower, upper), normalised_front)
                    )
            except Exception:  # noqa: BLE001
                continue

            finite = [v for v in taus if v == v]
            if not finite:
                continue

            # Undefined rather than perfect when there is only one fit: a single
            # recommendation is trivially identical to itself and says nothing about
            # whether refitting would change it.
            stability = float("nan")
            if len(topk_sets) > 1:
                common = frozenset.intersection(*topk_sets)
                stability = len(common) / RELIABILITY_K

            curve.points.append(BudgetPoint(
                size=size,
                candidate=candidate.name,
                tau=float(np.mean(finite)),
                tau_sd=float(np.std(finite)),
                tau_top_decile=float(np.mean([v for v in top_taus if v == v] or [np.nan])),
                precision_at_k=float(np.mean(precisions)) if precisions else float("nan"),
                topk_stability=stability,
                igd_plus_to_measured_front=float(np.mean(fronts)) if fronts else float("nan"),
            ))

        floor = [p for p in curve.points
                 if p.size == size and p.candidate == "sort_by_parameters"]
        if floor:
            curve.floor_tau[size] = floor[0].tau

    return curve
