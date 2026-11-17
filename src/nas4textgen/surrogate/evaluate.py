"""Score the slate on the criterion a surrogate is actually used for.

A surrogate in this campaign has one job: given a budget of true evaluations, tell the
search which architectures to spend them on. That is a ranking problem restricted to the
good end, and it is not the same problem as minimising squared error over the whole space —
a model can have excellent RMSE and still be useless if it misorders the top decile.

So three measures, and they answer different questions.

**Kendall tau, globally.** Does the model order the space at all. The headline number, and
the one the prior work reported.

**Kendall tau on the top decile.** Does it order the part anyone will look at. A model can
score well globally by getting the enormous gap between tiny and large architectures right
while shuffling everything near the front.

**Precision at k.** Of the k architectures the model recommends, how many are truly in the
top k. This is the number a practitioner spends money against, and it is the only one of
the three that is not a correlation.

Every candidate is scored against both trivial predictors and disqualified if it beats
neither. That rule is not decoration: in this space sorting by parameter count reaches
Kendall tau near 0.89, so a slate reported without it would present a marginal gain as a
strong result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Fraction of the space treated as the good end for the restricted measures.
TOP_DECILE = 0.10

#: Budgets precision@k is reported at. Small because they are what a search actually spends.
PRECISION_AT = (5, 10, 25, 50)


def kendall_tau(predicted: np.ndarray, actual: np.ndarray) -> float:
    from scipy.stats import kendalltau

    if len(predicted) < 2:
        return float("nan")
    value = kendalltau(predicted, actual).statistic
    return float(value) if value == value else float("nan")


def precision_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """Share of the model's top k that is genuinely in the top k.

    Lower is better in both arrays, so the top k is the k smallest. This is the measure a
    budget is spent against: not how well the model correlates, but how many of the runs it
    recommends turn out to have been worth doing.
    """
    if k > len(actual):
        return float("nan")
    recommended = set(np.argsort(predicted)[:k].tolist())
    truly_best = set(np.argsort(actual)[:k].tolist())
    return len(recommended & truly_best) / k


def top_decile_tau(predicted: np.ndarray, actual: np.ndarray,
                   fraction: float = TOP_DECILE) -> float:
    """Kendall tau restricted to the architectures that are genuinely good.

    A model can score well globally by getting the gap between tiny and large architectures
    right — which parameter count already gets right — while shuffling everything near the
    front, where the ordering is the only thing that matters.

    **Not comparable across tables of different size.** The decile is the best tenth of
    whatever rows exist, so it names different architectures as coverage grows: at 176
    measured rows it was the 17 between perplexity 57.4 and 68.1, at 321 rows the 32 between
    54.2 and 65.7. A rise between those two numbers therefore mixes "the model learned more"
    with "the question changed". Global tau carries no such confound, because it is
    evaluated over everything measured.

    Comparisons across coverage levels belong in the budget curve, which holds one evaluation
    set fixed and varies only the training size.
    """
    size = max(2, int(len(actual) * fraction))
    best = np.argsort(actual)[:size]
    return kendall_tau(predicted[best], actual[best])


@dataclass
class Score:
    name: str
    tier: str
    tau: float
    tau_sd: float
    tau_top_decile: float
    precision: dict[int, float] = field(default_factory=dict)
    unavailable: str = ""
    beats_parameter_floor: bool = False
    beats_linear_floor: bool = False

    #: Names of the two floors, so a score knows whether it is one.
    TRIVIAL = ("sort_by_parameters", "linear_on_genes")

    @property
    def is_trivial(self) -> bool:
        return self.name in self.TRIVIAL

    @property
    def disqualified(self) -> bool:
        """A predictor that outranks neither trivial baseline is not a predictor."""
        if self.unavailable:
            return True
        return not (self.beats_parameter_floor or self.beats_linear_floor)

    def as_dict(self) -> dict[str, object]:
        if self.unavailable:
            return {"name": self.name, "tier": self.tier, "unavailable": self.unavailable}
        return {
            "name": self.name,
            "tier": self.tier,
            "kendall_tau": round(self.tau, 4),
            "kendall_tau_sd": round(self.tau_sd, 4),
            "kendall_tau_top_decile": round(self.tau_top_decile, 4),
            "precision_at": {str(k): round(v, 4) for k, v in self.precision.items()},
            "beats_parameter_floor": self.beats_parameter_floor,
            "beats_linear_floor": self.beats_linear_floor,
            "disqualified": self.disqualified,
        }


def _fit_predict(candidate, features: np.ndarray, target: np.ndarray,
                 train: np.ndarray, test: np.ndarray, params: np.ndarray) -> np.ndarray:
    if candidate.closed_form is not None:
        return candidate.closed_form(features[test], params[test])

    model = candidate.build()
    if candidate.ranking_loss:
        # A pairwise ranker needs its examples grouped. One group per fold: the whole
        # training set is a single ranking problem, which is what it is.
        model.fit(features[train], np.argsort(np.argsort(-target[train])),
                  group=[len(train)])
        return -np.asarray(model.predict(features[test]), dtype=float)

    model.fit(features[train], target[train])
    return np.asarray(model.predict(features[test]), dtype=float)


def score_slate(
    candidates: list,
    features: np.ndarray,
    target: np.ndarray,
    params: np.ndarray,
    folds: int = 5,
    seed: int = 0,
) -> list[Score]:
    """Cross-validate every candidate and apply the disqualification rule.

    Folds are shared across candidates, so the comparison is on identical splits rather
    than on each model's luckiest partition.
    """
    from sklearn.model_selection import KFold

    splits = list(KFold(folds, shuffle=True, random_state=seed).split(features))

    raw: dict[str, Score] = {}
    for candidate in candidates:
        if candidate.build is None and candidate.closed_form is None:
            raw[candidate.name] = Score(
                candidate.name, candidate.tier, float("nan"), float("nan"), float("nan"),
                unavailable=candidate.why,
            )
            continue

        taus, top_taus = [], []
        precision: dict[int, list[float]] = {k: [] for k in PRECISION_AT}
        try:
            for train, test in splits:
                predicted = _fit_predict(candidate, features, target, train, test, params)
                taus.append(kendall_tau(predicted, target[test]))
                top_taus.append(top_decile_tau(predicted, target[test]))
                for k in PRECISION_AT:
                    if k <= len(test):
                        precision[k].append(precision_at_k(predicted, target[test], k))
        except Exception as exc:  # noqa: BLE001
            # A candidate that cannot fit is reported as such rather than dropped, because
            # a slate that shrinks silently is a slate whose result depends on the machine.
            raw[candidate.name] = Score(
                candidate.name, candidate.tier, float("nan"), float("nan"), float("nan"),
                unavailable=f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}",
            )
            continue

        def mean(values: list[float]) -> float:
            """Mean of what is defined, and nan when nothing is.

            A constant predictor makes Kendall tau undefined on every fold, and averaging
            an all-undefined list warns rather than answering. The nan is the right answer —
            the model has no ordering — and it flows through to disqualification.
            """
            finite = [v for v in values if v == v]
            return float(np.mean(finite)) if finite else float("nan")

        raw[candidate.name] = Score(
            name=candidate.name,
            tier=candidate.tier,
            tau=mean(taus),
            tau_sd=(
                float(np.std([v for v in taus if v == v]))
                if any(v == v for v in taus) else float("nan")
            ),
            tau_top_decile=mean(top_taus),
            precision={k: float(np.nanmean(v)) for k, v in precision.items() if v},
        )

    parameter_floor = raw.get("sort_by_parameters")
    linear_floor = raw.get("linear_on_genes")
    for score in raw.values():
        if score.unavailable:
            continue
        if parameter_floor and not parameter_floor.unavailable:
            score.beats_parameter_floor = score.tau > parameter_floor.tau
        if linear_floor and not linear_floor.unavailable:
            score.beats_linear_floor = score.tau > linear_floor.tau

    return list(raw.values())


def report(scores: list[Score], rows: int, space_size: int) -> dict[str, object]:
    """The comparison, with the floors named and the disqualifications counted."""
    from .slate import EXCLUDED

    usable = [s for s in scores if not s.unavailable]
    floor = next((s for s in scores if s.name == "sort_by_parameters"), None)
    best = max((s for s in usable if not s.is_trivial), key=lambda s: s.tau, default=None)

    return {
        "training_rows": rows,
        "coverage": round(rows / space_size, 4) if space_size else None,
        "folds": 5,
        "criterion": "Kendall tau on held-out perplexity rank; precision@k beside it",
        "why_three_measures": (
            "Global tau asks whether the model orders the space; tau on the top decile asks "
            "whether it orders the part anyone will look at, since a model can score well "
            "globally by getting the gap between tiny and large architectures right while "
            "shuffling everything near the front; precision@k is what a budget is actually "
            "spent against and is the only one of the three that is not a correlation."
        ),
        "trivial_floors": {
            "sort_by_parameters": round(floor.tau, 4) if floor and not floor.unavailable else None,
            "rule": (
                "A candidate that outranks neither trivial predictor is disqualified. In "
                "this space the parameter floor is high, so a slate reported without it "
                "would present a marginal gain as a strong result."
            ),
        },
        "best_non_trivial": best.name if best else None,
        "gain_over_floor": (
            round(best.tau - floor.tau, 4)
            if best and floor and not floor.unavailable else None
        ),
        "disqualified": [s.name for s in scores if s.disqualified and not s.unavailable],
        "unavailable": {s.name: s.unavailable for s in scores if s.unavailable},
        "scores": [s.as_dict() for s in sorted(
            scores, key=lambda s: (s.unavailable != "", -(s.tau if s.tau == s.tau else -9))
        )],
        "excluded_families": EXCLUDED,
    }
