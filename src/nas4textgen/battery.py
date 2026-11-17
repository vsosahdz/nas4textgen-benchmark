"""Whether the reporting battery says anything the objective does not.

Two reviewers objected that one metric cannot evaluate generation. The answer this project
gives is a battery — grammaticality, calibration, degeneration — but a battery is only an
answer if its members measure something the objective misses. If BLiMP accuracy were a
monotone function of perplexity, reporting both would be reporting one quantity twice and
the objection would stand unanswered behind more columns.

So the battery is scored against the objective rather than beside it. For each metric this
module reports the rank correlation with perplexity over the measured front, and the reading
is the opposite of the usual one: a correlation near one means the metric is redundant, and a
moderate correlation is what makes it worth the node time.

**Rank rather than value.** The metrics have incomparable units — a probability, an expected
calibration error, a share of repeated 4-grams — and only their orderings can be put side by
side. Rank correlation also survives the monotone rescalings that would otherwise make the
answer depend on whether one reports perplexity or its logarithm.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

#: The metric inside each battery entry that stands for it, and whether lower is better.
#: Named explicitly because several entries hold a dozen numbers and picking one silently
#: would make the correlation depend on an undocumented choice.
HEADLINE = {
    "blimp": ("macro_accuracy", False),
    "calibration": ("expected_calibration_error", True),
    "distinct_n": ("distinct_1", False),
    "repetition_rate": ("repetition_rate_4", True),
}

#: Above this, a metric is reported as carrying little the objective does not already.
#: Stated in advance, as the project states every threshold, so that it cannot be chosen
#: after seeing which metrics would survive it.
REDUNDANT_ABOVE = 0.9

#: Below this many architectures the correlation is not reported at all. With three points a
#: Kendall tau is +1, -1 or +/-0.333 almost whatever the data say, and the canary wave
#: produced exactly that: five of six metrics reading a flat +/-1.000 and being declared
#: redundant on three architectures. A threshold that a sample that small can clear is a
#: threshold that measures the sample size.
MINIMUM_ARCHITECTURES = 30


class NoBatteryRuns(RuntimeError):
    """Raised when no manifest carries a battery to analyse."""


@dataclass(frozen=True)
class MetricAgreement:
    """One battery metric against the objective, on one corpus."""

    metric: str
    key: str
    corpus: str
    architectures: int
    kendall_tau: float
    kendall_p: float
    spread: float

    @property
    def redundant(self) -> bool:
        """Whether the objective already orders architectures the way this metric does."""
        return abs(self.kendall_tau) >= REDUNDANT_ABOVE

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "key": self.key,
            "corpus": self.corpus,
            "architectures": self.architectures,
            "kendall_tau_against_perplexity": round(self.kendall_tau, 4),
            "p_value": self.kendall_p,
            "spread": round(self.spread, 6),
            "redundant": self.redundant,
        }


@dataclass(frozen=True)
class BatteryReport:
    """Every metric on every corpus that licensed it."""

    agreements: tuple[MetricAgreement, ...] = field(default_factory=tuple)
    threshold: float = REDUNDANT_ABOVE
    #: Metrics that ran but on too few architectures to correlate, as (corpus, metric).
    underpowered: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        informative = [a for a in self.agreements if not a.redundant]
        return {
            "redundant_above": self.threshold,
            "metrics": [a.as_dict() for a in self.agreements],
            "metrics_adding_information": len(informative),
            "metrics_measured": len(self.agreements),
            "minimum_architectures": MINIMUM_ARCHITECTURES,
            "measured_but_underpowered": [
                {"corpus": corpus, "metric": metric}
                for corpus, metric in self.underpowered
            ],
            "how_to_read_this": (
                "A rank correlation near one means the metric orders architectures the way "
                "perplexity already does, so reporting it adds a column and not a fact. A "
                "moderate correlation is the case for the battery: the metric and the "
                "objective disagree about which architectures are better, and the "
                "disagreement is what a single-metric evaluation cannot see."
            ),
            "why_sign_is_not_interpreted_as_quality": (
                "Each metric has its own direction — higher BLiMP is better, lower "
                "calibration error is better — so the sign of a correlation with perplexity "
                "follows from that direction and carries no separate meaning. Only the "
                "magnitude is read."
            ),
        }


def collect(manifests: Sequence[dict], corpus: str) -> dict[tuple[int, ...], dict]:
    """Battery measurements and the objective, by architecture, for one corpus."""
    rows: dict[tuple[int, ...], dict] = {}
    for manifest in manifests:
        if manifest.get("corpus") != corpus:
            continue
        battery = manifest.get("battery") or {}
        objectives = manifest.get("objectives") or {}
        if not battery or objectives.get("perplexity") is None:
            continue
        genes = tuple(manifest["genotype"]["values"])
        rows.setdefault(genes, {"perplexity": [], "battery": battery})
        rows[genes]["perplexity"].append(float(objectives["perplexity"]))
    return rows


def agreement(rows: dict[tuple[int, ...], dict], metric: str,
              corpus: str) -> MetricAgreement | None:
    """Rank correlation between the objective and one metric, or None if it was not run."""
    from scipy.stats import kendalltau

    key, _lower_is_better = HEADLINE[metric]
    perplexities, values = [], []
    for row in rows.values():
        entry = row["battery"].get(metric)
        if not isinstance(entry, dict) or key not in entry:
            continue
        value = entry[key]
        if not isinstance(value, (int, float)):
            continue
        perplexities.append(statistics.fmean(row["perplexity"]))
        values.append(float(value))

    if len(values) < MINIMUM_ARCHITECTURES:
        return None

    tau, p_value = kendalltau(perplexities, values)
    return MetricAgreement(
        metric=metric,
        key=key,
        corpus=corpus,
        architectures=len(values),
        kendall_tau=float(tau),
        kendall_p=float(p_value),
        spread=max(values) - min(values),
    )


def analyse(manifests: Sequence[dict],
            corpora: Sequence[str] | None = None) -> BatteryReport:
    """Every battery metric against the objective, on every corpus that measured it."""
    with_battery = [m for m in manifests if m.get("battery")]
    if not with_battery:
        raise NoBatteryRuns(
            "no manifest carries a battery. Submit the wave with `cli battery --submit`, "
            "then refresh the cache with `cli manifests`."
        )

    if corpora is None:
        corpora = sorted({m.get("corpus") for m in with_battery if m.get("corpus")})

    agreements, thin = [], []
    for corpus in corpora:
        rows = collect(with_battery, corpus)
        for metric in HEADLINE:
            found = agreement(rows, metric, corpus)
            if found is not None:
                agreements.append(found)
            elif any(isinstance(row["battery"].get(metric), dict)
                     and HEADLINE[metric][0] in row["battery"][metric]
                     for row in rows.values()):
                thin.append((corpus, metric))
    return BatteryReport(agreements=tuple(agreements), underpowered=tuple(thin))
