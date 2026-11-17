"""The structural gate: eight checks a search space must pass before compute is spent.

Every threshold has a stated origin — measured noise, literature, or arithmetic —
because a threshold chosen by taste is a threshold that will be moved when it becomes
inconvenient. Each check also corresponds to a defect the prior work shipped, which is
why the set is what it is rather than a generic tidiness sweep.

None of this requires a GPU or a trained model. The gate runs in seconds and its report
is the artifact that authorises a sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .cost import kv_cache_bytes, parameter_count
from .enumerate import (
    Enumeration,
    enumerate_with_ceiling,
    non_dominated,
    reference_point,
    reference_span,
)
from .genotype import GENES, LEVELS, decode, encode

#: A gene explaining more than this share of an objective's variance means the
#: objective is effectively that one gene. Origin: the prior parameter objective had
#: `e` at 0.90, and reduced to "minimise e" — one gene of seven.
MAX_SOBOL_INDEX = 0.70

#: Above this rank correlation two objectives are the same objective under different
#: names. Origin: FLOPs per token measured +0.965 against parameter count, the known
#: `FLOPs ~ 2 x params` identity.
MAX_OBJECTIVE_CORRELATION = 0.90

#: A cost objective that varies by less than this at matched parameter count adds
#: nothing the parameter count could not already express.
MIN_ISO_COST_SPREAD = 10.0

#: An axis spanning less than this fraction of its reference range cannot discriminate.
MIN_REFERENCE_SPAN = 0.01
MAX_REFERENCE_SPAN = 0.99


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    checks: list[Check]
    enumeration: Enumeration

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks)
        lines = [
            "STRUCTURAL GATE",
            "=" * (width + 46),
        ]
        for i, c in enumerate(self.checks, start=1):
            mark = "pass" if c.passed else "FAIL"
            lines.append(f"  {i}. {mark}  {c.name:<{width}}  {c.detail}")
        lines.append("=" * (width + 46))
        lines.append(
            "  gate PASSES — a sweep may be submitted"
            if self.passed
            else "  gate FAILS — no sweep may be submitted for this space"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "space": {
                "cardinality": len(self.enumeration),
                "total_before_ceiling": self.enumeration.total_before_ceiling,
                "parameter_ceiling": self.enumeration.ceiling,
                "excluded": self.enumeration.excluded,
                "excluded_fraction": round(self.enumeration.excluded_fraction, 6),
                "reference_context": self.enumeration.context,
            },
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "values": c.values}
                for c in self.checks
            ],
        }


def _sobol_first_order(levels: np.ndarray, values: np.ndarray) -> float:
    """First-order Sobol index over a full grid: ``Var(E[Y|X]) / Var(Y)``.

    Computed on log objectives, since the objectives span orders of magnitude and a
    variance decomposition on raw values would be dominated by the largest models.
    """
    total = values.var()
    if total == 0:
        return 0.0
    conditional = 0.0
    for value in np.unique(levels):
        mask = levels == value
        conditional += mask.mean() * (values[mask].mean() - values.mean()) ** 2
    return float(conditional / total)


def run_gate(
    ceiling: int | None = None,
    context: int | None = None,
) -> GateReport:
    """Run every check and return the report.

    ``ceiling`` is left unset by default: the gate examines the *space*, and applying
    the sweep's parameter ceiling first would hide properties of the region the ceiling
    excludes.
    """
    enumeration = (
        enumerate_with_ceiling(ceiling=ceiling)
        if context is None
        else enumerate_with_ceiling(ceiling=ceiling, context=context)
    )
    space = enumeration.genotypes
    objectives = enumeration.objectives
    params, kv = objectives[:, 0], objectives[:, 1]
    checks: list[Check] = []

    # 1 — Cardinality and enumerability.
    enumerable = len(space) < 10_000
    checks.append(
        Check(
            "cardinality and enumerability",
            True,
            f"{len(space):,} architectures; "
            + (
                "enumerable, so the exact front is available and no search "
                "algorithm is required to produce it"
                if enumerable
                else "beyond comfortable enumeration; a search algorithm is warranted"
            ),
            {"cardinality": len(space), "enumerable": enumerable},
        )
    )

    # 2 — Injectivity. Origin: the prior encoding admitted e = 8.0017.
    round_trip = [decode(encode(g)) == g for g in space]
    checks.append(
        Check(
            "encoding is injective",
            all(round_trip),
            f"decode(encode(g)) == g for {sum(round_trip):,}/{len(space):,}",
            {"round_trip_failures": int(len(space) - sum(round_trip))},
        )
    )

    # 3 — No inert genes. Origin: the prior cost model ignored 4 of 7 genes.
    #
    # Reported two ways, because they answer different questions and a gene can look
    # negligible on one while mattering on the other. `ffn` is the case in point: its
    # first-order Sobol index rounds to 0.00 because that statistic measures variance
    # explained across the whole space, which the ranges of `e` (64 to 2048) and `n`
    # (2 to 24) dominate — yet holding the other genes fixed, moving `ffn` from 2 to 4
    # changes the parameter count by roughly a sixth. Global share and local effect are
    # not the same measurement, and only the second answers "is this gene inert".
    effects: dict[str, dict[str, float]] = {}
    inert: list[str] = []
    for gene in GENES:
        relative: dict[str, list[float]] = {"params": [], "kv_bytes": []}
        for g in space:
            for level in LEVELS[gene]:
                if level == getattr(g, gene):
                    continue
                candidate = type(g)(**{**{k: getattr(g, k) for k in GENES}, gene: level})
                if not candidate.is_valid():
                    continue
                for label, fn in (("params", parameter_count), ("kv_bytes", kv_cache_bytes)):
                    base = fn(g)
                    if base:
                        relative[label].append(abs(fn(candidate) - base) / base)
        summary = {
            label: round(float(np.median(values)), 5) if values else 0.0
            for label, values in relative.items()
        }
        effects[gene] = summary
        if max(summary.values()) == 0.0:
            inert.append(gene)
    weakest = min(effects.items(), key=lambda kv_: max(kv_[1].values()))
    checks.append(
        Check(
            "every gene moves an objective",
            not inert,
            "no inert genes; weakest is "
            f"{weakest[0]} at a median relative effect of "
            f"{max(weakest[1].values()):.1%}"
            if not inert
            else f"inert: {', '.join(inert)}",
            {"inert_genes": inert, "median_relative_effect": effects},
        )
    )

    # 4 — Objective decomposition. Origin: `e` explained 0.90 of the prior objective.
    sobol: dict[str, dict[str, float]] = {}
    worst = 0.0
    worst_label = ""
    for label, values in (("params", params), ("kv_bytes", kv)):
        logged = np.log10(values.astype(np.float64))
        per_gene = {
            gene: round(
                _sobol_first_order(np.array([getattr(g, gene) for g in space]), logged), 3
            )
            for gene in GENES
        }
        sobol[label] = per_gene
        top_gene, top = max(per_gene.items(), key=lambda kv_: kv_[1])
        if top > worst:
            worst, worst_label = top, f"{label}:{top_gene}"
    checks.append(
        Check(
            "no single gene dominates an objective",
            worst <= MAX_SOBOL_INDEX,
            f"largest first-order Sobol index {worst:.2f} ({worst_label}), "
            f"threshold {MAX_SOBOL_INDEX}",
            {"sobol_first_order": sobol, "threshold": MAX_SOBOL_INDEX},
        )
    )

    # 5 — Objectives are distinct. Origin: FLOPs/token measured +0.965 against params.
    rho = float(spearmanr(params, kv).statistic)
    checks.append(
        Check(
            "cost objectives are not duplicates",
            abs(rho) < MAX_OBJECTIVE_CORRELATION,
            f"rho(params, kv_bytes) = {rho:+.3f}, threshold {MAX_OBJECTIVE_CORRELATION}",
            {"spearman": round(rho, 4), "threshold": MAX_OBJECTIVE_CORRELATION},
        )
    )

    # 6 — Conflict at matched cost. Cost objectives never conflict among themselves —
    # everything shrinks as the model shrinks — so the front of costs alone is a single
    # point. The conflict that justifies the formulation lives at matched parameters.
    bands: dict[str, float] = {}
    edges = [1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 1e10]
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (params >= lo) & (params < hi)
        if mask.sum() > 5:
            bands[f"{lo/1e6:.0f}M-{hi/1e6:.0f}M"] = round(
                float(kv[mask].max() / kv[mask].min()), 1
            )
    best_spread = max(bands.values()) if bands else 0.0
    front_of_costs = len(non_dominated(objectives))
    checks.append(
        Check(
            "objectives conflict at matched cost",
            best_spread >= MIN_ISO_COST_SPREAD,
            f"KV-cache spread up to {best_spread:.0f}x within a parameter band "
            f"(cost-only front is {front_of_costs} point"
            f"{'s' if front_of_costs != 1 else ''}, as expected)",
            {
                "iso_parameter_spread": bands,
                "cost_only_front_size": front_of_costs,
                "threshold": MIN_ISO_COST_SPREAD,
            },
        )
    )

    # 7 — Repair produces valid genotypes for arbitrary continuous input.
    rng = np.random.default_rng(0)
    repaired = [decode(rng.random(len(GENES))) for _ in range(3000)]
    valid = sum(g.is_valid() for g in repaired)
    stable = sum(decode(encode(g)) == g for g in repaired)
    checks.append(
        Check(
            "repair yields valid, stable genotypes",
            valid == len(repaired) and stable == len(repaired),
            f"{valid:,}/{len(repaired):,} valid, {stable:,}/{len(repaired):,} stable "
            "under a further round trip",
            {"valid": valid, "stable": stable, "trials": len(repaired)},
        )
    )

    # 8 — Reference point. Origin: the prior [1e9, 0.0] failed to dominate 12.3% of
    # the space, invalidating those hypervolume contributions.
    reference = reference_point(objectives)
    dominated = int(np.all(objectives <= reference, axis=1).sum())
    span = reference_span(objectives, reference)
    span_ok = bool(np.all((span >= MIN_REFERENCE_SPAN) & (span <= MAX_REFERENCE_SPAN)))
    checks.append(
        Check(
            "reference point dominates and discriminates",
            dominated == len(space) and span_ok,
            f"dominates {dominated:,}/{len(space):,}; axis spans "
            + ", ".join(f"{s:.2f}" for s in span),
            {
                "reference_point": [float(x) for x in reference],
                "dominated": dominated,
                "axis_span": [round(float(s), 4) for s in span],
            },
        )
    )

    return GateReport(checks=checks, enumeration=enumeration)
