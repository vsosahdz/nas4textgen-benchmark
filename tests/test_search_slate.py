"""The algorithm slate, and why SMS-EMOA is the instrument rather than NSGA-II.

SMS-EMOA is steady-state and removes whichever individual contributes least to the
population's hypervolume, so that hypervolume cannot decrease. Under NSGA-II, truncation by
crowding distance carries no such guarantee, and a fall in its curve is uninterpretable —
it might be exploration or it might be a defect in the harness, and the curve cannot say.

These tests measure that difference rather than asserting it from theory.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymoo")

from nas4textgen.search import SLATE, BenchmarkTable, run_algorithm  # noqa: E402
from nas4textgen.space.cost import parameter_count  # noqa: E402
from nas4textgen.space.enumerate import enumerate_with_ceiling  # noqa: E402

SEEDS = range(6)


@pytest.fixture(scope="module")
def table() -> BenchmarkTable:
    rng = np.random.default_rng(0)
    rows = {}
    for genotype in enumerate_with_ceiling().genotypes:
        params = parameter_count(genotype)
        quality = 5000.0 / np.log10(max(10, params)) + rng.normal(0, 20)
        rows[genotype.as_tuple()] = np.array([float(params), quality])
    return BenchmarkTable(rows, ("params", "perplexity"), (True, True), source="synthetic")


def _drops(trace) -> int:
    deltas = np.diff(trace.population_hypervolume)
    return int((deltas < -1e-9).sum())


def test_sms_emoa_never_discards_population_hypervolume(table):
    """Its survival operator removes the least contribution, so it cannot."""
    for seed in SEEDS:
        trace = run_algorithm(table, "sms-emoa", seed=seed, generations=15)
        assert _drops(trace) == 0, f"seed {seed} lost hypervolume"
        assert trace.population_is_monotone


def test_nsga2_does_discard_it_and_that_is_why_it_is_not_the_instrument(table):
    """Measured, not assumed. Crowding-distance truncation has no monotonicity guarantee,
    so its curve cannot distinguish exploration from a harness defect."""
    seeds_with_drops = sum(
        _drops(run_algorithm(table, "nsga2", seed=seed, generations=15)) > 0
        for seed in SEEDS
    )
    assert seeds_with_drops > 0, (
        "NSGA-II showed no non-monotonicity in any seed, which would undercut the stated "
        "reason for preferring SMS-EMOA as the convergence instrument"
    )


def test_a_monotonicity_violation_from_sms_emoa_is_raised_not_recorded(table, monkeypatch):
    """Where the guarantee holds, a violation is a bug in this code rather than a fact
    about the search. Returning such a trace would publish the bug as a measurement.

    Forced here by corrupting the curve after scoring, since a genuine violation is
    exactly what must never happen.
    """
    from nas4textgen.search import run as run_module
    from nas4textgen.search.scoring import score_trace as real_score_trace

    def falling(*args, **kwargs):
        trace = real_score_trace(*args, **kwargs)
        object.__setattr__(
            trace, "population_hypervolume", np.array([1.0, 0.9, 0.8])
        )
        return trace

    monkeypatch.setattr(run_module, "score_trace", falling)
    with pytest.raises(AssertionError, match="defect in the harness"):
        run_algorithm(table, "sms-emoa", seed=0, generations=5)


def test_algorithms_are_compared_at_equal_true_evaluations(table):
    """Final metrics reward whichever run was given the most budget, and in NAS the budget
    is the entire cost."""
    traces = {
        name: run_algorithm(table, name, seed=0, generations=15)
        for name in ("random", "sms-emoa", "nsga2")
    }
    spent = {name: int(t.evaluations[-1]) for name, t in traces.items()}
    assert len(set(spent.values())) > 1, "the runs did not spend the same budget"

    budget = min(spent.values())
    at_budget = {name: t.at_budget(budget) for name, t in traces.items()}
    assert all(np.isfinite(v) for v in at_budget.values())


def test_every_slate_entry_states_why_it_is_there(table):
    """A comparison assembled from whatever was available is not a designed comparison."""
    assert len(SLATE) >= 4
    for entry in SLATE:
        assert len(entry.role) > 40, f"{entry.name} has no stated role"
    monotone = [entry.name for entry in SLATE if entry.monotone_population]
    assert monotone == ["sms-emoa"]
